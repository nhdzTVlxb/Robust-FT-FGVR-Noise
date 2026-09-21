"""Training and evaluation for cosine heads on cached features."""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robustft.models import CosineClassifier


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def stratified_split(labels: torch.Tensor, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    lab = labels.numpy()
    train_parts, val_parts = [], []
    for cls in np.unique(lab):
        idx = np.nonzero(lab == cls)[0]
        rng.shuffle(idx)
        requested = max(1, int(round(len(idx) * val_frac)))
        n_val = min(len(idx) - 1, requested) if len(idx) > 1 else 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    return torch.from_numpy(train_idx).long(), torch.from_numpy(val_idx).long()


def stratified_three_way_split(
    labels: torch.Tensor, val_frac: float, holdout_frac: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified train/val/holdout split (val_frac=holdout_frac=0.1 -> 8:1:1).

    The holdout partition is carved out first and must stay untouched by every
    label-driven step including checkpoint selection; val is then split from
    the remainder so its overall fraction still equals val_frac.
    """
    trainval_idx, holdout_idx = stratified_split(labels, holdout_frac, seed)
    rel_frac = val_frac / (1.0 - holdout_frac)
    tr_rel, va_rel = stratified_split(labels[trainval_idx], rel_frac, seed + 1)
    return trainval_idx[tr_rel], trainval_idx[va_rel], holdout_idx


def soft_ce(logits: torch.Tensor, target_probs: torch.Tensor, sample_w: torch.Tensor | None = None) -> torch.Tensor:
    loss = -(target_probs * logits.log_softmax(dim=1)).sum(1)
    if sample_w is not None:
        loss = loss * sample_w
        return loss.sum() / sample_w.sum().clamp_min(1e-6)
    return loss.mean()


def elr_regularizer(probs: torch.Tensor, target: torch.Tensor, lam: float) -> torch.Tensor:
    """Early-Learning Regularization anchor term (Liu et al., 2020).

    L_ELR = lam * mean_i( log(1 - <p_i, t_i>) ), added to the (cross-entropy) loss.
    `probs` is the current softmax prediction (carries grad); `target` is the
    per-sample running EMA target (detached). Minimising the term drives
    <p_i, t_i> -> 1, i.e. keeps predictions aligned with the early-learning target
    and resists later memorisation of noisy labels. Returns a scalar.
    """
    inner = (probs * target).sum(dim=1)
    return lam * torch.log((1.0 - inner).clamp_min(1e-4)).mean()


def trust_project_grad(
    grad: torch.Tensor, trust: torch.Tensor, mode: str = "conflict", eps: float = 1e-12,
) -> torch.Tensor:
    """Trust-aligned Gradient Projection (TrustCLIP, ACM MM 2025), adapted to a
    single-optimizer LoRA setup.

    Project a flattened gradient ``grad`` with respect to a trusted (clean-sample)
    gradient direction ``trust``:

    * ``mode="conflict"`` (PCGrad-style, Yu 2020): if ``grad`` conflicts with the
      trusted direction (negative inner product), remove that conflicting
      component; otherwise leave ``grad`` untouched. Keeps the aligned/orthogonal
      signal and neutralises only the noise-driven part that fights the trusted
      direction.
    * ``mode="onto"`` (GPM-style, Saha 2021): keep only ``grad``'s projection
      inside ``span(trust)``.

    A near-zero ``trust`` is a no-op for ``"conflict"`` and returns zeros for
    ``"onto"``. Returns a tensor with the same shape/dtype as ``grad``.
    """
    tn = torch.dot(trust, trust)
    if tn <= eps:
        return grad if mode == "conflict" else torch.zeros_like(grad)
    coef = torch.dot(grad, trust) / tn
    if mode == "conflict":
        return grad - coef * trust if coef < 0 else grad
    if mode == "onto":
        return coef * trust
    raise ValueError(f"unknown mode: {mode!r}")


def select_trusted_idx(weights: torch.Tensor, frac: float, min_count: int) -> torch.Tensor:
    """Pick the trusted (clean) subset of a batch for Trust-aligned Gradient
    Projection: the top ``frac`` of samples by per-sample weight, considering only
    positive-weight samples, requiring at least ``min_count``.

    Robust to the weight distribution (an absolute threshold is brittle because
    weight = min_w + (1-min_w)*reliability and reliability is a product of
    imperfect signals that rarely approaches 1). If fewer than ``min_count``
    samples have positive weight, returns an empty LongTensor (skip projection
    this step). Returns 1-D LongTensor of selected indices into ``weights``.
    """
    pos = (weights > 0).nonzero(as_tuple=True)[0]
    if pos.numel() < min_count:
        return weights.new_empty(0, dtype=torch.long)
    k = max(min_count, math.ceil(frac * pos.numel()))
    k = min(k, pos.numel())
    topi = torch.topk(weights[pos], k).indices
    return pos[topi]


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    t = torch.full((labels.size(0), num_classes), smoothing / (num_classes - 1), device=labels.device)
    t.scatter_(1, labels[:, None], 1.0 - smoothing)
    return t


def train_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    indices: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    epochs: int = 15,
    batch_size: int = 8192,
    lr: float = 5e-3,
    weight_decay: float = 1e-2,
    smoothing: float = 0.1,
    dropout: float = 0.1,
    mixup_alpha: float = 0.0,
    ema_decay: float = 0.0,
    init_state: dict | None = None,
    device: torch.device = torch.device("cuda"),
    verbose: bool = False,
) -> CosineClassifier:
    """targets: hard labels (long) or soft distributions (float, (N, C))."""
    model = CosineClassifier(features.size(1), num_classes, dropout=dropout).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    ema_model = None
    if ema_decay > 0:
        ema_model = CosineClassifier(features.size(1), num_classes, dropout=dropout).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)
    soft_targets = targets.dim() == 2
    n = indices.numel()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = indices[torch.randperm(n, device=device)]
        tot_loss, tot_seen = 0.0, 0
        for start in range(0, n, batch_size):
            bidx = perm[start : start + batch_size]
            xb = features[bidx]
            if soft_targets:
                tb = targets[bidx]
            else:
                tb = smooth_one_hot(targets[bidx], num_classes, smoothing)
            wb = sample_weights[bidx] if sample_weights is not None else None
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm2 = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm2]
                tb = lam * tb + (1.0 - lam) * tb[perm2]
                if wb is not None:
                    wb = lam * wb + (1.0 - lam) * wb[perm2]
            opt.zero_grad(set_to_none=True)
            loss = soft_ce(model(xb), tb, wb)
            loss.backward()
            opt.step()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.mul_(ema_decay).add_(pm, alpha=1.0 - ema_decay)
            tot_loss += loss.item() * bidx.numel()
            tot_seen += bidx.numel()
        sched.step()
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs} loss={tot_loss / max(1, tot_seen):.4f}")
    return ema_model if ema_model is not None else model


@torch.inference_mode()
def evaluate(model: CosineClassifier, feats: torch.Tensor, labels: torch.Tensor, batch: int = 16384) -> float:
    model.eval()
    correct = 0
    for start in range(0, feats.size(0), batch):
        logits = model(feats[start : start + batch])
        correct += (logits.argmax(1) == labels[start : start + batch]).sum().item()
    return correct / feats.size(0)


@torch.inference_mode()
def per_sample_stats(model: CosineClassifier, feats: torch.Tensor, labels: torch.Tensor, batch: int = 16384):
    """Returns (loss, prob_of_label, argmax, max_prob) per sample."""
    model.eval()
    losses, p_label, preds, p_max = [], [], [], []
    for start in range(0, feats.size(0), batch):
        logits = model(feats[start : start + batch])
        yb = labels[start : start + batch]
        logp = logits.log_softmax(1)
        losses.append(F.nll_loss(logp, yb, reduction="none"))
        probs = logp.exp()
        p_label.append(probs.gather(1, yb[:, None]).squeeze(1))
        mx = probs.max(1)
        preds.append(mx.indices)
        p_max.append(mx.values)
    return (torch.cat(losses), torch.cat(p_label), torch.cat(preds), torch.cat(p_max))


@torch.inference_mode()
def teacher_stats(model: CosineClassifier, features: torch.Tensor, labels: torch.Tensor, batch: int = 16384):
    """Returns (prob_of_label, argmax, max_prob, top1-top2 margin) per sample."""
    model.eval()
    p_label, preds, pmax, margins = [], [], [], []
    for start in range(0, features.size(0), batch):
        logits = model(features[start : start + batch])
        probs = logits.softmax(1)
        top2 = probs.topk(2, dim=1)
        yb = labels[start : start + batch]
        p_label.append(probs.gather(1, yb[:, None]).squeeze(1))
        preds.append(top2.indices[:, 0])
        pmax.append(top2.values[:, 0])
        margins.append(top2.values[:, 0] - top2.values[:, 1])
    return tuple(torch.cat(x) for x in (p_label, preds, pmax, margins))


def banded_eval(model: CosineClassifier, feats: torch.Tensor, labels: torch.Tensor,
                agreement: torch.Tensor) -> dict:
    """Val accuracy inside kNN-agreement bands; mid band is the model-selection metric."""
    bands = {
        "noisy_all": torch.ones_like(labels, dtype=torch.bool),
        "low_lt03": agreement < 0.3,
        "mid_03_06": (agreement >= 0.3) & (agreement < 0.6),
        "high_ge06": agreement >= 0.6,
    }
    return {name: round(evaluate(model, feats[m], labels[m]), 4) for name, m in bands.items() if int(m.sum()) > 0}
