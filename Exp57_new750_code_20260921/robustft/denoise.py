"""Noise handling: kNN label-agreement scoring, sample selection, pseudo-labels."""

from __future__ import annotations

import torch

from robustft.engine import smooth_one_hot
from robustft.models import CosineClassifier


@torch.inference_mode()
def knn_agreement(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int,
    exclude_self: bool,
    chunk: int = 2048,
) -> torch.Tensor:
    """Weighted fraction of k nearest gallery neighbours sharing the query label.

    Features must be L2-normalised. exclude_self assumes query == gallery row-aligned.
    """
    n = query.size(0)
    agreement = torch.empty(n, dtype=torch.float32)
    g = gallery.t().contiguous()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = query[start:end] @ g  # (b, N) fp16
        if exclude_self:
            rows = torch.arange(start, end, device=sim.device)
            sim[torch.arange(end - start, device=sim.device), rows] = -2.0
        topv, topi = sim.topk(k, dim=1)
        same = (gallery_labels[topi] == query_labels[start:end, None]).float()
        w = topv.float().clamp_min(0)
        agreement[start:end] = (same * w).sum(1) / w.sum(1).clamp_min(1e-6)
    return agreement


@torch.inference_mode()
def knn_majority_prediction(
    query: torch.Tensor,
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int,
    num_classes: int,
    *,
    exclude_self: bool,
    chunk: int = 2048,
) -> torch.Tensor:
    """Similarity-weighted kNN majority vote; returns predicted class per query."""
    predictions = torch.empty(query.size(0), dtype=torch.long, device=query.device)
    gallery_t = gallery.t().contiguous()
    for start in range(0, query.size(0), chunk):
        end = min(start + chunk, query.size(0))
        sim = query[start:end] @ gallery_t
        if exclude_self:
            rows = torch.arange(start, end, device=sim.device)
            sim[torch.arange(end - start, device=sim.device), rows] = -2.0
        topv, topi = sim.topk(k, dim=1)
        votes = torch.zeros((end - start, num_classes), device=sim.device)
        votes.scatter_add_(1, gallery_labels[topi], topv.float().clamp_min(0))
        predictions[start:end] = votes.argmax(1)
    return predictions


def fit_gmm_1d(x: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """2-component 1D GMM via EM. Returns posterior prob of the low-mean (clean) component."""
    x = x.float()
    mu = torch.tensor([x.quantile(0.25), x.quantile(0.75)], device=x.device)
    var = torch.full((2,), x.var().item() + 1e-6, device=x.device)
    pi = torch.tensor([0.5, 0.5], device=x.device)
    for _ in range(iters):
        log_p = (
            -0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]
            - 0.5 * var.log()[None, :]
            + pi.log()[None, :]
        )
        resp = log_p.softmax(dim=1)
        nk = resp.sum(0).clamp_min(1e-6)
        mu = (resp * x[:, None]).sum(0) / nk
        var = ((resp * (x[:, None] - mu[None, :]) ** 2).sum(0) / nk).clamp_min(1e-8)
        pi = nk / x.numel()
    clean_comp = int(mu.argmin())
    return resp[:, clean_comp]


@torch.inference_mode()
def gmm_divide_select(
    losses: torch.Tensor,
    confs: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    clean_thresh: float = 0.5,
    conf_gate: float = 0.95,
    clean_weight: torch.Tensor | float = 1.0,
    noisy_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DivideMix-style per-epoch clean/noisy re-selection from the *training
    model's* per-sample loss (not the static frozen-feature teacher).

    Mechanism (Li et al., 2020, "DivideMix"): early-learning makes correctly
    labelled samples have systematically lower loss than mislabelled ones; a
    2-component GMM on the loss separates them, and the split is recomputed each
    epoch so the clean set tracks the improving model (self-cleaning loop). This
    is what distinguishes it from a one-shot static relabel (which introduced
    noise here, see clrelabel 74.88): noisy samples are treated as *unlabelled*
    and only rejoin via a high-confidence pseudo-label gate, never as a hard
    wrong label.

    Args:
      losses   (N,) per-sample CE loss vs the ORIGINAL label, from the EMA model.
      confs    (N,) model max softmax prob.
      preds    (N,) model argmax class.
      labels   (N,) original (noisy) labels.
      clean_thresh  GMM clean-posterior >= this -> treated as clean.
      conf_gate     a noisy sample rejoins as pseudo-labelled only if conf >= this.
      clean_weight  per-sample (or scalar) loss weight for clean samples.
      noisy_weight  loss weight for confidently pseudo-labelled noisy samples.

    Returns (target_labels, weights, w_clean):
      clean:                 target=original label, weight=clean_weight.
      noisy & conf>=gate:    target=model pred,     weight=noisy_weight (unlabelled).
      noisy & conf<gate:     weight 0 (dropped this epoch; re-decided next epoch).
    """
    w_clean = fit_gmm_1d(losses)
    clean = w_clean >= clean_thresh
    target = labels.clone()
    weights = torch.zeros_like(losses, dtype=torch.float32)
    if isinstance(clean_weight, torch.Tensor):
        weights[clean] = clean_weight[clean].to(weights.dtype)
    else:
        weights[clean] = float(clean_weight)
    noisy_conf = (~clean) & (confs >= conf_gate)
    target[noisy_conf] = preds[noisy_conf]
    weights[noisy_conf] = float(noisy_weight)
    return target, weights, w_clean


@torch.inference_mode()
def confident_learning_keep(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confident Learning (Northcutt et al. 2021) clean-sample selection.

    Per-class self-confidence threshold t_j = mean predicted prob assigned to
    class j over samples *labelled* j. A sample is judged clean iff, among the
    classes whose predicted prob clears their own threshold, the most-confident
    one equals the sample's noisy label. Samples that clear no threshold are
    uncertain (kept, conservative). Mechanism is orthogonal to kNN agreement:
    it uses global per-class confidence structure, not local neighbour voting.

    Returns (keep_mask, cl_pred, cl_conf, has_above):
      cl_pred   - the confident class (relabel target for confident-wrong samples),
      cl_conf   - predicted prob at cl_pred (confidence weight for relabeling),
      has_above - whether any class cleared its threshold (False = uncertain).
    Confident-wrong = has_above & (cl_pred != label): the samples worth relabelling.
    """
    probs = probs.float()
    t = torch.zeros(num_classes, device=probs.device)
    for j in range(num_classes):
        m = labels == j
        if m.any():
            t[j] = probs[m, j].mean()
    above = probs >= t[None, :]
    cl_pred = probs.masked_fill(~above, -1.0).argmax(1)
    cl_conf = probs.gather(1, cl_pred.unsqueeze(1)).squeeze(1)
    has_above = above.any(1)
    keep = ((cl_pred == labels) & has_above) | (~has_above)
    return keep, cl_pred, cl_conf, has_above


@torch.inference_mode()
def sinkhorn_balance(probs: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Sinkhorn-Knopp: rescale a (N,C) prob matrix so column marginals are uniform
    (each class gets ~N/C mass), exploiting the known balanced test prior. Returns
    a refined per-sample distribution. Rows are renormalised to sum to 1."""
    p = probs.float().clamp_min(1e-8)
    n, c = p.shape
    col_target = n / c
    for _ in range(iters):
        p = p / p.sum(1, keepdim=True)                       # row-normalise
        p = p * (col_target / p.sum(0, keepdim=True).clamp_min(1e-8))  # col-balance
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


@torch.inference_mode()
def label_propagation(
    features: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    num_classes: int,
    k: int = 16,
    alpha: float = 0.8,
    iters: int = 20,
    chunk: int = 2048,
) -> torch.Tensor:
    """Iterative label propagation on the feature kNN graph. Anchor (trusted)
    samples inject their one-hot label each step; labels diffuse to neighbours.
    features L2-normalised. Returns (N,C) propagated label distribution."""
    n = labels.size(0)
    seed = torch.zeros(n, num_classes, device=features.device)
    seed[anchors, labels[anchors]] = 1.0
    z = seed.clone()
    g = features.t().contiguous()
    # precompute sparse-ish kNN (indices + weights) per chunk, reused each iter
    nbr_idx = torch.empty(n, k, dtype=torch.long, device=features.device)
    nbr_w = torch.empty(n, k, device=features.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = features[s:e] @ g
        sim[torch.arange(e - s, device=sim.device), torch.arange(s, e, device=sim.device)] = -2.0
        w, i = sim.topk(k, dim=1)
        nbr_idx[s:e], nbr_w[s:e] = i, w.float().clamp_min(0)
    nbr_w = nbr_w / nbr_w.sum(1, keepdim=True).clamp_min(1e-8)
    for _ in range(iters):
        agg = (z[nbr_idx] * nbr_w.unsqueeze(-1)).sum(1)      # neighbour-averaged
        z = alpha * agg + (1 - alpha) * seed
        z[anchors] = seed[anchors]                            # clamp anchors
    return z


def per_class_topk_keep(score: torch.Tensor, labels: torch.Tensor, num_classes: int, keep_ratio: float) -> torch.Tensor:
    """Keep top keep_ratio of each class by score; returns bool mask."""
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for cls in range(num_classes):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).squeeze(1)
        if cls_idx.numel() == 0:
            continue
        k = min(cls_idx.numel(), max(1, int(cls_idx.numel() * keep_ratio)))
        top = torch.topk(score[cls_idx], k=k, largest=True).indices
        keep[cls_idx[top]] = True
    return keep


def select_clean_indices(
    model: CosineClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy baseline selection: per-class top-k by classifier confidence."""
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        probs = logits.softmax(dim=1)
        conf = probs[torch.arange(labels.size(0), device=labels.device), labels]
    num_classes = logits.size(1)
    keep_chunks = []
    for cls in range(num_classes):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).squeeze(1)
        if cls_idx.numel() == 0:
            continue
        k = max(1, int(cls_idx.numel() * keep_ratio))
        k = min(k, cls_idx.numel())
        topk = torch.topk(conf[cls_idx], k=k, largest=True).indices
        keep_chunks.append(cls_idx[topk])
    keep_idx = torch.cat(keep_chunks) if keep_chunks else torch.empty(0, dtype=torch.long, device=labels.device)
    keep_idx = keep_idx[torch.argsort(keep_idx)]
    return keep_idx, conf


def make_soft_targets_with_pseudo(
    labels: torch.Tensor,
    keep_mask: torch.Tensor,
    teacher_preds: torch.Tensor,
    teacher_pmax: torch.Tensor,
    num_classes: int,
    smoothing: float,
    pseudo_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kept samples: smoothed original label, weight 1.
    Dropped samples with confident teacher: smoothed pseudo-label, weight = teacher confidence.
    Others: excluded (weight 0)."""
    n = labels.size(0)
    targets = smooth_one_hot(labels, num_classes, smoothing)
    weights = torch.zeros(n, device=labels.device)
    weights[keep_mask] = 1.0
    pseudo_mask = (~keep_mask) & (teacher_pmax >= pseudo_thresh)
    if pseudo_mask.any():
        targets[pseudo_mask] = smooth_one_hot(teacher_preds[pseudo_mask], num_classes, smoothing)
        weights[pseudo_mask] = teacher_pmax[pseudo_mask]
    used = torch.nonzero(weights > 0, as_tuple=False).squeeze(1)
    return targets, weights, used
