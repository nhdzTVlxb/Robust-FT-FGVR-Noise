import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from config import Config
from model_utils import (
    build_clip_lora,
    build_original_clip,
    ModelEMA,
    LabelSmoothingCELoss,
    FeatureConsistencyLoss,
)


# ============================================================================
# Basic utils
# ============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_epoch_log(cfg: Config, text: str):
    """
    只记录每轮训练结束后的指标行。
    不记录 tqdm，不重定向 stdout。
    """
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cfg.log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def label_smoothing_ce_per_sample(logits, targets, smoothing=0.1):
    """
    返回每个样本自己的 CE loss，方便做 reweight 和记录小损失样本。
    """
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)

    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

    loss = -(true_dist * log_probs).sum(dim=1)
    return loss


def weighted_mean(loss_vec, weight_vec):
    """
    按样本权重求平均，除以权重和，避免 loss 尺度因为降权而整体变小。
    """
    weight_sum = weight_vec.sum().clamp_min(1e-12)
    return (loss_vec * weight_vec).sum() / weight_sum


def get_current_clean_ratio(cfg: Config, epoch_num: int):
    if not getattr(cfg, "use_clean_ratio_schedule", False):
        return cfg.clean_ratio

    if epoch_num <= cfg.reweight_warmup_epochs:
        return cfg.clean_ratio
    if epoch_num <= 10:
        return getattr(cfg, "clean_ratio_epoch_6_10", 0.9)
    if epoch_num <= 20:
        return getattr(cfg, "clean_ratio_epoch_11_20", 0.85)
    return getattr(cfg, "clean_ratio_epoch_after_20", 0.8)


def hard_ce_per_sample(logits, targets):
    return F.cross_entropy(logits, targets, reduction="none")


def kl_to_soft_targets_per_sample(logits, teacher_logits, temperature=1.0):
    log_probs = F.log_softmax(logits / temperature, dim=1)
    with torch.no_grad():
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_probs, soft_targets, reduction="none").sum(dim=1) * (temperature ** 2)


def rand_bbox(width, height, lam):
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    return int(x1), int(y1), int(x2), int(y2)


def apply_mixup_cutmix(images, labels, weights, cfg: Config):
    if (
        not getattr(cfg, "use_mix_aug", False)
        or random.random() >= getattr(cfg, "mix_aug_prob", 0.3)
        or images.size(0) < 2
    ):
        return images, labels, labels, weights, 1.0, False

    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    use_cutmix = random.random() < 0.5
    mixed = images.clone()

    if use_cutmix:
        lam = np.random.beta(getattr(cfg, "cutmix_alpha", 0.5), getattr(cfg, "cutmix_alpha", 0.5))
        _, _, h, w = images.shape
        x1, y1, x2, y2 = rand_bbox(w, h, lam)
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))
    else:
        lam = np.random.beta(getattr(cfg, "mixup_alpha", 0.1), getattr(cfg, "mixup_alpha", 0.1))
        mixed = lam * images + (1.0 - lam) * images[perm]

    labels_a = labels
    labels_b = labels[perm]
    mixed_weights = lam * weights + (1.0 - lam) * weights[perm]

    return mixed, labels_a, labels_b, mixed_weights, float(lam), True


@torch.no_grad()
def ema_teacher_logits(model, classifier, ema, images, cfg: Config):
    if ema is None:
        return None

    was_training_model = model.training
    was_training_classifier = classifier.training

    ema.apply_shadow(model, classifier)
    model.eval()
    classifier.eval()

    with torch.amp.autocast("cuda", enabled=cfg.use_amp):
        features = model.encode_image(images)
        logits = classifier(features)

    ema.restore(model, classifier)

    if was_training_model:
        model.train()
    if was_training_classifier:
        classifier.train()

    return logits.detach()


def set_lora_trainable(lora_params, trainable: bool):
    for p in lora_params:
        p.requires_grad_(trainable)


# ============================================================================
# Dataset with index
# ============================================================================
class IndexedImageFolder(ImageFolder):
    """
    返回 image, label, index。
    index 是 ImageFolder 原始样本下标，用来记录每个样本的 loss。
    """

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


# ============================================================================
# Transforms
# ============================================================================
def get_train_transform(cfg: Config):
    return T.Compose([
        T.RandomResizedCrop(
            cfg.image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.333),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
        T.RandomErasing(
            p=0.2,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])


def get_val_transform(cfg: Config, size=None):
    size = size or cfg.image_size
    resize_size = int(size * 256 / 224)

    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


# ============================================================================
# Data
# ============================================================================
def split_indices(n_total: int, val_ratio: float, seed: int):
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(n_total * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices, val_indices


def pil_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels


def dataloader_worker_kwargs(cfg: Config):
    if cfg.num_workers <= 0:
        return {}

    return {
        "persistent_workers": getattr(cfg, "persistent_workers", True),
        "prefetch_factor": getattr(cfg, "prefetch_factor", 4),
    }


def create_data_loaders(cfg: Config):
    base_dataset = ImageFolder(str(cfg.train_dir))
    n_total = len(base_dataset)

    if n_total == 0:
        raise RuntimeError(f"训练目录为空: {cfg.train_dir}")

    train_indices, val_indices = split_indices(n_total, cfg.val_ratio, cfg.seed)

    if getattr(cfg, "train_on_all", False):
        train_indices = list(range(n_total))
        print("[Data] train_on_all=True: train loader uses all samples; val split is monitor-only")

    train_full = IndexedImageFolder(str(cfg.train_dir), transform=get_train_transform(cfg))
    val_full = ImageFolder(str(cfg.train_dir), transform=get_val_transform(cfg))
    val_tta_full = ImageFolder(str(cfg.train_dir), transform=None)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    val_tta_dataset = Subset(val_tta_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        **dataloader_worker_kwargs(cfg),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        **dataloader_worker_kwargs(cfg),
    )

    val_tta_loader = DataLoader(
        val_tta_dataset,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pil_collate_fn,
        **dataloader_worker_kwargs(cfg),
    )

    print(f"[Data] Train dir: {cfg.train_dir}")
    print(f"[Data] Total: {n_total} | Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"[Data] Classes found: {len(base_dataset.classes)} | Config classes: {cfg.num_classes}")

    if len(base_dataset.classes) != cfg.num_classes:
        print(f"[Warning] ImageFolder classes={len(base_dataset.classes)}, cfg.num_classes={cfg.num_classes}")

    return train_loader, val_loader, val_tta_loader, train_indices, base_dataset.targets


# ============================================================================
# CLIP Prototype Cleaning
# ============================================================================
@torch.no_grad()
def compute_clip_prototype_weights(cfg: Config, orig_model, train_indices, sample_targets, device):
    """
    用原始冻结 CLIP 特征做每类 prototype，生成每个样本的先验清洗权重。
    注意：正常验证阶段只使用 train_indices 计算 prototype，避免 val 泄漏。
    """
    n_total = len(sample_targets)
    cache_path = getattr(cfg, "prototype_cache_path", None)

    if cache_path is not None:
        cache_path = cfg.prototype_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not getattr(cfg, "prototype_force_recompute", False):
            print(f"[Prototype] Loading cached weights: {cache_path}")
            state = torch.load(cache_path, map_location="cpu")
            if isinstance(state, dict) and "prototype_weights" in state:
                weights = state["prototype_weights"].float()
            else:
                weights = state.float()
            if weights.numel() == n_total:
                print(
                    f"[Prototype] loaded: mean={weights[train_indices].mean():.4f}, "
                    f"min={weights[train_indices].min():.4f}, max={weights[train_indices].max():.4f}"
                )
                return weights
            print("[Prototype Warning] cache size mismatch, recomputing...")

    image_size = getattr(cfg, "prototype_image_size", cfg.image_size)
    batch_size = getattr(cfg, "prototype_batch_size", max(1, cfg.batch_size * 2))

    proto_dataset_full = IndexedImageFolder(
        str(cfg.train_dir),
        transform=get_val_transform(cfg, size=image_size),
    )
    proto_dataset = Subset(proto_dataset_full, train_indices)
    proto_loader = DataLoader(
        proto_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        **dataloader_worker_kwargs(cfg),
    )

    orig_model.eval()
    features = torch.zeros(n_total, 512, dtype=torch.float32)
    seen = torch.zeros(n_total, dtype=torch.bool)

    print(f"[Prototype] Extracting original CLIP features: size={image_size}, batch={batch_size}")
    for images, labels, indices in tqdm(proto_loader, desc="[Prototype Feature]", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            feats = orig_model.encode_image(images)
        feats = F.normalize(feats.float(), dim=1).cpu()
        features[indices] = feats
        seen[indices] = True

    sample_targets_np = np.array(sample_targets)
    train_indices_np = np.array(train_indices)

    prototypes = torch.zeros(cfg.num_classes, 512, dtype=torch.float32)
    valid_class = torch.zeros(cfg.num_classes, dtype=torch.bool)

    # 1) 初始 prototype：每类所有训练样本均值
    for cls_id in range(cfg.num_classes):
        cls_indices = train_indices_np[sample_targets_np[train_indices_np] == cls_id]
        cls_indices = [idx for idx in cls_indices if seen[idx]]
        if len(cls_indices) == 0:
            continue
        proto = features[cls_indices].mean(dim=0)
        prototypes[cls_id] = F.normalize(proto, dim=0)
        valid_class[cls_id] = True

    # 2) 可选 refined prototype：先剔除最不像本类的一部分，再重算
    if getattr(cfg, "prototype_refine", True):
        refine_ratio = getattr(cfg, "prototype_refine_ratio", 0.8)
        refined = torch.zeros_like(prototypes)
        refined_valid = torch.zeros_like(valid_class)

        for cls_id in range(cfg.num_classes):
            if not valid_class[cls_id]:
                continue
            cls_indices = train_indices_np[sample_targets_np[train_indices_np] == cls_id]
            cls_indices = [idx for idx in cls_indices if seen[idx]]
            if len(cls_indices) == 0:
                continue

            sims = features[cls_indices] @ prototypes[cls_id]
            order = torch.argsort(sims, descending=True)
            keep_num = int(math.ceil(len(cls_indices) * refine_ratio))
            keep_num = max(1, min(len(cls_indices), keep_num))
            keep_indices = np.array(cls_indices)[order[:keep_num].numpy()]

            proto = features[keep_indices].mean(dim=0)
            refined[cls_id] = F.normalize(proto, dim=0)
            refined_valid[cls_id] = True

        prototypes[refined_valid] = refined[refined_valid]
        valid_class = refined_valid

    # 3) 每类内部按相似度排名生成 prototype_weight
    weights = torch.ones(n_total, dtype=torch.float32)
    clean_ratio = getattr(cfg, "prototype_clean_ratio", 0.85)
    clean_weight = getattr(cfg, "prototype_clean_weight", 1.0)
    noisy_weight = getattr(cfg, "prototype_noisy_weight", 0.5)

    total_down = 0
    for cls_id in range(cfg.num_classes):
        if not valid_class[cls_id]:
            continue
        cls_indices = train_indices_np[sample_targets_np[train_indices_np] == cls_id]
        cls_indices = [idx for idx in cls_indices if seen[idx]]
        if len(cls_indices) == 0:
            continue

        sims = features[cls_indices] @ prototypes[cls_id]
        order = torch.argsort(sims, descending=True)
        num_clean = int(math.ceil(len(cls_indices) * clean_ratio))
        num_clean = max(1, min(len(cls_indices), num_clean))

        cls_indices_np = np.array(cls_indices)
        clean_indices = cls_indices_np[order[:num_clean].numpy()]
        noisy_indices = cls_indices_np[order[num_clean:].numpy()]

        weights[clean_indices] = clean_weight
        if len(noisy_indices) > 0:
            weights[noisy_indices] = noisy_weight
            total_down += len(noisy_indices)

    train_w = weights[train_indices]
    print(
        f"[Prototype] done | clean_ratio={clean_ratio} | down_weighted={total_down} | "
        f"mean={train_w.mean():.4f} | min={train_w.min():.4f} | max={train_w.max():.4f}"
    )

    if cache_path is not None:
        torch.save(
            {
                "prototype_weights": weights,
                "train_indices": torch.tensor(train_indices, dtype=torch.long),
                "prototype_clean_ratio": clean_ratio,
                "prototype_noisy_weight": noisy_weight,
                "prototype_image_size": image_size,
            },
            cache_path,
        )
        print(f"[Prototype] Saved weights: {cache_path}")

    return weights


# ============================================================================
# Small-loss reweight
# ============================================================================
def init_sample_weights(n_total, device):
    return torch.ones(n_total, dtype=torch.float32, device=device)


def update_sample_weights_by_loss(
    cfg: Config,
    sample_losses_cpu: torch.Tensor,
    sample_seen_cpu: torch.Tensor,
    sample_targets,
    train_indices,
    epoch_num=None,
):
    """
    根据当前 epoch 记录到的每个样本 loss，生成下一轮训练要用的 sample_weights。
    每个类别内部：
        loss 最小 clean_ratio 的样本 weight=1.0
        剩下样本 weight=noisy_weight
    """
    n_total = len(sample_targets)
    new_weights = torch.ones(n_total, dtype=torch.float32)

    sample_targets_np = np.array(sample_targets)
    train_indices_np = np.array(train_indices)

    valid_mask = sample_seen_cpu[train_indices_np] > 0
    valid_train_indices = train_indices_np[valid_mask.numpy()]

    if len(valid_train_indices) == 0:
        return new_weights

    for cls_id in range(cfg.num_classes):
        cls_indices = valid_train_indices[sample_targets_np[valid_train_indices] == cls_id]

        if len(cls_indices) == 0:
            continue

        cls_losses = sample_losses_cpu[cls_indices]
        order = torch.argsort(cls_losses, descending=False)

        num_cls = len(cls_indices)
        clean_ratio = get_current_clean_ratio(cfg, epoch_num) if epoch_num is not None else cfg.clean_ratio
        num_clean = int(math.ceil(num_cls * clean_ratio))
        num_clean = max(1, min(num_cls, num_clean))

        clean_indices = cls_indices[order[:num_clean].numpy()]
        noisy_indices = cls_indices[order[num_clean:].numpy()]

        new_weights[clean_indices] = cfg.clean_weight

        if len(noisy_indices) > 0:
            new_weights[noisy_indices] = cfg.noisy_weight

    return new_weights


def summarize_sample_weights(sample_weights_cpu, train_indices):
    train_w = sample_weights_cpu[train_indices]
    clean_count = int((train_w >= 0.999).sum().item())
    noisy_count = int((train_w < 0.999).sum().item())
    mean_weight = float(train_w.mean().item())

    return clean_count, noisy_count, mean_weight


# ============================================================================
# Train / Val
# ============================================================================
def train_epoch(
    model,
    classifier,
    loader,
    optimizer,
    scaler,
    consistency_loss,
    epoch,
    cfg,
    device,
    orig_model,
    sample_weights,
    ema=None,
):
    model.train()
    classifier.train()

    total_loss = 0.0
    total_ce = 0.0
    total_cons = 0.0
    correct = 0
    total = 0

    total_weight = 0.0

    n_total = sample_weights.numel()
    sample_loss_sum = torch.zeros(n_total, dtype=torch.float32)
    sample_seen = torch.zeros(n_total, dtype=torch.float32)

    reweight_active = (
        getattr(cfg, "use_loss_reweight", False)
        and (epoch + 1) > cfg.reweight_warmup_epochs
    )

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]", leave=False)

    for images, labels, indices in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        batch_weights = sample_weights[indices]
        if not reweight_active:
            batch_weights = torch.ones_like(batch_weights)

        train_images, labels_a, labels_b, effective_weights, mix_lam, mixed = apply_mixup_cutmix(
            images=images,
            labels=labels,
            weights=batch_weights,
            cfg=cfg,
        )

        teacher_logits = None
        if (
            getattr(cfg, "use_noisy_ema_soft", False)
            and reweight_active
            and ema is not None
        ):
            teacher_logits = ema_teacher_logits(
                model=model,
                classifier=classifier,
                ema=ema,
                images=train_images,
                cfg=cfg,
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            features = model.encode_image(train_images)
            logits = classifier(features)

            with torch.no_grad():
                orig_features = orig_model.encode_image(train_images)

            ce_a = label_smoothing_ce_per_sample(
                logits=logits,
                targets=labels_a,
                smoothing=cfg.label_smoothing,
            )
            if mixed:
                ce_b = label_smoothing_ce_per_sample(
                    logits=logits,
                    targets=labels_b,
                    smoothing=cfg.label_smoothing,
                )
                hard_ce_vec = mix_lam * ce_a + (1.0 - mix_lam) * ce_b
            else:
                hard_ce_vec = ce_a

            loss_vec = hard_ce_vec

            if teacher_logits is not None:
                kl_vec = kl_to_soft_targets_per_sample(
                    logits=logits,
                    teacher_logits=teacher_logits,
                    temperature=getattr(cfg, "ema_soft_temperature", 1.0),
                )

                denom = max(1e-6, 1.0 - float(cfg.noisy_weight))
                noisy_strength = ((1.0 - effective_weights) / denom).clamp(0.0, 1.0)
                noisy_loss_vec = (
                    getattr(cfg, "noisy_hard_ce_weight", 0.3) * hard_ce_vec
                    + getattr(cfg, "noisy_ema_kl_weight", 0.7) * kl_vec
                )
                loss_vec = (1.0 - noisy_strength) * hard_ce_vec + noisy_strength * noisy_loss_vec

            loss_ce = weighted_mean(loss_vec, effective_weights)

            loss_cons = consistency_loss(features, orig_features)
            loss = loss_ce + loss_cons

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad],
            max_norm=cfg.grad_clip,
        )

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model, classifier)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += loss_ce.item() * batch_size
        total_cons += loss_cons.item() * batch_size
        total_weight += effective_weights.detach().sum().item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += batch_size

        # 记录当前 epoch 每个样本的 unweighted CE，用于下一轮 reweight
        with torch.no_grad():
            indices_cpu = indices.detach().cpu()
            ce_cpu = hard_ce_vec.detach().float().cpu()

            sample_loss_sum[indices_cpu] += ce_cpu
            sample_seen[indices_cpu] += 1.0

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "ce": f"{total_ce / total:.4f}",
            "cons": f"{total_cons / total:.4f}",
            "acc": f"{correct / total:.4f}",
            "rw": "on" if reweight_active else "off",
            "w": f"{total_weight / total:.3f}",
        })

    avg_sample_loss = sample_loss_sum / sample_seen.clamp_min(1.0)

    return {
        "loss": total_loss / total,
        "ce": total_ce / total,
        "cons": total_cons / total,
        "acc": correct / total,
        "reweight_active": reweight_active,
        "mean_batch_weight": total_weight / total,
        "sample_losses_cpu": avg_sample_loss,
        "sample_seen_cpu": sample_seen,
    }


@torch.no_grad()
def validate(model, classifier, loader, criterion, cfg, device, desc="[Val]"):
    model.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            features = model.encode_image(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


@torch.no_grad()
def validate_ema_tta(model, classifier, loader, cfg, device):
    """
    注意：这个函数应在 ema.apply_shadow(model, classifier) 之后调用。
    因此这里计算的是 EMA 权重 + TTA 的准确率。
    """
    model.eval()
    classifier.eval()

    transforms = {
        size: get_val_transform(cfg, size=size)
        for size in cfg.tta_sizes
    }

    correct = 0
    total = 0

    for pil_images, labels in tqdm(loader, desc="[Val EMA+TTA]", leave=False):
        labels = labels.to(device, non_blocking=True)

        logits_sum = None
        n_view = 0

        for size in cfg.tta_sizes:
            transform = transforms[size]

            images = torch.stack([transform(img.convert("RGB")) for img in pil_images], dim=0)
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                features = model.encode_image(images)
                logits = classifier(features)

            logits_sum = logits if logits_sum is None else logits_sum + logits
            n_view += 1

            if cfg.tta_hflip:
                images_flip = torch.flip(images, dims=[3])

                with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                    features_flip = model.encode_image(images_flip)
                    logits_flip = classifier(features_flip)

                logits_sum = logits_sum + logits_flip
                n_view += 1

        logits_avg = logits_sum / max(1, n_view)
        pred = logits_avg.argmax(dim=1)

        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "acc": correct / total,
    }


# ============================================================================
# Checkpoint
# ============================================================================
def save_checkpoint(
    path,
    epoch,
    model,
    classifier,
    optimizer,
    scheduler,
    scaler,
    ema,
    cfg,
    train_metrics,
    val_metrics,
    val_ema_metrics=None,
    val_ema_tta_metrics=None,
    best_ema_acc=0.0,
    best_ema_tta_acc=0.0,
    best_ema_tta_epoch=-1,
    best_epoch=-1,
    best_metric_value=0.0,
    best_metrics_info=None,
    early_stop_counter=0,
    sample_weights=None,
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "cfg": vars(cfg),
        "train_metrics": {
            k: v for k, v in train_metrics.items()
            if k not in ["sample_losses_cpu", "sample_seen_cpu"]
        },
        "val_metrics": val_metrics,
        "val_ema_metrics": val_ema_metrics,
        "val_ema_tta_metrics": val_ema_tta_metrics,
        "best_ema_acc": best_ema_acc,
        "best_ema_tta_acc": best_ema_tta_acc,
        "best_ema_tta_epoch": best_ema_tta_epoch,
        "best_epoch": best_epoch,
        "best_metric_value": best_metric_value,
        "best_metrics_info": best_metrics_info,
        "early_stop_counter": early_stop_counter,
    }

    if ema is not None:
        checkpoint["ema_state"] = ema.state_dict()

    if sample_weights is not None:
        checkpoint["sample_weights"] = sample_weights.detach().cpu()

    torch.save(checkpoint, path)


# ============================================================================
# Args
# ============================================================================
def update_cfg_from_args(cfg, args):
    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    if args.image_size is not None:
        cfg.image_size = args.image_size

    if args.lr is not None:
        cfg.learning_rate = args.lr

    if args.lr_classifier is not None:
        cfg.lr_classifier = args.lr_classifier

    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay

    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs

    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing

    if args.lora_rank is not None:
        cfg.lora_rank = args.lora_rank

    if args.lora_alpha is not None:
        cfg.lora_alpha = args.lora_alpha

    if args.lora_target is not None:
        cfg.lora_target = args.lora_target

    if args.lora_last_n_layers is not None:
        cfg.lora_last_n_layers = args.lora_last_n_layers

    if args.val_ratio is not None:
        cfg.val_ratio = args.val_ratio

    if args.resume is not None:
        cfg.resume_from = args.resume

    if args.no_amp:
        cfg.use_amp = False

    if args.no_ema:
        cfg.use_ema = False

    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay

    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience

    if args.no_loss_reweight:
        cfg.use_loss_reweight = False

    if args.reweight_warmup_epochs is not None:
        cfg.reweight_warmup_epochs = args.reweight_warmup_epochs

    if args.clean_ratio is not None:
        cfg.clean_ratio = args.clean_ratio

    if args.noisy_weight is not None:
        cfg.noisy_weight = args.noisy_weight

    if args.no_ema_tta_val:
        cfg.use_ema_tta_val = False

    if args.tta_sizes is not None:
        cfg.tta_sizes = [int(x.strip()) for x in args.tta_sizes.split(",") if x.strip()]

    return cfg


def select_early_stop_metric(cfg: Config, val_acc, ema_acc, ema_tta_acc):
    metric_name = getattr(cfg, "early_stop_metric", "ema_acc")
    metric_values = {
        "val_acc": val_acc,
        "ema_acc": ema_acc,
        "ema_tta_acc": ema_tta_acc,
    }

    metric_value = metric_values.get(metric_name, None)
    if metric_value is None:
        print(f"[Warning] early_stop_metric={metric_name} is unavailable, fallback to ema_acc")
        metric_name = "ema_acc"
        metric_value = ema_acc

    return metric_name, metric_value


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--image_size", type=int, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_classifier", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)

    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_target", type=str, default=None)
    parser.add_argument("--lora_last_n_layers", type=int, default=None)

    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)

    parser.add_argument("--early_stop_patience", type=int, default=None)

    parser.add_argument("--no_loss_reweight", action="store_true")
    parser.add_argument("--reweight_warmup_epochs", type=int, default=None)
    parser.add_argument("--clean_ratio", type=float, default=None)
    parser.add_argument("--noisy_weight", type=float, default=None)

    parser.add_argument("--no_ema_tta_val", action="store_true")
    parser.add_argument("--tta_sizes", type=str, default=None)

    args = parser.parse_args()

    cfg = Config()
    cfg = update_cfg_from_args(cfg, args)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {device}")

    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU Memory] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"[Config] train_dir: {cfg.train_dir}")
    print(f"[Config] pretrained_path: {cfg.pretrained_path}")
    print(f"[Config] checkpoint_dir: {cfg.checkpoint_dir}")
    print(f"[Config] log_path: {cfg.log_path}")
    print(f"[Config] image_size: {cfg.image_size}")
    print(f"[Config] EMA: {cfg.use_ema}, decay={cfg.ema_decay}")
    print(f"[Config] EMA+TTA val: {cfg.use_ema_tta_val}, sizes={cfg.tta_sizes}, hflip={cfg.tta_hflip}")
    print(f"[Config] train_on_all: {getattr(cfg, 'train_on_all', False)}")
    print(f"[Config] Early stop: patience={cfg.early_stop_patience}, metric={cfg.early_stop_metric}")
    print(f"[Config] LoRA target: {cfg.lora_target}")
    print(f"[Config] LoRA last N layers: {getattr(cfg, 'lora_last_n_layers', 12)}")
    print(f"[Config] Head warmup epochs: {getattr(cfg, 'head_warmup_epochs', 0)}")
    print(f"[Config] Loss reweight: {cfg.use_loss_reweight}")
    print(f"[Config] Reweight warmup epochs: {cfg.reweight_warmup_epochs}")
    print(f"[Config] Clean ratio: {cfg.clean_ratio}")
    print(f"[Config] Clean/noisy weight: {cfg.clean_weight}/{cfg.noisy_weight}")
    print(f"[Config] Prototype cleaning: {getattr(cfg, 'use_prototype_cleaning', False)}")
    if getattr(cfg, "use_prototype_cleaning", False):
        print(
            f"[Config] Prototype: clean_ratio={getattr(cfg, 'prototype_clean_ratio', 'N/A')}, "
            f"noisy_weight={getattr(cfg, 'prototype_noisy_weight', 'N/A')}, "
            f"size={getattr(cfg, 'prototype_image_size', cfg.image_size)}"
        )

    model, classifier = build_clip_lora(cfg, device=device)
    model = model.to(device)
    classifier = classifier.to(device)

    orig_model = build_original_clip(cfg, device=device)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    classifier_params = list(classifier.parameters())
    head_warmup_epochs = getattr(cfg, "head_warmup_epochs", 0)

    optimizer = torch.optim.AdamW([
        {
            "params": lora_params,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": classifier_params,
            "lr": cfg.lr_classifier,
            "weight_decay": cfg.weight_decay,
        },
    ])

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return float(epoch + 1) / float(max(1, cfg.warmup_epochs))

        progress = float(epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_loader, val_loader, val_tta_loader, train_indices, sample_targets = create_data_loaders(cfg)

    criterion = LabelSmoothingCELoss(label_smoothing=cfg.label_smoothing)
    consistency_loss = FeatureConsistencyLoss(weight=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    ema = ModelEMA(model, classifier, decay=cfg.ema_decay) if cfg.use_ema else None

    n_total = len(sample_targets)
    prototype_weights_cpu = torch.ones(n_total, dtype=torch.float32)
    if getattr(cfg, "use_prototype_cleaning", False):
        prototype_weights_cpu = compute_clip_prototype_weights(
            cfg=cfg,
            orig_model=orig_model,
            train_indices=train_indices,
            sample_targets=sample_targets,
            device=device,
        )
    prototype_weights = prototype_weights_cpu.to(device)
    sample_weights = init_sample_weights(n_total, device=device) * prototype_weights

    start_epoch = 0

    best_ema_acc = 0.0
    best_ema_tta_acc = 0.0
    best_ema_tta_epoch = -1
    best_epoch = -1
    best_metric_value = 0.0
    best_metrics_info = None
    early_stop_counter = 0

    if cfg.resume_from:
        print(f"[Resume] Loading: {cfg.resume_from}")
        state = torch.load(cfg.resume_from, map_location=device)

        model.load_state_dict(state["model_state"], strict=False)
        classifier.load_state_dict(state["classifier_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])

        if "scaler_state" in state:
            scaler.load_state_dict(state["scaler_state"])

        if ema is not None and "ema_state" in state:
            ema.load_state_dict(state["ema_state"])

        if "sample_weights" in state:
            sample_weights = state["sample_weights"].to(device)

        start_epoch = state["epoch"] + 1

        best_ema_acc = state.get("best_ema_acc", 0.0)
        best_ema_tta_acc = state.get("best_ema_tta_acc", 0.0)
        best_ema_tta_epoch = state.get("best_ema_tta_epoch", -1)
        best_epoch = state.get("best_epoch", best_epoch)
        best_metric_value = state.get("best_metric_value", best_metric_value)
        best_metrics_info = state.get("best_metrics_info", best_metrics_info)
        early_stop_counter = state.get("early_stop_counter", early_stop_counter)

        print(
            f"[Resume] start_epoch={start_epoch}, "
            f"best_epoch={best_epoch}, "
            f"best_ema_acc={best_metric_value:.4f}, "
            f"early_stop_counter={early_stop_counter}"
        )

    print("\n[Start Training]")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LoRA LR: {cfg.learning_rate}")
    print(f"  Classifier LR: {cfg.lr_classifier}")
    print(f"  Label smoothing: {cfg.label_smoothing}")
    print(f"  Trainable params: {sum(p.numel() for p in lora_params) + sum(p.numel() for p in classifier_params):,}")
    print()

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        lora_trainable = (epoch + 1) > head_warmup_epochs
        set_lora_trainable(lora_params, lora_trainable)

        sample_weights_cpu_before = sample_weights.detach().cpu()
        clean_count, noisy_count, mean_sample_weight = summarize_sample_weights(
            sample_weights_cpu_before,
            train_indices,
        )

        train_metrics = train_epoch(
            model=model,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            consistency_loss=consistency_loss,
            epoch=epoch,
            cfg=cfg,
            device=device,
            orig_model=orig_model,
            sample_weights=sample_weights,
            ema=ema,
        )

        # 1. 普通模型单尺度验证
        val_metrics = validate(
            model=model,
            classifier=classifier,
            loader=val_loader,
            criterion=criterion,
            cfg=cfg,
            device=device,
            desc="[Val]",
        )

        val_ema_metrics = None
        val_ema_tta_metrics = None

        # 2. EMA 权重验证：单尺度 + TTA
        if ema is not None:
            ema.apply_shadow(model, classifier)

            val_ema_metrics = validate(
                model=model,
                classifier=classifier,
                loader=val_loader,
                criterion=criterion,
                cfg=cfg,
                device=device,
                desc="[Val EMA]",
            )

            if cfg.use_ema_tta_val:
                val_ema_tta_metrics = validate_ema_tta(
                    model=model,
                    classifier=classifier,
                    loader=val_tta_loader,
                    cfg=cfg,
                    device=device,
                )

            ema.restore(model, classifier)

        scheduler.step()

        lr_lora = optimizer.param_groups[0]["lr"]
        lr_cls = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - epoch_start

        val_acc = val_metrics["acc"]
        ema_acc = val_ema_metrics["acc"] if val_ema_metrics is not None else val_acc
        ema_tta_acc = val_ema_tta_metrics["acc"] if val_ema_tta_metrics is not None else None

        selected_metric_name, current_metric = select_early_stop_metric(
            cfg=cfg,
            val_acc=val_acc,
            ema_acc=ema_acc,
            ema_tta_acc=ema_tta_acc,
        )
        is_best = current_metric > best_metric_value

        if is_best:
            best_metric_value = current_metric
            best_ema_acc = ema_acc
            best_epoch = epoch + 1
            early_stop_counter = 0

            best_metrics_info = {
                "epoch": best_epoch,
                "best_metric": selected_metric_name,
                "best_metric_value": best_metric_value,

                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce"],
                "train_cons": train_metrics["cons"],
                "train_acc": train_metrics["acc"],
                "reweight_active": train_metrics["reweight_active"],
                "mean_batch_weight": train_metrics["mean_batch_weight"],

                "clean_count": clean_count,
                "noisy_count": noisy_count,
                "mean_sample_weight": mean_sample_weight,

                "val_loss": val_metrics["loss"],
                "val_acc": val_acc,

                "ema_loss": val_ema_metrics["loss"] if val_ema_metrics is not None else None,
                "ema_acc": ema_acc if val_ema_metrics is not None else None,

                "ema_tta_acc": ema_tta_acc,
            }

            save_checkpoint(
                path=cfg.checkpoint_dir / "checkpoint_best_ema.pt",
                epoch=epoch,
                model=model,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                cfg=cfg,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_ema_metrics=val_ema_metrics,
                val_ema_tta_metrics=val_ema_tta_metrics,
                best_ema_acc=best_ema_acc,
                best_ema_tta_acc=best_ema_tta_acc,
                best_ema_tta_epoch=best_ema_tta_epoch,
                best_epoch=best_epoch,
                best_metric_value=best_metric_value,
                best_metrics_info=best_metrics_info,
                early_stop_counter=early_stop_counter,
                sample_weights=sample_weights,
            )

            print(f"  >> Save best selected: epoch={epoch + 1}, {selected_metric_name}={best_metric_value:.4f}")

        else:
            early_stop_counter += 1

        if ema_tta_acc is not None and ema_tta_acc > best_ema_tta_acc:
            best_ema_tta_acc = ema_tta_acc
            best_ema_tta_epoch = epoch + 1
            tta_best_info = {
                "epoch": best_ema_tta_epoch,
                "best_metric": "ema_tta_acc",
                "best_metric_value": best_ema_tta_acc,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce"],
                "train_cons": train_metrics["cons"],
                "train_acc": train_metrics["acc"],
                "reweight_active": train_metrics["reweight_active"],
                "mean_batch_weight": train_metrics["mean_batch_weight"],
                "clean_count": clean_count,
                "noisy_count": noisy_count,
                "mean_sample_weight": mean_sample_weight,
                "val_loss": val_metrics["loss"],
                "val_acc": val_acc,
                "ema_loss": val_ema_metrics["loss"] if val_ema_metrics is not None else None,
                "ema_acc": ema_acc if val_ema_metrics is not None else None,
                "ema_tta_acc": ema_tta_acc,
            }

            save_checkpoint(
                path=cfg.checkpoint_dir / "checkpoint_best_ema_tta.pt",
                epoch=epoch,
                model=model,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                cfg=cfg,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_ema_metrics=val_ema_metrics,
                val_ema_tta_metrics=val_ema_tta_metrics,
                best_ema_acc=ema_acc,
                best_ema_tta_acc=best_ema_tta_acc,
                best_ema_tta_epoch=best_ema_tta_epoch,
                best_epoch=best_ema_tta_epoch,
                best_metric_value=best_ema_tta_acc,
                best_metrics_info=tta_best_info,
                early_stop_counter=early_stop_counter,
                sample_weights=sample_weights,
            )

            print(f"  >> Save best EMA+TTA: epoch={epoch + 1}, ema_tta_acc={best_ema_tta_acc:.4f}")

        epoch_line = (
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | "
            f"time={elapsed:.1f}s | "
            f"lr={lr_lora:.2e}/{lr_cls:.2e} | "
            f"lora_trainable={lora_trainable} | "
            f"reweight={'on' if train_metrics['reweight_active'] else 'off'} | "
            f"clean={clean_count} | noisy={noisy_count} | "
            f"mean_sample_w={mean_sample_weight:.3f} | "
            f"mean_batch_w={train_metrics['mean_batch_weight']:.3f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_ce={train_metrics['ce']:.4f} | "
            f"train_cons={train_metrics['cons']:.4f} | "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_acc:.4f}"
            + (f" | ema_loss={val_ema_metrics['loss']:.4f} | ema_acc={ema_acc:.4f}" if val_ema_metrics is not None else "")
            + (f" | ema_tta_acc={ema_tta_acc:.4f}" if ema_tta_acc is not None else "")
            + f" | best_epoch={best_epoch}"
            + f" | selected={selected_metric_name}:{current_metric:.4f}"
            + f" | best_metric={best_metric_value:.4f}"
            + f" | best_ema_tta_acc={best_ema_tta_acc:.4f}"
            + f" | early_stop={early_stop_counter}/{cfg.early_stop_patience}"
        )

        print(epoch_line)
        write_epoch_log(cfg, epoch_line)

        # 用本轮 sample loss 更新下一轮权重
        if getattr(cfg, "use_loss_reweight", False) and (epoch + 1) >= cfg.reweight_warmup_epochs:
            new_sample_weights_cpu = update_sample_weights_by_loss(
                cfg=cfg,
                sample_losses_cpu=train_metrics["sample_losses_cpu"],
                sample_seen_cpu=train_metrics["sample_seen_cpu"],
                sample_targets=sample_targets,
                train_indices=train_indices,
                epoch_num=epoch + 2,
            )
            # loss-based reweight 与 CLIP prototype 先验权重融合
            sample_weights = new_sample_weights_cpu.to(device) * prototype_weights

        save_checkpoint(
            path=cfg.checkpoint_dir / "checkpoint_last.pt",
            epoch=epoch,
            model=model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            cfg=cfg,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            val_ema_metrics=val_ema_metrics,
            val_ema_tta_metrics=val_ema_tta_metrics,
            best_ema_acc=best_ema_acc,
            best_ema_tta_acc=best_ema_tta_acc,
            best_ema_tta_epoch=best_ema_tta_epoch,
            best_epoch=best_epoch,
            best_metric_value=best_metric_value,
            best_metrics_info=best_metrics_info,
            early_stop_counter=early_stop_counter,
            sample_weights=sample_weights,
        )

        if early_stop_counter >= cfg.early_stop_patience:
            stop_line = (
                f"[Early Stop] No improvement on {selected_metric_name} for "
                f"{cfg.early_stop_patience} epochs. "
                f"Best epoch: {best_epoch}, "
                f"best {selected_metric_name}: {best_metric_value:.4f}"
            )
            print(stop_line)
            write_epoch_log(cfg, stop_line)
            break

    print("\n[Done]")

    if best_metrics_info is not None:
        print(f"  Best checkpoint metric: {best_metrics_info['best_metric']}")
        print(f"  Best epoch: {best_metrics_info['epoch']}")
        print(f"  Best metric value: {best_metrics_info['best_metric_value']:.4f}")
        print(f"  Train loss: {best_metrics_info['train_loss']:.4f}")
        print(f"  Train CE: {best_metrics_info['train_ce']:.4f}")
        print(f"  Train Cons: {best_metrics_info['train_cons']:.4f}")
        print(f"  Train acc: {best_metrics_info['train_acc']:.4f}")
        print(f"  Reweight active: {best_metrics_info['reweight_active']}")
        print(f"  Clean count: {best_metrics_info['clean_count']}")
        print(f"  Noisy count: {best_metrics_info['noisy_count']}")
        print(f"  Mean sample weight: {best_metrics_info['mean_sample_weight']:.4f}")
        print(f"  Val loss: {best_metrics_info['val_loss']:.4f}")
        print(f"  Val acc: {best_metrics_info['val_acc']:.4f}")

        if best_metrics_info["ema_acc"] is not None:
            print(f"  EMA loss: {best_metrics_info['ema_loss']:.4f}")
            print(f"  EMA acc: {best_metrics_info['ema_acc']:.4f}")

        if best_metrics_info["ema_tta_acc"] is not None:
            print(f"  EMA+TTA acc: {best_metrics_info['ema_tta_acc']:.4f}")

        final_best_line = (
            f"[Best] epoch={best_metrics_info['epoch']} | "
            f"metric={best_metrics_info['best_metric']} | "
            f"value={best_metrics_info['best_metric_value']:.4f} | "
            f"reweight={best_metrics_info['reweight_active']} | "
            f"clean={best_metrics_info['clean_count']} | "
            f"noisy={best_metrics_info['noisy_count']} | "
            f"mean_sample_w={best_metrics_info['mean_sample_weight']:.4f} | "
            f"train_loss={best_metrics_info['train_loss']:.4f} | "
            f"train_acc={best_metrics_info['train_acc']:.4f} | "
            f"val_loss={best_metrics_info['val_loss']:.4f} | "
            f"val_acc={best_metrics_info['val_acc']:.4f} | "
            f"ema_acc={best_metrics_info['ema_acc']} | "
            f"ema_tta_acc={best_metrics_info['ema_tta_acc']}"
        )

        write_epoch_log(cfg, final_best_line)

    print(f"  Best EMA acc: {best_ema_acc:.4f}")
    print(f"  Best EMA+TTA acc: {best_ema_tta_acc:.4f} at epoch {best_ema_tta_epoch}")
    print(f"  Checkpoints: {cfg.checkpoint_dir}")
    print(f"  Epoch log: {cfg.log_path}")


if __name__ == "__main__":
    main()
