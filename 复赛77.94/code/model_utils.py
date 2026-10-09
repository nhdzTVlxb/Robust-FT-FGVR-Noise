from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ============================================================================
# Data
# ============================================================================
def list_samples(root_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """按类别名和文件名排序，得到稳定的 (path, label) 清单。"""
    root = Path(root_dir)
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    skipped_empty = 0

    for class_name in class_names:
        class_dir = root / class_name
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                continue
            if path.stat().st_size == 0:
                skipped_empty += 1
                continue
            samples.append((str(path), class_to_idx[class_name]))

    if skipped_empty:
        print(f"[Warning] skipped {skipped_empty} empty image(s)")
    return samples, class_names


def build_train_val_split(
    train_dir: str,
    val_split: float = 0.0,
    seed: int = 42,
    min_train_per_class: int = 2,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[str]]:
    """按类别划分开发验证集；val_split=0 时返回全量训练集。"""
    samples, class_names = list_samples(train_dir)
    if val_split <= 0:
        return samples, [], class_names

    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    rng = random.Random(seed)

    for label, indices in by_class.items():
        rng.shuffle(indices)
        n = len(indices)
        if n < 3:
            n_val = 0
        else:
            min_train = min(min_train_per_class, n)
            n_val = min(max(1, int(n * val_split)), n - min_train)
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        class_names,
    )


class FineGrainedDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform=None,
        return_index: bool = False,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot decode training image: {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        if self.return_index:
            return image, label, index
        return image, label


class TestDataset(Dataset):
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.files = sorted(
            p
            for p in self.test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not self.files:
            raise RuntimeError(f"No test images found in {self.test_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot decode test image: {path}") from exc
        return image, path.name


def train_transform(cfg, image_size: int):
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(cfg.crop_min, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandAugment(
                num_ops=cfg.randaugment_num_ops,
                magnitude=cfg.randaugment_magnitude,
            ),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def val_transform(image_size: int):
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def make_loaders(cfg, train_samples, val_samples, actual_size):
    train_loader = DataLoader(
        FineGrainedDataset(
            train_samples,
            train_transform(cfg, actual_size),
            return_index=True,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # At 512px, keep validation batches at training size to avoid a
    # validation-only memory spike (the old 2x rule would make this 192).
    val_batch_size = cfg.batch_size if actual_size >= 512 else cfg.batch_size * 2
    val_loader = (
        DataLoader(
            FineGrainedDataset(val_samples, val_transform(actual_size)),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if val_samples
        else None
    )
    return train_loader, val_loader


# ============================================================================
# Model and LoRA
# ============================================================================
class GCELoss(nn.Module):
    """保留历史接口；A0 训练主流程不调用它。"""

    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        p_y = p.gather(1, targets.view(-1, 1)).squeeze(1).clamp(1e-7, 1 - 1e-7)
        return (1.0 - p_y**self.q) / self.q


class HybridLoss(nn.Module):
    """保留历史接口；A0 实际损失是 soft CE + feature distillation。"""

    def __init__(
        self,
        num_classes: int,
        ce_epochs: int = 5,
        gce_q: float = 0.7,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.gce = GCELoss(q=gce_q)
        self.ce_epochs = ce_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, logits, targets):
        if self.current_epoch <= self.ce_epochs:
            return self.ce(logits, targets)
        transition = min(1.0, (self.current_epoch - self.ce_epochs) / 3.0)
        return (1.0 - transition) * self.ce(logits, targets) + transition * self.gce(
            logits, targets
        ).mean()


def interpolate_pos_embed(vision_model, target_size: int = 448) -> int:
    """把 CLIP 的 7x7 位置编码插值到可用的方形 patch 网格。"""
    patch_size = int(vision_model.config.patch_size)
    grid = int(math.ceil(target_size / patch_size))
    actual_size = grid * patch_size

    weight = vision_model.embeddings.position_embedding.weight.data
    original_dim = weight.dim()
    pos = weight.squeeze(0) if original_dim == 3 else weight

    cls_token = pos[:1]
    patch_tokens = pos[1:]
    old_grid = int(math.sqrt(patch_tokens.shape[0]))
    if old_grid * old_grid != patch_tokens.shape[0]:
        raise ValueError(f"Invalid position embedding shape: {tuple(pos.shape)}")

    patch_tokens = patch_tokens.reshape(old_grid, old_grid, -1)
    patch_tokens = patch_tokens.permute(2, 0, 1).unsqueeze(0)
    patch_tokens = F.interpolate(
        patch_tokens,
        size=(grid, grid),
        mode="bicubic",
        align_corners=False,
    )
    patch_tokens = patch_tokens.squeeze(0).permute(1, 2, 0).reshape(
        grid * grid, -1
    )
    new_pos = torch.cat([cls_token, patch_tokens], dim=0)

    if original_dim == 3:
        new_pos = new_pos.unsqueeze(0)
    vision_model.embeddings.position_embedding.weight.data = new_pos
    vision_model.embeddings.position_embedding.num_embeddings = new_pos.shape[1] if original_dim == 3 else new_pos.shape[0]
    vision_model.embeddings.position_ids = torch.arange(
        new_pos.shape[1] if original_dim == 3 else new_pos.shape[0]
    ).unsqueeze(0)
    vision_model.config.image_size = actual_size
    vision_model.embeddings.image_size = actual_size
    print(f"[Model] position grid: {old_grid}x{old_grid} -> {grid}x{grid} ({actual_size}px)")
    return actual_size


def build_teacher_model(clip_model: str, image_size: int):
    from transformers import CLIPVisionModel

    teacher = CLIPVisionModel.from_pretrained(clip_model)
    actual_size = interpolate_pos_embed(teacher, image_size) if image_size > 224 else 224
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, actual_size


def build_clip_model(
    clip_model: str,
    num_classes: int,
    image_size: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: Sequence[str],
):
    """构造 HF CLIPVisionModel + PEFT LoRA + 线性分类头。"""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import CLIPVisionModel

    vision = CLIPVisionModel.from_pretrained(clip_model)
    actual_size = interpolate_pos_embed(vision, image_size) if image_size > 224 else 224
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_targets),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    vision = get_peft_model(vision, lora_config)
    head = nn.Linear(vision.config.hidden_size, num_classes)
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)

    trainable_vision = sum(
        parameter.numel() for parameter in vision.parameters() if parameter.requires_grad
    )
    print(f"[Model] hidden_size={vision.config.hidden_size}, image_size={actual_size}")
    print(f"[Model] LoRA rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"[Model] LoRA targets={list(lora_targets)}")
    print(f"[Model] trainable LoRA params={trainable_vision:,}")
    print(f"[Model] trainable classifier params={sum(p.numel() for p in head.parameters()):,}")
    return vision, head, actual_size


def clip_forward(vision, head, images, return_features: bool = False):
    base = vision.base_model if hasattr(vision, "base_model") else vision
    outputs = base(pixel_values=images)
    features = outputs.pooler_output
    logits = head(features)
    return (logits, features) if return_features else logits


class CosineClassifier(nn.Module):
    """保留历史可选头；A0 默认使用线性头。"""

    def __init__(self, in_features, num_classes, scale_init=20.0, learnable_scale=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.trunc_normal_(self.weight, std=0.02)
        log_scale = torch.tensor(math.log(scale_init), dtype=torch.float32)
        if learnable_scale:
            self.logit_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("logit_scale", log_scale)

    def forward(self, features):
        features = F.normalize(features.float(), dim=-1)
        weight = F.normalize(self.weight.float(), dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        return F.linear(features, weight) * scale


# ============================================================================
# Optimizer, EMA, augmentation and loss helpers
# ============================================================================
def build_optimizer(vision, head, lr: float, weight_decay: float):
    params = list(vision.parameters()) + list(head.parameters())
    decay, no_decay = [], []
    for parameter in params:
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.dim() < 2 else decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(optimizer, warmup_epochs, num_epochs, steps_per_epoch):
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * steps_per_epoch,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, (num_epochs - warmup_epochs) * steps_per_epoch),
    )
    return SequentialLR(
        optimizer,
        [warmup, cosine],
        milestones=[warmup_epochs * steps_per_epoch],
    )


def rand_bbox(size, lam):
    width, height = size[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha, num_classes, device):
    """保留历史实现：每个 batch 必做 MixUp 或 CutMix，返回 soft labels。"""
    batch_size = images.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    use_cutmix = (
        cutmix_alpha > 0
        and torch.rand(1, device=device).item() < 0.5
        if mixup_alpha > 0
        else cutmix_alpha > 0
    )
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha > 0:
        draw_a = torch.distributions.Beta(alpha, alpha).sample().item()
        draw_b = torch.distributions.Beta(alpha, alpha).sample().item()
        lam = max(draw_a, 1.0 - draw_b)
    else:
        lam = 1.0

    permutation = torch.randperm(batch_size, device=device)
    if use_cutmix:
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[
            permutation, :, y1:y2, x1:x2
        ]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(images.size(-1) * images.size(-2)))
    else:
        images = lam * images + (1.0 - lam) * images[permutation]

    soft_targets = lam * one_hot + (1.0 - lam) * one_hot[permutation]
    return images, soft_targets


def mixup_cutmix_with_meta(
    images,
    targets,
    mixup_alpha,
    cutmix_alpha,
    num_classes,
    device,
):
    """MixUp/CutMix plus the metadata needed for per-sample loss ranking."""
    batch_size = images.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    permutation = torch.randperm(batch_size, device=device)

    use_cutmix = (
        cutmix_alpha > 0
        and torch.rand(1, device=device).item() < 0.5
        if mixup_alpha > 0
        else cutmix_alpha > 0
    )
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha > 0 and batch_size >= 2:
        draw_a = torch.distributions.Beta(alpha, alpha).sample().item()
        draw_b = torch.distributions.Beta(alpha, alpha).sample().item()
        lam = max(draw_a, 1.0 - draw_b)
    else:
        lam = 1.0
        permutation = torch.arange(batch_size, device=device)

    if use_cutmix and alpha > 0 and batch_size >= 2:
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[
            permutation, :, y1:y2, x1:x2
        ]
        lam = 1.0 - (
            (x2 - x1) * (y2 - y1)
            / float(images.size(-1) * images.size(-2))
        )
    elif alpha > 0 and batch_size >= 2:
        images = lam * images + (1.0 - lam) * images[permutation]

    soft_targets = lam * one_hot + (1.0 - lam) * one_hot[permutation]
    return images, soft_targets, permutation, float(lam), bool(alpha > 0 and batch_size >= 2)


class ModelEMA:
    """EMA over trainable parameters.

    Frozen CLIP parameters never change, so shadowing them would make every
    soft-label step needlessly copy hundreds of megabytes.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema = {
            key: value.detach().clone().float()
            for key, value in model.named_parameters()
            if value.requires_grad
        }
        self.decay = decay
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            self.ema[key].mul_(self.decay).add_(
                value.detach().float(), alpha=1.0 - self.decay
            )

    def apply_to(self, model: nn.Module):
        with torch.no_grad():
            for key, value in model.named_parameters():
                if key in self.ema:
                    value.copy_(self.ema[key].to(device=value.device, dtype=value.dtype))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self._backup = {
            key: value.detach().clone()
            for key, value in model.named_parameters()
            if key in self.ema
        }
        for key, value in model.named_parameters():
            if key in self.ema:
                value.copy_(self.ema[key].to(device=value.device, dtype=value.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup:
            for key, value in model.named_parameters():
                backup = self._backup.get(key)
                if backup is not None:
                    value.copy_(backup)
            self._backup.clear()


def soft_cross_entropy(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def soft_cross_entropy_per_sample(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)


def kl_to_soft_targets_per_sample(logits, teacher_logits, temperature=1.0):
    temperature = float(temperature)
    student_log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        * (temperature**2)
    )


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def current_clean_ratio(cfg, epoch: int) -> float:
    if not getattr(cfg, "use_clean_ratio_schedule", True):
        return float(getattr(cfg, "clean_ratio", 0.8))
    warmup = int(getattr(cfg, "small_loss_warmup_epochs", 8))
    if epoch <= warmup:
        return float(getattr(cfg, "clean_ratio", 0.8))
    if epoch <= 10:
        return float(getattr(cfg, "clean_ratio_epoch_6_10", 0.9))
    if epoch <= 25:
        return float(getattr(cfg, "clean_ratio_epoch_11_25", 0.9))
    return float(getattr(cfg, "clean_ratio_epoch_after_25", 0.85))


def update_small_loss_weights(
    loss_sum: torch.Tensor,
    seen: torch.Tensor,
    previous_loss: torch.Tensor,
    targets: torch.Tensor,
    previous_weights: torch.Tensor,
    cfg,
    epoch: int,
):
    """Rank per-sample losses within each class and downweight high-loss samples."""
    updated_loss = previous_loss.clone()
    observed = seen > 0
    updated_loss[observed] = loss_sum[observed] / seen[observed]

    clean_ratio = current_clean_ratio(cfg, epoch)
    clean_weight = float(getattr(cfg, "clean_weight", 1.0))
    noisy_weight = float(getattr(cfg, "noisy_weight", 0.3))
    new_weights = torch.full_like(previous_weights, noisy_weight)

    for cls in torch.unique(targets).tolist():
        indices = torch.nonzero(targets == int(cls), as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        keep = max(1, min(indices.numel(), int(math.ceil(indices.numel() * clean_ratio))))
        order = torch.argsort(updated_loss[indices], stable=True)
        new_weights[indices[order[:keep]]] = clean_weight

    return new_weights, updated_loss


# ============================================================================
# Metadata and compliance checks
# ============================================================================
def submission_ids(class_names: Sequence[str], class_map: Optional[str] = None) -> List[str]:
    if class_map:
        mapping = json.loads(Path(class_map).read_text(encoding="utf-8"))
        if set(mapping) != set(class_names):
            raise ValueError("class-map must contain exactly every class folder")
        raw = [str(mapping[name]) for name in class_names]
    else:
        raw = list(class_names)
    if not all(value.isascii() and value.isdigit() and 0 <= int(value) <= 9999 for value in raw):
        raise ValueError("Class folders must be numeric IDs or provide --class-map")
    values = [f"{int(value):04d}" for value in raw]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate submission IDs")
    return values


def inspect_model(path: str):
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision = config.get("vision_config", config)
    expected = {"patch_size": 32, "hidden_size": 768, "num_hidden_layers": 12}
    if any(vision.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected official CLIP ViT-B/32")
    if not any((root / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
        raise FileNotFoundError(f"Missing model weights in {root}")
    return config


def environment():
    dependencies = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "numpy",
        "Pillow",
        "huggingface-hub",
        "safetensors",
        "accelerate",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "packages": dependencies,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def atomic_json(path: Path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def data_signature(train_samples, val_samples, class_names, labels, data_root):
    relative = lambda samples: [
        [str(Path(path).resolve().relative_to(data_root)), int(label)]
        for path, label in samples
    ]
    manifest = {
        "classes": list(class_names),
        "submission_ids": list(labels),
        "train": relative(train_samples),
        "validation": relative(val_samples),
    }
    signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return manifest, signature
