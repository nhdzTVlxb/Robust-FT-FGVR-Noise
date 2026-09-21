"""Frozen-feature extraction and on-disk cache handling."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from robustft.engine import maybe_autocast


def extract_features(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_flip: bool = True,
    desc: str = "extract",
):
    if len(dataset) == 0:
        raise ValueError(f"{desc}: dataset is empty")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feat_dim = getattr(model, "num_features", None)
    if feat_dim is None:
        raise RuntimeError("backbone does not expose num_features")
    features = torch.empty((len(dataset), feat_dim), dtype=torch.float16)
    labels = None
    names: list[str] | None = None
    offset = 0
    model.eval()
    with torch.inference_mode():
        for images, meta in tqdm(loader, desc=desc, total=len(loader)):
            images = images.to(device, non_blocking=True)
            with maybe_autocast(device):
                feat = model(images)
                if tta_flip:
                    flipped = torch.flip(images, dims=[3])
                    feat_flip = model(flipped)
                    feat = 0.5 * (F.normalize(feat.float(), dim=-1) + F.normalize(feat_flip.float(), dim=-1))
                else:
                    feat = feat.float()
            feat = F.normalize(feat, dim=-1).cpu().to(torch.float16)
            bs = feat.size(0)
            features[offset : offset + bs] = feat
            if torch.is_tensor(meta):
                if labels is None:
                    labels = torch.empty((len(dataset),), dtype=torch.long)
                labels[offset : offset + bs] = meta
            else:
                if names is None:
                    names = []
                names.extend(list(meta))
            offset += bs
    if labels is not None:
        return features, labels
    return features, names or []


def save_tensor_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_tensor_cache(path: Path):
    return torch.load(path, map_location="cpu")


def cache_matches(payload: dict, expected_names: list[str], name_key: str, expected_meta: dict | None = None) -> bool:
    if "features" not in payload or name_key not in payload:
        return False
    if int(payload["features"].size(0)) != len(expected_names):
        return False
    if list(payload[name_key]) != expected_names:
        return False
    return expected_meta is None or all(payload.get(k) == v for k, v in expected_meta.items())
