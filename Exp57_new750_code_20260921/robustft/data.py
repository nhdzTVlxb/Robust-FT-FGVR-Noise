"""Datasets and transforms.

`ImageFile.LOAD_TRUNCATED_IMAGES` is set at import time: the competition data
contains truncated files, and Windows DataLoader workers re-import this module
on spawn, so the setting must live at module level here.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch.nn as nn
from PIL import Image, ImageFile
from timm.data import create_transform, resolve_data_config
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RandomResizeShortSide:
    """Resize the short side to one randomly selected training scale."""

    def __init__(self, sizes: list[int]):
        if not sizes or any(int(size) <= 0 for size in sizes):
            raise ValueError("training resize scales must be positive integers")
        self.sizes = tuple(int(size) for size in sizes)

    def __call__(self, image: Image.Image) -> Image.Image:
        size = random.choice(self.sizes)
        return transforms.functional.resize(
            image,
            size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )


class LabeledImageDataset(Dataset):
    """(image, label) pairs for feature extraction over the train set."""

    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


class UnlabeledImageDataset(Dataset):
    """(image, filename) pairs for feature extraction over the test set."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), Path(path).name


class IndexedImageDataset(Dataset):
    """(image, dataset index) pairs; used by LoRA training and inference."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img), idx


def sample_items(items: list, max_items: int | None, seed: int) -> list:
    if max_items is None or max_items >= len(items):
        return items
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in indices]


def load_train_samples(train_dir: Path, max_items: int | None, seed: int) -> tuple[list[tuple[str, int]], list[str]]:
    base = ImageFolder(str(train_dir))
    samples = sample_items(list(base.samples), max_items, seed)
    return samples, list(base.classes)


def load_test_paths(test_dir: Path, max_items: int | None, seed: int) -> list[str]:
    paths = sorted(
        str(p)
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return sample_items(paths, max_items, seed)


def build_extract_transform(model: nn.Module):
    """Deterministic transform used for frozen-feature extraction (main.py)."""
    cfg = resolve_data_config(model=model, use_test_size=True)
    return create_transform(
        input_size=cfg["input_size"],
        is_training=False,
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
        crop_pct=cfg.get("crop_pct"),
        crop_mode=cfg.get("crop_mode"),
    )


def build_finetune_transforms(model: nn.Module, crop_min_scale: float = 0.8,
                              img_size: int = 224, randaug: bool = False,
                              train_resize_scales: list[int] | None = None):
    """(train, eval) transform pair used for LoRA fine-tuning."""
    cfg = resolve_data_config(model=model)
    mean, std = cfg["mean"], cfg["std"]
    aug = []
    if train_resize_scales:
        aug.append(RandomResizeShortSide(train_resize_scales))
    aug.extend([
        transforms.RandomResizedCrop(img_size, scale=(crop_min_scale, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ])
    if randaug:
        aug.append(transforms.RandAugment(num_ops=2, magnitude=7))
    else:
        aug.append(transforms.ColorJitter(0.2, 0.2, 0.1))
    train_tf = transforms.Compose(aug + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf
