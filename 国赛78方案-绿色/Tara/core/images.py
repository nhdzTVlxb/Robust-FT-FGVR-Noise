import random
from pathlib import Path
from typing import List, Sequence, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
ASPECT_SPAN = (3 / 4, 4 / 3)
WINDOW_TRIES = 10

def decode(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))

def jpeg_files(folder: Path) -> List[Path]:
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and (not p.name.startswith('.'))))

def inspect_class_folders(root: Path) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    folders = sorted((p for p in Path(root).iterdir() if p.is_dir()))
    if any((not (len(p.name) == 4 and p.name.isdigit()) for p in folders)):
        raise ValueError('every class folder must carry a four-digit name')
    paths: List[Path] = []
    labels: List[int] = []
    for (k, folder) in enumerate(folders):
        files = jpeg_files(folder)
        paths += files
        labels += [k] * len(files)
    return (paths, np.asarray(labels, np.int64), np.asarray([int(p.name) for p in folders], np.int64))

def fit(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    k = min(img.shape[0] // max(rows, 1), img.shape[1] // max(cols, 1))
    img = cv2.resize(img, (img.shape[1] // k, img.shape[0] // k), interpolation=cv2.INTER_AREA) if k >= 2 else img
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_LINEAR if img.shape[0] * img.shape[1] > rows * cols else cv2.INTER_CUBIC)

def square_view(img: np.ndarray, side: int, long_cap: int=0) -> np.ndarray:
    (h, w) = img.shape[:2]
    n = min(h, w)
    sq = img[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n]
    if long_cap and max(h, w) > long_cap:
        e = max(1, int(round(n * long_cap / max(h, w))))
        sq = fit(sq, e, e)
    return fit(sq, side, side)

class EvaluationPhotoDataset(Dataset):

    def __init__(self, paths: Sequence[Path], side: int, long_cap: int=0) -> None:
        (self.paths, self.side, self.long_cap) = (list(paths), side, long_cap)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return torch.from_numpy(square_view(decode(self.paths[i]), self.side, self.long_cap))

class AugmentedPhotoDataset(Dataset):

    def __init__(self, paths: Sequence[Path], labels: np.ndarray, cfg, seed: int) -> None:
        (self.paths, self.labels, self.cfg, self.seed, self.epoch) = (list(paths), labels, cfg, seed, 0)
        self._jitter = None

    def __len__(self) -> int:
        return len(self.paths)

    def _policy(self):
        if self._jitter is None:
            from timm.data.auto_augment import rand_augment_transform
            self._jitter = rand_augment_transform(self.cfg.rand_policy, {'img_mean': tuple((round(255 * m) for m in MEAN))})
        return self._jitter

    def _light_color(self, piece: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        color = self.cfg.color
        if rng.random() >= color.rate:
            return piece
        out = piece.astype(np.float32)
        contrast = rng.uniform(1.0 - color.contrast, 1.0 + color.contrast)
        out = (out - 127.5) * contrast + 127.5
        out *= rng.uniform(1.0 - color.brightness, 1.0 + color.brightness)
        out = np.clip(out, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(1.0 - color.saturation, 1.0 + color.saturation)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-color.hue, color.hue) * 180.0) % 180.0
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        img = decode(self.paths[i])
        (h, w) = img.shape[:2]
        log_area = np.log(h * w * rng.uniform(self.cfg.crop_floor, 1.0, WINDOW_TRIES))
        log_aspect = rng.uniform(np.log(ASPECT_SPAN[0]), np.log(ASPECT_SPAN[1]), WINDOW_TRIES)
        rows = np.round(np.exp((log_area - log_aspect) / 2)).astype(np.int64)
        cols = np.round(np.exp((log_area + log_aspect) / 2)).astype(np.int64)
        inside = np.flatnonzero((rows > 0) & (rows <= h) & (cols > 0) & (cols <= w))
        if len(inside):
            (rows, cols) = (int(rows[inside[0]]), int(cols[inside[0]]))
            (top, left) = (int(rng.integers(0, h - rows + 1)), int(rng.integers(0, w - cols + 1)))
        else:
            rows = cols = min(h, w)
            (top, left) = ((h - rows) // 2, (w - cols) // 2)
        piece = img[top:top + rows, left:left + cols]
        if rng.random() < self.cfg.detail_probability:
            k = rng.uniform(*self.cfg.detail_span) / max(h, w)
            if k < 1.0:
                piece = fit(piece, max(1, int(round(rows * k))), max(1, int(round(cols * k))))
        piece = fit(piece, self.cfg.side, self.cfg.side)
        piece = self._light_color(piece, rng)
        if rng.random() < 0.5:
            piece = piece[:, ::-1]
        random.seed(int(rng.integers(1 << 62)))
        out = np.array(self._policy()(Image.fromarray(np.ascontiguousarray(piece))), copy=True)
        return (torch.from_numpy(out), int(self.labels[i]), int(i))

def quiet_worker(_: int) -> None:
    cv2.setNumThreads(1)

class ClipPixelNormalizer:

    def __init__(self, device: str) -> None:
        self.shift = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255.0
        self.scale = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255.0

    def __call__(self, block: torch.Tensor) -> torch.Tensor:
        x = block.to(self.shift.device, non_blocking=True).permute(0, 3, 1, 2).float()
        return (x - self.shift) / self.scale
