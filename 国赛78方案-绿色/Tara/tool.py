import csv
import io
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIEW_COUNT = 7
LONG_EDGE_CAP = 500
SCALES = (448, 480, 512, 544, 576)
BALANCE_STRENGTH = 0.28
VIEW_BATCH_SIZE = 704
LOADER_WORKERS = 8
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
PATCH = 32
TIMM_ID = 'vit_base_patch32_clip_quickgelu_224.openai'


def decode(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def fit(image: np.ndarray, rows: int, columns: int) -> np.ndarray:
    factor = min(image.shape[0] // max(rows, 1), image.shape[1] // max(columns, 1))
    if factor >= 2:
        image = cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_AREA)
    mode = cv2.INTER_LINEAR if image.shape[0] * image.shape[1] > rows * columns else cv2.INTER_CUBIC
    return cv2.resize(image, (columns, rows), interpolation=mode)


def square_view(image: np.ndarray, side: int, long_cap: int) -> np.ndarray:
    height, width = image.shape[:2]
    edge = min(height, width)
    square = image[(height-edge)//2:(height-edge)//2+edge, (width-edge)//2:(width-edge)//2+edge]
    if long_cap and max(height, width) > long_cap:
        scaled = max(1, int(round(edge * long_cap / max(height, width))))
        square = fit(square, scaled, scaled)
    return fit(square, side, side)


class ClipNormalizer:
    def __init__(self, device: str):
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255.0

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        tensor = pixels.to(self.mean.device, non_blocking=True).permute(0, 3, 1, 2).float()
        return (tensor - self.mean) / self.std


class TaraClassifier(nn.Module):
    def __init__(self, classes: int, side: int):
        super().__init__()
        import timm
        self.tower = timm.create_model(TIMM_ID, pretrained=False, img_size=side, weight_init='skip')
        self.tower.patch_embed.strict_img_size = False
        self.readout = nn.Linear(self.tower.num_classes, classes)
        self.register_buffer('class_codes', torch.arange(classes, dtype=torch.int64))
        self.grid = side // PATCH

    def position_table(self, grid: int) -> torch.Tensor:
        table = self.tower.pos_embed
        if grid == self.grid:
            return table
        body = F.interpolate(
            table[:, 1:].unflatten(1, (self.grid, self.grid)).movedim(-1, 1).float(),
            size=(grid, grid), mode='bicubic', align_corners=False, antialias=grid < self.grid,
        )
        body = body.to(table.dtype).movedim(1, -1).flatten(1, 2)
        return torch.cat((table[:, :1], body), dim=1)

    def forward(self, pixels: torch.Tensor):
        tower = self.tower
        tokens = tower.patch_embed(pixels)
        tokens = torch.cat((tower.cls_token.expand(len(tokens), -1, -1), tokens), dim=1)
        tokens = tokens + self.position_table(pixels.shape[-1] // PATCH)
        tokens = tower.norm_pre(tokens)
        for block in tower.blocks:
            tokens = block(tokens)
        embeddings = tower.head(tower.norm(tokens)[:, 0])
        return self.readout(embeddings), embeddings


def restore(weights_file: Path) -> TaraClassifier:
    state = torch.load(weights_file, map_location='cpu', weights_only=True)
    classes = len(state['class_codes'])
    grid = round((state['tower.pos_embed'].shape[1] - 1) ** 0.5)
    model = TaraClassifier(classes, grid * PATCH)
    state.pop('init_temperature', None)
    model.load_state_dict(state, strict=True)
    return model


def quiet_worker(_: int) -> None:
    cv2.setNumThreads(1)


def image_files(folder: Path):
    return sorted(path for path in Path(folder).iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


class LegacyCrops(Dataset):
    def __init__(self, paths: Sequence[Path], side: int, long_cap: int = LONG_EDGE_CAP):
        self.paths = list(paths)
        self.side = side
        self.long_cap = long_cap
        self.canvas_side = int(side * 1.14)
        delta = self.canvas_side - side
        self.positions = [
            (delta // 2, delta // 2),
            (0, 0),
            (0, delta),
            (delta, 0),
            (delta, delta),
            (0, delta // 2),
            (delta // 2, 0),
        ]
        self.cached_index = -1
        self.cached_canvas = None

    def __len__(self):
        return len(self.paths) * len(self.positions)

    def __getitem__(self, index: int):
        image_index, view_index = divmod(index, len(self.positions))
        if image_index != self.cached_index:
            self.cached_index = image_index
            source = decode(self.paths[image_index])
            self.cached_canvas = square_view(source, self.canvas_side, self.long_cap)
        top, left = self.positions[view_index]
        crop = self.cached_canvas[top:top + self.side, left:left + self.side]
        if crop.shape != (self.side, self.side, 3):
            raise RuntimeError('invalid crop geometry for %s' % self.paths[image_index])
        return torch.from_numpy(np.ascontiguousarray(crop)), image_index


@torch.inference_mode()
def collect_side(model, paths: Sequence[Path], side: int, device: torch.device,
                 forward_model=None, batch_size: int = VIEW_BATCH_SIZE,
                 workers: int = LOADER_WORKERS):
    dataset = LegacyCrops(paths, side)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=quiet_worker,
        prefetch_factor=2 if workers else None,
    )
    normalizer = ClipNormalizer(str(device))
    class_probs = np.zeros((len(paths), model.readout.out_features), dtype=np.float32)
    embeddings = np.zeros((len(paths), model.readout.in_features), dtype=np.float32)
    counts = np.zeros(len(paths), dtype=np.int16)
    model.eval()
    runner = forward_model if forward_model is not None else model
    for pixels, image_ids in loader:
        x = normalizer(pixels)
        mirrored = x.flip(-1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits_a, embedding_a = runner(x)
            logits_b, embedding_b = runner(mirrored)
        probabilities = (logits_a.float().softmax(1) + logits_b.float().softmax(1)) * 0.5
        features = F.normalize(embedding_a.float(), dim=1) + F.normalize(embedding_b.float(), dim=1)
        probabilities = probabilities.cpu().numpy()
        features = features.cpu().numpy()
        ids = image_ids.numpy()
        np.add.at(class_probs, ids, probabilities)
        np.add.at(embeddings, ids, features)
        np.add.at(counts, ids, 1)
    if len(counts) and (counts.min() != VIEW_COUNT or counts.max() != VIEW_COUNT):
        raise RuntimeError('view count mismatch: %d..%d' % (counts.min(), counts.max()))
    class_probs /= VIEW_COUNT
    embeddings /= VIEW_COUNT
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return class_probs.astype(np.float16), embeddings.astype(np.float16)


def balanced_rows(log_kernel: torch.Tensor, sweeps: int = 500, tolerance: float = 1e-5) -> torch.Tensor:
    rows, columns = log_kernel.shape
    column_total = log_kernel.new_tensor(rows / columns).log()
    right = log_kernel.new_zeros(columns)
    for sweep in range(1, sweeps + 1):
        left = -torch.logsumexp(log_kernel + right, dim=1)
        right = column_total - torch.logsumexp(log_kernel + left[:, None], dim=0)
        if sweep % 10 == 0:
            row_error = (torch.logsumexp(log_kernel + left[:, None] + right, dim=1).exp() - 1).abs().max()
            if row_error < tolerance:
                break
    return torch.exp(log_kernel + left[:, None] + right)


def even_scaling(matrix: torch.Tensor, sweeps: int = 200, tolerance: float = 1e-5) -> torch.Tensor:
    rows, columns = matrix.shape
    matrix = torch.maximum(matrix, matrix.new_tensor(1e-30))
    for sweep in range(sweeps):
        matrix.div_(matrix.sum(1, keepdim=True)).mul_((rows / columns) / matrix.sum(0, keepdim=True))
        if sweep % 10 == 9 and matrix.sum(1).sub(1).abs().amax() < tolerance:
            break
    return matrix


def soft_rows(log_kernel: torch.Tensor) -> torch.Tensor:
    plain = log_kernel.softmax(1)
    balanced = balanced_rows(log_kernel)
    return (1.0 - BALANCE_STRENGTH) * plain + BALANCE_STRENGTH * balanced


def soft_even(matrix: torch.Tensor) -> torch.Tensor:
    plain = torch.maximum(matrix, matrix.new_tensor(1e-30))
    plain = plain / plain.sum(1, keepdim=True)
    balanced = even_scaling(matrix.clone())
    return (1.0 - BALANCE_STRENGTH) * plain + BALANCE_STRENGTH * balanced


class PrototypePropagation:
    def __init__(self, features: torch.Tensor, probabilities: torch.Tensor,
                 anchors: torch.Tensor, finish: dict):
        self.anchors = anchors
        self.finish = finish
        x = features.double()
        q = anchors.double()
        mass = q.sum(0).clamp(min=1e-9)
        centres = (q.mT @ x) / mass.unsqueeze(1)
        covariance = (x.mT @ x - (centres * mass.unsqueeze(1)).mT @ centres) / len(x)
        ridge = finish['shrink'] * torch.trace(covariance) / x.shape[1]
        covariance *= 1.0 - finish['shrink']
        covariance.diagonal().add_(ridge)
        values, vectors = torch.linalg.eigh(covariance.cpu())
        root = ((vectors * values.clamp(min=1e-12).rsqrt()) @ vectors.mT).to(x.device)
        self.features = F.normalize(((x - x.mean(0)) @ root).float(), dim=1)
        self.log_evidence = probabilities.clamp(min=1e-12).log()
        self.identity = torch.eye(probabilities.shape[1], device=probabilities.device)
        self.prototypes = F.normalize(anchors.mT @ self.features / anchors.sum(0).clamp(min=1e-9).unsqueeze(1), dim=1)

    def membership(self) -> torch.Tensor:
        gap = (2.0 - 2.0 * self.features @ self.prototypes.mT).clamp(min=0)
        return soft_rows(self.log_evidence - self.finish['kernel_sharpness'] * gap)

    def smoothed(self, assignments: torch.Tensor) -> torch.Tensor:
        gram = assignments.mT @ assignments
        laplacian = self.identity - assignments.mT @ (assignments / assignments.sum(0).clamp(min=1e-9))
        fitted = torch.linalg.solve(gram + self.finish['ridge'] * laplacian,
                                    assignments.mT @ self.anchors)
        return soft_even((assignments @ fitted).clamp(min=0))

    def blend(self, assignments: torch.Tensor) -> torch.Tensor:
        amount = self.finish['graph_weight']
        return (amount * self.smoothed(assignments) + (1.0 - amount) * assignments).clamp(0, 1)

    def run(self) -> torch.Tensor:
        for _ in range(self.finish['sweeps']):
            weights = self.blend(self.membership())
            target = weights.mT @ self.features / weights.sum(0).clamp(min=1e-9).unsqueeze(1)
            self.prototypes = F.normalize(
                self.prototypes + self.finish['drift'] * (target - self.prototypes), dim=1
            )
        return soft_even(self.blend(self.membership()).clamp(min=1e-12))


@torch.inference_mode()
def infer_labels(model, paths: Sequence[Path], device: torch.device,
                 batch_size: int = VIEW_BATCH_SIZE,
                 workers: int = LOADER_WORKERS) -> np.ndarray:
    probabilities_by_scale = []
    features_by_scale = []
    forward_model = torch.compile(model) if hasattr(torch, 'compile') else model
    for side in SCALES:
        probabilities, features = collect_side(
            model,
            paths,
            side,
            device,
            forward_model=forward_model,
            batch_size=batch_size,
            workers=workers,
        )
        probabilities_by_scale.append(probabilities)
        features_by_scale.append(features)
        print('inference | side=%d | views=%d | images=%d' % (side, VIEW_COUNT * 2, len(paths)), flush=True)
    probabilities = torch.stack([torch.as_tensor(value).float() for value in probabilities_by_scale]).mean(0).to(device)
    features = torch.stack([torch.as_tensor(value).float() for value in features_by_scale]).mean(0).to(device)
    features = F.normalize(features, dim=1)
    anchors = soft_rows(probabilities.clamp(min=1e-12).log())
    finish = {
        'shrink': 0.3,
        'sweeps': 20,
        'kernel_sharpness': 10.0,
        'drift': 0.2,
        'graph_weight': 0.6,
        'ridge': 1.0,
    }
    assignments = PrototypePropagation(features, probabilities, anchors, finish).run()
    return assignments.argmax(1).cpu().numpy()


def submission_zip(destination: Path, paths: Sequence[Path], labels: np.ndarray,
                   class_names: Sequence[str]) -> None:
    if len(paths) != len(labels):
        raise ValueError('prediction count does not match input image count')
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows((path.name, class_names[int(label)]) for path, label in zip(paths, labels))
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('pred_results.csv', output.getvalue().encode('utf-8'))
    with zipfile.ZipFile(destination) as archive:
        if archive.namelist() != ['pred_results.csv'] or archive.testzip() is not None:
            raise RuntimeError('submission zip validation failed')
        rows = list(csv.reader(io.StringIO(archive.read('pred_results.csv').decode('utf-8'))))
    if len(rows) != len(paths) or any(len(row) != 2 for row in rows):
        raise RuntimeError('submission CSV validation failed')
    if [row[0] for row in rows] != [path.name for path in paths]:
        raise RuntimeError('submission image ordering changed')
