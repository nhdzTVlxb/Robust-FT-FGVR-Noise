"""Exp22 EMA inference with score-level class-capacity constraints.

The normal argmax prediction is kept as a baseline.  Optional balanced
variants change the assignment while the logits are still available:

* ``soft_prior``: a conservative log-prior correction;
* ``range30_70``: score-preserving assignment with 30--70 samples/class;
* ``exact_uniform``: score-preserving assignment with 49--50 samples/class
  for the 24,967-image / 500-class preliminary test set.

The CSV is never edited after the fact.  Each label is selected from the
model's per-image score vector under the requested capacity constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from config import Config
from model_utils import (
    CLIP_MEAN,
    CLIP_STD,
    build_clip_model,
    clip_forward,
    inspect_model,
)


class CropView:
    def __init__(self, size: int, top: int, left: int, flip: bool = False):
        self.size = size
        self.top = top
        self.left = left
        self.flip = flip

    def __call__(self, image):
        if self.flip:
            image = TF.hflip(image)
        return TF.crop(image, self.top, self.left, self.size, self.size)


def build_views(size: int, mode: str = "legacy14"):
    larger = int(size * 1.14)
    delta = larger - size
    normalize = [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if mode == "center2":
        return [
            T.Compose([T.Resize(larger), T.CenterCrop(size)] + normalize),
            T.Compose(
                [
                    T.Resize(larger),
                    T.CenterCrop(size),
                    T.RandomHorizontalFlip(p=1.0),
                ]
                + normalize
            ),
        ]
    positions = [
        (delta // 2, delta // 2),
        (0, 0),
        (0, delta),
        (delta, 0),
        (delta, delta),
        (0, delta // 2),
        (delta // 2, 0),
    ]
    return [
        T.Compose(
            [T.Resize(larger), CropView(size, top, left, flip), T.Resize(size)]
            + normalize
        )
        for top, left in positions
        for flip in (False, True)
    ]


class TTADataset(Dataset):
    def __init__(self, files, views):
        self.files = list(files)
        self.views = list(views)

    def __len__(self):
        return len(self.files) * len(self.views)

    def __getitem__(self, index):
        source_index, view_index = divmod(index, len(self.views))
        path = self.files[source_index]
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot decode test image: {path}") from exc
        return self.views[view_index](image), source_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--output", type=str, default="pred_v22_balanced")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tta", choices=["legacy14", "center2"], default="legacy14")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["raw", "soft_prior", "range30_70", "exact_uniform"],
        default=["raw", "soft_prior", "range30_70", "exact_uniform"],
    )
    parser.add_argument(
        "--soft-strength",
        type=float,
        default=0.35,
        help="log-prior correction strength for soft_prior",
    )
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--with-header", action="store_true")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, clip_model: Path, device: str):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    vision, head, actual_size = build_clip_model(
        str(clip_model),
        config["num_classes"],
        config["image_size"],
        config["lora_rank"],
        config["lora_alpha"],
        config["lora_dropout"],
        config["lora_targets"],
    )
    if state.get("weights_kind") == "lora_ema_only":
        # Compact handoff checkpoint: rebuild the frozen CLIP base and load
        # only the EMA LoRA/classifier tensors.
        combined = torch.nn.Sequential(vision, head)
        missing, unexpected = combined.load_state_dict(
            state["ema_state_dict"], strict=False
        )
        trainable_names = {
            name for name, parameter in combined.named_parameters() if parameter.requires_grad
        }
        if unexpected or any(key not in trainable_names for key in missing):
            raise RuntimeError(
                f"compact EMA checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
    elif state.get("weights_kind") == "raw_resume":
        combined = torch.nn.Sequential(vision, head)
        vision.load_state_dict(state["vision_state_dict"], strict=True)
        head.load_state_dict(state["head_state_dict"], strict=True)
        combined.load_state_dict(state["ema_state_dict"], strict=False)
    else:
        vision.load_state_dict(state["vision_state_dict"], strict=True)
        head.load_state_dict(state["head_state_dict"], strict=True)
    return (
        state,
        config,
        vision.to(device).eval(),
        head.to(device).eval(),
        actual_size,
    )


def _counts(labels: np.ndarray, classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=classes).astype(np.int64)


def _repair_capacity(
    scores: np.ndarray,
    lower: int,
    upper: int,
    topk: int = 64,
) -> tuple[np.ndarray, dict]:
    """Greedily repair argmax labels while minimizing score loss.

    First fill classes below ``lower`` using the best available reassignment,
    then move only the weakest-margin samples out of classes above ``upper``.
    With lower <= N/C <= upper, this always has a feasible assignment.
    """
    n, classes = scores.shape
    if lower < 0 or upper < lower or upper * classes < n or lower * classes > n:
        raise ValueError(
            f"infeasible capacities lower={lower}, upper={upper}, n={n}, c={classes}"
        )

    labels = scores.argmax(axis=1).astype(np.int32)
    rows = np.arange(n)
    chosen = scores[rows, labels].copy()
    counts = _counts(labels, classes)
    moves = 0
    score_loss = 0.0

    # Fill lower bounds.  Donors are never taken below their own lower bound.
    while True:
        under = np.flatnonzero(counts < lower)
        if under.size == 0:
            break
        best_gain = -np.inf
        best_i = -1
        best_target = -1
        donor_ok = counts[labels] > lower
        for target in under.tolist():
            mask = donor_ok & (labels != target)
            if not np.any(mask):
                continue
            gains = scores[:, target] - chosen
            gains = np.where(mask, gains, -np.inf)
            i = int(np.argmax(gains))
            if gains[i] > best_gain:
                best_gain = float(gains[i])
                best_i = i
                best_target = target
        if best_i < 0:
            raise RuntimeError("Could not satisfy class lower capacities")
        source = int(labels[best_i])
        labels[best_i] = best_target
        counts[source] -= 1
        counts[best_target] += 1
        score_loss -= best_gain
        chosen[best_i] = scores[best_i, best_target]
        moves += 1

    # A top-k table makes the over-capacity repair fast.  There are many
    # available destination classes in these balanced regimes; full-score
    # fallback handles the rare case where all top-k classes are full.
    k = min(max(2, int(topk)), classes)
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)

    while True:
        over = np.flatnonzero(counts > upper)
        if over.size == 0:
            break
        best_gain = -np.inf
        best_i = -1
        best_target = -1
        for source in over.tolist():
            idx = np.flatnonzero(labels == source)
            if idx.size == 0:
                continue
            cand = top_idx[idx]
            cand_scores = top_scores[idx]
            available = counts[cand] < upper
            gains = cand_scores - chosen[idx, None]
            gains = np.where(available, gains, -np.inf)
            flat = int(np.argmax(gains))
            r, col = np.unravel_index(flat, gains.shape)
            gain = float(gains[r, col])
            if not np.isfinite(gain):
                # This should be uncommon; use the complete score row.
                for local, i in enumerate(idx.tolist()):
                    mask = counts < upper
                    mask[source] = False
                    if not np.any(mask):
                        continue
                    row = np.where(mask, scores[i], -np.inf)
                    target = int(np.argmax(row))
                    gain_i = float(row[target] - chosen[i])
                    if gain_i > best_gain:
                        best_gain, best_i, best_target = gain_i, i, target
            elif gain > best_gain:
                best_gain = gain
                best_i = int(idx[r])
                best_target = int(cand[r, col])
        if best_i < 0 or best_target < 0:
            raise RuntimeError("Could not satisfy class upper capacities")
        source = int(labels[best_i])
        labels[best_i] = best_target
        counts[source] -= 1
        counts[best_target] += 1
        score_loss -= best_gain
        chosen[best_i] = scores[best_i, best_target]
        moves += 1

    if np.any(counts < lower) or np.any(counts > upper):
        raise AssertionError("capacity repair produced invalid class counts")
    info = {
        "lower": int(lower),
        "upper": int(upper),
        "moves": int(moves),
        "score_loss_sum": float(score_loss),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "count_mean": float(counts.mean()),
    }
    return labels, info


def soft_prior_labels(
    scores: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, dict]:
    raw = scores.argmax(axis=1)
    counts = _counts(raw, scores.shape[1]).astype(np.float64)
    target = float(len(raw)) / float(scores.shape[1])
    # Add a conservative correction toward a uniform test prior.  The
    # epsilon prevents a missing raw class from causing an infinite bias.
    bias = float(strength) * np.log(target / np.maximum(counts, 1.0))
    adjusted = scores + bias[None, :]
    labels = adjusted.argmax(axis=1).astype(np.int32)
    new_counts = _counts(labels, scores.shape[1])
    return labels, {
        "strength": float(strength),
        "count_min": int(new_counts.min()),
        "count_max": int(new_counts.max()),
        "count_mean": float(new_counts.mean()),
        "raw_count_min": int(counts.min()),
        "raw_count_max": int(counts.max()),
    }


def write_variant(
    output_root: Path,
    name: str,
    files,
    labels: np.ndarray,
    submission_ids,
    args,
    state,
    actual_size: int,
    info: dict,
):
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pred_results.csv"
    zip_path = out_dir / "submission.zip"
    with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
        if args.with_header:
            handle.write("filename,label\n")
        for path, label in zip(files, labels.tolist()):
            handle.write(f"{path.name},{submission_ids[int(label)]}\n")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, "pred_results.csv")
    counts = _counts(labels, len(submission_ids))
    report = {
        "variant": name,
        "images": len(files),
        "classes": len(submission_ids),
        "tta": args.tta,
        "views": int(info.get("views", 0)),
        "checkpoint_epoch": int(state["epoch"]),
        "weights_kind": "EMA",
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "count_mean": float(counts.mean()),
        "count_std": float(counts.std()),
        "counts": counts.tolist(),
        "algorithm": info,
        "csv": str(csv_path),
        "zip": str(zip_path),
    }
    (out_dir / "prediction_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "variant",
                    "images",
                    "count_min",
                    "count_max",
                    "count_mean",
                    "count_std",
                    "algorithm",
                    "csv",
                    "zip",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


@torch.inference_mode()
def predict(args):
    cfg = Config()
    test_dir = Path(args.test_dir) if args.test_dir else cfg.test_dir
    clip_model = Path(args.clip_model) if args.clip_model else cfg.clip_model
    checkpoint = Path(args.checkpoint).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inspect_model(str(clip_model))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    state, config, vision, head, actual_size = load_checkpoint(
        checkpoint, clip_model, str(device)
    )
    files = sorted(
        p
        for p in test_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not files:
        raise RuntimeError(f"No test images found in {test_dir}")
    if any("," in p.name or "\n" in p.name for p in files):
        raise ValueError("Image filenames cannot contain commas or newlines")

    views = build_views(actual_size, args.tta)
    loader = DataLoader(
        TTADataset(files, views),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    scores_total = torch.zeros(
        len(files), config["num_classes"], dtype=torch.float32
    )
    counts = torch.zeros(len(files), dtype=torch.long)
    amp_on = device.type == "cuda" and config.get("amp", "float16") != "float32"
    amp_dtype = getattr(torch, config.get("amp", "float16"))
    for step, (images, indices) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp_on, dtype=amp_dtype):
            scores = clip_forward(vision, head, images)
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite prediction scores")
        scores_total.index_add_(0, indices, scores.float().cpu())
        counts.index_add_(0, indices, torch.ones_like(indices))
        if step % 100 == 0:
            print(f"[Inference] batch {step + 1}/{len(loader)}", flush=True)
    if not torch.all(counts == len(views)):
        raise RuntimeError("Missing or duplicated TTA views")

    score_matrix = (scores_total / len(views)).numpy()
    if args.save_scores:
        np.save(output_root / "exp22_ema_tta_logits.npy", score_matrix)
    submission_ids = config["submission_ids"]
    common = {"views": len(views)}
    for variant in args.variants:
        if variant == "raw":
            labels = score_matrix.argmax(axis=1).astype(np.int32)
            info = {**common, "method": "plain argmax"}
        elif variant == "soft_prior":
            labels, info = soft_prior_labels(score_matrix, args.soft_strength)
            info = {**common, "method": "uniform log-prior correction", **info}
        elif variant == "range30_70":
            labels, info = _repair_capacity(
                score_matrix, lower=30, upper=70, topk=args.topk
            )
            info = {**common, "method": "capacity-constrained reassignment", **info}
        elif variant == "exact_uniform":
            # N=24967 and C=500 gives feasible capacities 49--50.
            lower = len(files) // len(submission_ids)
            upper = lower + (1 if len(files) % len(submission_ids) else 0)
            labels, info = _repair_capacity(
                score_matrix, lower=lower, upper=upper, topk=args.topk
            )
            info = {**common, "method": "exact uniform capacity reassignment", **info}
        else:
            raise AssertionError(variant)
        write_variant(
            output_root,
            variant,
            files,
            labels,
            submission_ids,
            args,
            state,
            actual_size,
            info,
        )

    raw = score_matrix.argmax(axis=1)
    print(
        json.dumps(
            {
                "images": len(files),
                "classes": len(submission_ids),
                "tta": args.tta,
                "views": len(views),
                "checkpoint_epoch": state["epoch"],
                "raw_count_min": int(_counts(raw, len(submission_ids)).min()),
                "raw_count_max": int(_counts(raw, len(submission_ids)).max()),
                "output": str(output_root),
            },
            ensure_ascii=False,
        )
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    predict(args)


if __name__ == "__main__":
    main()
