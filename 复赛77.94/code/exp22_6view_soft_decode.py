#!/usr/bin/env python3
"""Decode Exp22 six-view logits with a soft uniform class-prior correction."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import torch


def fit_uniform_bias(logits: torch.Tensor, iters: int = 200, step: float = 1.0) -> torch.Tensor:
    """Fit one bias per class so expected soft counts approach uniform."""
    n, classes = logits.shape
    target = float(n) / float(classes)
    bias = torch.zeros(classes, dtype=torch.float32)
    for _ in range(iters):
        expected = torch.softmax(logits + bias, dim=1).sum(dim=0)
        bias -= step * torch.log((expected / target).clamp_min(1e-8))
        bias -= bias.mean()
    return bias


def fit_bounded_bias(logits: np.ndarray, low: int, high: int, steps: int = 1600):
    """Fit one global class bias while requiring argmax counts in [low, high]."""
    classes = logits.shape[1]
    target = logits.shape[0] / classes
    bias = np.zeros(classes, dtype=np.float32)
    best = None
    best_violation = float("inf")
    for step_index in range(steps):
        labels = np.argmax(logits + bias[None, :], axis=1)
        counts = np.bincount(labels, minlength=classes)
        violation = np.maximum(low - counts, 0) + np.maximum(counts - high, 0)
        total_violation = float(violation.sum())
        if total_violation < best_violation:
            best_violation = total_violation
            best = (bias.copy(), labels.copy(), counts.copy(), step_index)
        if total_violation == 0:
            return bias, labels, counts, step_index
        error = np.clip((target - counts) / max(target, 1.0), -1.0, 1.0)
        rate = 0.10 if step_index < 300 else 0.04
        bias += rate * error.astype(np.float32)
        bias -= bias.mean()
    return best


def save_submission(out_dir: Path, tag: str, names: list[str], labels: np.ndarray, classes: list[str], report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"pred_results_{tag}.csv"
    zip_path = out_dir / f"submission_{tag}.zip"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for name, label in zip(names, labels.tolist()):
            writer.writerow((name, classes[int(label)]))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="pred_results.csv")
    report = dict(report)
    report.update({"csv": str(csv_path), "zip": str(zip_path)})
    (out_dir / f"prediction_summary_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    logits = torch.from_numpy(np.load(args.logits).astype(np.float32))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    names = [
        p.name
        for p in sorted(args.test_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    classes = list(config["submission_ids"])
    if logits.shape != (len(names), len(classes)):
        raise ValueError(f"logits shape {tuple(logits.shape)} != {(len(names), len(classes))}")

    raw = logits.argmax(dim=1).numpy().astype(np.int32)
    bias = fit_uniform_bias(logits)
    soft = (logits + bias[None, :]).argmax(dim=1).numpy().astype(np.int32)
    raw_counts = np.bincount(raw, minlength=len(classes))
    soft_counts = np.bincount(soft, minlength=len(classes))
    common = {
        "experiment": "Exp22-750-b512-50ep",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "weights_kind": "EMA",
        "image_size": int(config["image_size"]),
        "tta": "multiscale6",
        "views": "512/576/640 x original/horizontal-flip",
        "images": len(names),
        "classes": len(classes),
        "raw_min": int(raw_counts.min()),
        "raw_max": int(raw_counts.max()),
    }
    save_submission(
        args.output_dir,
        "raw",
        names,
        raw,
        classes,
        {**common, "method": "six-view argmax", "final_min": int(raw_counts.min()), "final_max": int(raw_counts.max()), "changed_from_raw": 0},
    )
    save_submission(
        args.output_dir,
        "soft40_60",
        names,
        soft,
        classes,
        {
            **common,
            "method": "soft uniform logit-bias decoding",
            "requested_range": [40, 60],
            "final_min": int(soft_counts.min()),
            "final_max": int(soft_counts.max()),
            "final_mean": float(soft_counts.mean()),
            "final_std": float(soft_counts.std()),
            "within_40_60": bool(soft_counts.min() >= 40 and soft_counts.max() <= 60),
            "changed_from_raw": int(np.count_nonzero(soft != raw)),
        },
    )
    bounded_bias, bounded, bounded_counts, iterations = fit_bounded_bias(
        logits.numpy(), low=40, high=60
    )
    save_submission(
        args.output_dir,
        "soft40_60_bounded",
        names,
        bounded,
        classes,
        {
            **common,
            "method": "soft global logit-bias bounded decoding",
            "requested_range": [40, 60],
            "final_min": int(bounded_counts.min()),
            "final_max": int(bounded_counts.max()),
            "final_mean": float(bounded_counts.mean()),
            "final_std": float(bounded_counts.std()),
            "within_40_60": bool(bounded_counts.min() >= 40 and bounded_counts.max() <= 60),
            "changed_from_raw": int(np.count_nonzero(bounded != raw)),
            "iterations": int(iterations),
        },
    )


if __name__ == "__main__":
    main()
