"""Feature extraction + legacy confidence-selection baseline (thin entry).

Primary modern use: building the frozen-feature cache (outputs/cache/*.pt)
that every other script consumes. The 2-stage confidence baseline kept here is
superseded by exp_head.py --final and finetune_lora.py but remains runnable.

Run: python main.py
Debug: python main.py --max-train-samples 2000 --max-test-samples 2000 --work-dir outputs_tmp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import config
from robustft.data import (
    LabeledImageDataset,
    UnlabeledImageDataset,
    build_extract_transform,
    load_test_paths,
    load_train_samples,
)
from robustft.denoise import select_clean_indices
from robustft.engine import build_device, seed_everything, train_head
from robustft.features import cache_matches, extract_features, load_tensor_cache, save_tensor_cache
from robustft.models import MODEL_NAME, build_frozen_backbone
from robustft.submission import save_predictions, zip_submission


def run_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = build_device(args.device)
    print(f"device: {device}")
    print(f"model: {MODEL_NAME}")

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train_features.pt"
    test_cache = cache_dir / "test_features.pt"

    backbone = build_frozen_backbone(device)
    transform = build_extract_transform(backbone)

    train_samples, class_names = load_train_samples(train_dir, args.max_train_samples, args.seed)
    test_paths = load_test_paths(test_dir, args.max_test_samples, args.seed)
    train_names = [Path(p).name for p, _ in train_samples]
    test_expected_names = [Path(p).name for p in test_paths]
    cache_meta = {"model_name": MODEL_NAME, "tta_flip": not args.no_tta_flip}
    print(f"train samples: {len(train_samples)}")
    print(f"test samples: {len(test_paths)}")
    print(f"classes: {len(class_names)}")

    train_cache_ok = False
    if not args.rebuild_cache and train_cache.exists():
        cached = load_tensor_cache(train_cache)
        train_cache_ok = cache_matches(cached, train_names, "image_names", cache_meta) and cached.get("class_names") == class_names
        if train_cache_ok:
            train_features = cached["features"]
            train_labels = cached["labels"]
            class_names = cached.get("class_names", class_names)
        else:
            print("train feature cache does not match current data; rebuilding")

    if not train_cache_ok:
        train_dataset = LabeledImageDataset(train_samples, transform)
        train_features, train_labels = extract_features(
            backbone,
            train_dataset,
            device,
            batch_size=args.extract_batch_size,
            num_workers=args.num_workers,
            tta_flip=not args.no_tta_flip,
            desc="train features",
        )
        save_tensor_cache(
            train_cache,
            {
                "features": train_features,
                "labels": train_labels,
                "class_names": class_names,
                "image_names": train_names,
                "model_name": MODEL_NAME,
                "tta_flip": not args.no_tta_flip,
            },
        )

    test_cache_ok = False
    if not args.rebuild_cache and test_cache.exists():
        cached = load_tensor_cache(test_cache)
        test_cache_ok = cache_matches(cached, test_expected_names, "names", cache_meta)
        if test_cache_ok:
            test_features = cached["features"]
            test_names = cached["names"]
        else:
            print("test feature cache does not match current data; rebuilding")

    if not test_cache_ok:
        test_dataset = UnlabeledImageDataset(test_paths, transform)
        test_features, test_names = extract_features(
            backbone,
            test_dataset,
            device,
            batch_size=args.extract_batch_size,
            num_workers=args.num_workers,
            tta_flip=not args.no_tta_flip,
            desc="test features",
        )
        save_tensor_cache(
            test_cache,
            {
                "features": test_features,
                "names": test_names,
                "model_name": MODEL_NAME,
                "tta_flip": not args.no_tta_flip,
            },
        )

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features = train_features.contiguous().to(device=device, dtype=torch.float32)
    test_features = test_features.contiguous().to(device=device, dtype=torch.float32)
    train_labels = train_labels.contiguous().to(device=device, dtype=torch.long)
    train_idx = torch.arange(train_labels.size(0), device=device)

    print("stage 1: noisy-label warmup")
    stage1 = train_head(
        train_features,
        train_labels,
        len(class_names),
        train_idx,
        epochs=args.stage1_epochs,
        batch_size=args.head_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoothing=args.stage1_label_smoothing,
        dropout=args.dropout,
        device=device,
        verbose=True,
    )

    print("selecting clean samples")
    keep_idx, conf = select_clean_indices(stage1, train_features, train_labels, args.keep_ratio)
    keep_count = int(keep_idx.numel())
    print(f"kept {keep_count}/{train_labels.numel()} samples ({keep_count / max(1, train_labels.numel()):.2%})")

    print("stage 2: clean subset refinement")
    stage2 = train_head(
        train_features,
        train_labels,
        len(class_names),
        keep_idx,
        epochs=args.stage2_epochs,
        batch_size=args.head_batch_size,
        lr=args.lr * args.stage2_lr_mult,
        weight_decay=args.weight_decay,
        smoothing=args.stage2_label_smoothing,
        dropout=args.dropout,
        device=device,
        init_state=stage1.state_dict(),
        verbose=True,
    )

    artifact_dir = work_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for stage_num, model in ((1, stage1), (2, stage2)):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_features": int(train_features.size(1)),
                "num_classes": len(class_names),
                "class_names": class_names,
                "stage": stage_num,
                "keep_ratio": args.keep_ratio if stage_num == 2 else None,
            },
            artifact_dir / f"stage{stage_num}_head.pt",
        )

    print("predicting test set")
    stage2.eval()
    pred_indices: list[int] = []
    with torch.inference_mode():
        for start in tqdm(range(0, test_features.size(0), args.head_batch_size), desc="predict"):
            xb = test_features[start : start + args.head_batch_size]
            logits = stage2(xb)
            pred_indices.extend(logits.argmax(dim=1).tolist())

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = Path.cwd() / output_csv
    save_predictions(output_csv, test_names, pred_indices, class_names)
    zip_path = zip_submission(output_csv)

    meta = {
        "model": MODEL_NAME,
        "device": str(device),
        "train_samples": int(train_labels.numel()),
        "test_samples": int(test_features.size(0)),
        "classes": len(class_names),
        "keep_ratio": args.keep_ratio,
        "output_csv": str(output_csv),
        "output_zip": str(zip_path),
    }
    with (artifact_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_csv}")
    print(f"saved: {zip_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP ViT-B/32 feature extraction + legacy baseline")
    parser.add_argument("--train-dir", default=str(config.TRAIN_DIR))
    parser.add_argument("--test-dir", default=str(config.TEST_DIR))
    parser.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR))
    parser.add_argument("--output-csv", default=str(config.DEFAULT_OUTPUT_CSV))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=8192)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--stage2-lr-mult", type=float, default=0.6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stage1-label-smoothing", type=float, default=0.1)
    parser.add_argument("--stage2-label-smoothing", type=float, default=0.05)
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--max-train-samples", type=int, default=None, help="debug helper")
    parser.add_argument("--max-test-samples", type=int, default=None, help="debug helper")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-tta-flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
