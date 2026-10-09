from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import Config
from model_utils import (
    ModelEMA,
    atomic_json,
    build_clip_model,
    build_optimizer,
    build_scheduler,
    build_teacher_model,
    build_train_val_split,
    clip_forward,
    data_signature,
    environment,
    inspect_model,
    make_loaders,
    mixup_cutmix_with_meta,
    kl_to_soft_targets_per_sample,
    soft_cross_entropy,
    soft_cross_entropy_per_sample,
    submission_ids,
    update_small_loss_weights,
    weighted_mean,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--amp",
        choices=["float16", "bfloat16", "float32"],
        default=None,
    )
    parser.add_argument("--class-map", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    return parser.parse_args()


def apply_args(cfg: Config, args):
    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)
        cfg.train_dir = cfg.data_root / "train"
        cfg.test_dir = cfg.data_root / "test"
    if args.train_dir is not None:
        cfg.train_dir = Path(args.train_dir)
        cfg.data_root = cfg.train_dir.parent
    if args.clip_model is not None:
        cfg.clip_model = Path(args.clip_model)
    if args.output is not None:
        cfg.output_dir = Path(args.output)
    for name in ("epochs", "batch_size", "num_workers", "val_split", "num_classes", "seed", "class_map"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.amp is not None:
        cfg.amp_dtype = args.amp
    cfg.resume = bool(args.resume)
    if args.stop_after_epoch is not None:
        cfg.stop_after_epoch = args.stop_after_epoch
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool, dtype):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def checkpoint_payload(
    epoch,
    vision,
    head,
    optimizer,
    scheduler,
    scaler,
    ema,
    cfg_dict,
    validation,
    history,
    best_macro,
    weights_kind,
    early_stop_counter=0,
    sample_weights=None,
    sample_loss=None,
):
    return {
        "epoch": epoch,
        "vision_state_dict": vision.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "ema_state_dict": ema.ema,
        "config": cfg_dict,
        "validation": validation,
        "history": history,
        "best_macro": best_macro,
        "early_stop_counter": early_stop_counter,
        "sample_weights": (
            sample_weights.detach().cpu() if sample_weights is not None else None
        ),
        "sample_loss": (
            sample_loss.detach().cpu() if sample_loss is not None else None
        ),
        "weights_kind": weights_kind,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_torch(path: Path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(vision, head, loader, device, amp_on, amp_dtype, num_classes):
    if loader is None:
        return None
    vision.eval()
    head.eval()
    class_correct = np.zeros(num_classes, dtype=np.float64)
    class_total = np.zeros(num_classes, dtype=np.float64)
    loss_sum = 0.0
    count = 0

    for images, labels in tqdm(loader, desc="[Validation]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp_on, amp_dtype):
            logits = clip_forward(vision, head, images)
        loss_sum += torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="sum"
        ).item()
        predicted = logits.argmax(dim=1)
        labels_np = labels.cpu().numpy()
        correct_np = predicted.eq(labels).cpu().numpy()
        class_total += np.bincount(labels_np, minlength=num_classes)
        class_correct += np.bincount(
            labels_np, weights=correct_np, minlength=num_classes
        )
        count += labels.numel()

    valid = class_total > 0
    return {
        "loss": loss_sum / max(1, count),
        "micro": float(class_correct.sum() / max(1, class_total.sum())),
        "macro": float((class_correct[valid] / class_total[valid]).mean()),
    }


@torch.no_grad()
def ema_teacher_logits(
    vision,
    head,
    combined,
    ema,
    images,
    device,
    amp_on,
    amp_dtype,
):
    """Generate soft targets from the current EMA student without gradients."""
    was_training_vision = vision.training
    was_training_head = head.training
    ema.apply_shadow(combined)
    try:
        vision.eval()
        head.eval()
        with autocast_context(device, amp_on, amp_dtype):
            logits = clip_forward(vision, head, images)
        return logits.detach()
    finally:
        ema.restore(combined)
        if was_training_vision:
            vision.train()
        if was_training_head:
            head.train()


def run(cfg: Config, device: torch.device):
    if cfg.epochs <= cfg.warmup_epochs:
        raise ValueError("epochs must exceed warmup_epochs")
    if cfg.batch_size < 1 or cfg.num_workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if not 0.0 <= cfg.val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    if cfg.stop_after_epoch is not None and not 1 <= cfg.stop_after_epoch <= cfg.epochs:
        raise ValueError("stop_after_epoch must be within the configured epoch range")
    if cfg.early_stop_patience is not None and cfg.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be positive")

    cfg.train_dir = Path(cfg.train_dir).resolve()
    cfg.clip_model = Path(cfg.clip_model).resolve()
    cfg.output_dir = Path(cfg.output_dir).resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    inspect_model(str(cfg.clip_model))
    train_samples, val_samples, class_names = build_train_val_split(
        str(cfg.train_dir), cfg.val_split, cfg.seed
    )
    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"found {len(class_names)} classes, expected {cfg.num_classes}"
        )
    labels = submission_ids(class_names, cfg.class_map)
    if len(train_samples) < cfg.batch_size:
        raise ValueError("training samples are fewer than batch_size")
    train_targets = torch.tensor(
        [label for _, label in train_samples], dtype=torch.long
    )

    manifest, sample_hash = data_signature(
        train_samples,
        val_samples,
        class_names,
        labels,
        cfg.data_root.resolve(),
    )
    cfg_dict = {
        "recipe": "v35_a0_v18_style",
        "image_size": cfg.image_size,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_targets": list(cfg.lora_targets),
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_epochs": cfg.warmup_epochs,
        "ema_decay": cfg.ema_decay,
        "mixup_alpha": cfg.mixup_alpha,
        "cutmix_alpha": cfg.cutmix_alpha,
        "randaugment_num_ops": cfg.randaugment_num_ops,
        "randaugment_magnitude": cfg.randaugment_magnitude,
        "crop_min": cfg.crop_min,
        "distill_weight": cfg.distill_weight,
        "teacher_dtype": "float32",
        "loss": "small-loss reweighted CE + EMA soft-label KL + cosine feature distillation",
        "use_small_loss": cfg.use_small_loss,
        "small_loss_warmup_epochs": cfg.small_loss_warmup_epochs,
        "clean_ratio": cfg.clean_ratio,
        "use_clean_ratio_schedule": cfg.use_clean_ratio_schedule,
        "clean_ratio_epoch_6_10": cfg.clean_ratio_epoch_6_10,
        "clean_ratio_epoch_11_25": cfg.clean_ratio_epoch_11_25,
        "clean_ratio_epoch_after_25": cfg.clean_ratio_epoch_after_25,
        "clean_weight": cfg.clean_weight,
        "noisy_weight": cfg.noisy_weight,
        "use_ema_soft_label": cfg.use_ema_soft_label,
        "noisy_hard_ce_weight": cfg.noisy_hard_ce_weight,
        "noisy_ema_kl_weight": cfg.noisy_ema_kl_weight,
        "ema_soft_temperature": cfg.ema_soft_temperature,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "workers": cfg.num_workers,
        "val_split": cfg.val_split,
        "num_classes": cfg.num_classes,
        "seed": cfg.seed,
        "amp": cfg.amp_dtype,
        "early_stop_patience": cfg.early_stop_patience,
        "sample_order_hash": sample_hash,
        "class_names": class_names,
        "submission_ids": labels,
        "clip_model": str(cfg.clip_model),
    }

    config_path = cfg.output_dir / "config.json"
    split_path = cfg.output_dir / "split_manifest.json"
    last_path = cfg.output_dir / "last.pth"
    best_path = cfg.output_dir / "best.pth"
    if cfg.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"missing resume checkpoint: {last_path}")
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != cfg_dict:
            raise ValueError("resume configuration or data split changed")
    elif any(cfg.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {cfg.output_dir}; use --resume or a new path"
        )
    atomic_json(config_path, cfg_dict)
    atomic_json(split_path, manifest)
    atomic_json(cfg.output_dir / "environment.json", environment())

    print(
        f"[Data] train={len(train_samples)}, val={len(val_samples)}, "
        f"classes={len(class_names)}, device={device}",
        flush=True,
    )

    teacher, actual_size = build_teacher_model(str(cfg.clip_model), cfg.image_size)
    teacher = teacher.to(device).eval()
    vision, head, actual_size = build_clip_model(
        str(cfg.clip_model),
        cfg.num_classes,
        actual_size,
        cfg.lora_rank,
        cfg.lora_alpha,
        cfg.lora_dropout,
        cfg.lora_targets,
    )
    vision = vision.to(device)
    head = head.to(device)
    train_loader, val_loader = make_loaders(
        cfg, train_samples, val_samples, actual_size
    )
    optimizer = build_optimizer(
        vision, head, cfg.learning_rate, cfg.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, cfg.warmup_epochs, cfg.epochs, len(train_loader)
    )
    amp_dtype = getattr(torch, cfg.amp_dtype)
    amp_on = device.type == "cuda" and cfg.amp_dtype != "float32"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_on and cfg.amp_dtype == "float16",
    )
    combined = nn.Sequential(vision, head)
    ema = ModelEMA(combined, cfg.ema_decay)

    start_epoch = 1
    best_macro = -1.0
    early_stop_counter = 0
    history = []
    # Keep sample-level statistics on CPU; only the current batch's weights
    # are copied to CUDA.  This avoids device-indexing overhead and makes
    # checkpoint/resume lightweight.
    sample_weights = torch.ones(len(train_samples), dtype=torch.float32)
    # Samples omitted by drop_last in a given epoch stay at +inf until seen;
    # they must not be incorrectly promoted to the clean subset.
    sample_loss = torch.full(
        (len(train_samples),), float("inf"), dtype=torch.float32
    )
    sample_loss_sum = torch.zeros(len(train_samples), dtype=torch.float32)
    sample_seen = torch.zeros(len(train_samples), dtype=torch.float32)
    if cfg.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["config"] != cfg_dict:
            raise ValueError("checkpoint configuration mismatch")
        vision.load_state_dict(state["vision_state_dict"], strict=True)
        head.load_state_dict(state["head_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        ema.ema = state["ema_state_dict"]
        start_epoch = int(state["epoch"]) + 1
        best_macro = float(state["best_macro"])
        early_stop_counter = int(state.get("early_stop_counter", 0))
        history = state["history"]
        if state.get("sample_weights") is not None:
            sample_weights = state["sample_weights"].float().cpu()
        if state.get("sample_loss") is not None:
            sample_loss = state["sample_loss"].float()
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
        print(f"[Resume] starting epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        vision.train()
        head.train()
        totals = np.zeros(3, dtype=np.float64)
        seen = 0
        sample_loss_sum.zero_()
        sample_seen.zero_()
        reweight_active = (
            bool(cfg.use_small_loss)
            and epoch > int(cfg.small_loss_warmup_epochs)
        )
        current_weights = (
            sample_weights.to(device)
            if reweight_active
            else torch.ones(len(train_samples), device=device)
        )
        noisy_count = int((current_weights < 0.999).sum().item())
        total_weight = 0.0

        for step, (images, labels_batch, sample_indices) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        ):
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            sample_indices = sample_indices.to(device, non_blocking=True)
            batch_weights = current_weights[sample_indices]
            images, soft_targets, permutation, mix_lam, mixed = mixup_cutmix_with_meta(
                images,
                labels_batch,
                cfg.mixup_alpha,
                cfg.cutmix_alpha,
                cfg.num_classes,
                device,
            )
            if mixed:
                effective_weights = (
                    mix_lam * batch_weights
                    + (1.0 - mix_lam) * batch_weights[permutation]
                )
            else:
                effective_weights = batch_weights

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_on, amp_dtype):
                logits, features = clip_forward(
                    vision, head, images, return_features=True
                )
                hard_loss_vec = soft_cross_entropy_per_sample(logits, soft_targets)

                loss_vec = hard_loss_vec
                if (
                    reweight_active
                    and cfg.use_ema_soft_label
                    and ema is not None
                ):
                    teacher_logits = ema_teacher_logits(
                        vision,
                        head,
                        combined,
                        ema,
                        images,
                        device,
                        amp_on,
                        amp_dtype,
                    )
                    kl_vec = kl_to_soft_targets_per_sample(
                        logits,
                        teacher_logits,
                        cfg.ema_soft_temperature,
                    )
                    noisy_strength = (
                        (1.0 - effective_weights)
                        / max(1e-6, 1.0 - float(cfg.noisy_weight))
                    ).clamp(0.0, 1.0)
                    noisy_vec = (
                        float(cfg.noisy_hard_ce_weight) * hard_loss_vec
                        + float(cfg.noisy_ema_kl_weight) * kl_vec
                    )
                    loss_vec = (
                        (1.0 - noisy_strength) * hard_loss_vec
                        + noisy_strength * noisy_vec
                    )

                task_loss = (
                    weighted_mean(loss_vec, effective_weights)
                    if reweight_active
                    else loss_vec.mean()
                )

            # The teacher is deliberately outside autocast: FP32 frozen CLIP.
            with torch.no_grad():
                target_features = teacher(pixel_values=images).pooler_output
            distill_loss = (
                1.0
                - F_cosine(features.float(), target_features.float(), dim=1)
            ).mean()
            loss = task_loss + cfg.distill_weight * distill_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch}, step={step}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            trainable = [
                parameter
                for parameter in list(vision.parameters()) + list(head.parameters())
                if parameter.requires_grad
            ]
            nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(combined)

            batch_size = labels_batch.numel()
            totals += np.array(
                [loss.item(), task_loss.item(), distill_loss.item()]
            ) * batch_size
            seen += batch_size
            total_weight += effective_weights.detach().sum().item()

            with torch.no_grad():
                indices_cpu = sample_indices.detach().cpu()
                sample_loss_sum[indices_cpu] += hard_loss_vec.detach().float().cpu()
                sample_seen[indices_cpu] += 1.0

        # Evaluate EMA weights, then restore the raw weights for checkpoint/resume.
        raw_vision = {key: value.detach().clone() for key, value in vision.state_dict().items()}
        raw_head = {key: value.detach().clone() for key, value in head.state_dict().items()}
        ema.apply_to(combined)
        validation = validate(
            vision,
            head,
            val_loader,
            device,
            amp_on,
            amp_dtype,
            cfg.num_classes,
        )
        is_best = validation is not None and validation["macro"] > best_macro
        if validation is None:
            is_best = epoch == cfg.epochs
        if is_best:
            best_macro = validation["macro"] if validation is not None else best_macro
            early_stop_counter = 0
            save_torch(
                best_path,
                checkpoint_payload(
                    epoch,
                    vision,
                    head,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    cfg_dict,
                    validation,
                    history,
                    best_macro,
                    "EMA",
                    early_stop_counter,
                    sample_weights,
                    sample_loss,
                ),
            )
        elif validation is not None:
            early_stop_counter += 1
        vision.load_state_dict(raw_vision, strict=True)
        head.load_state_dict(raw_head, strict=True)
        del raw_vision, raw_head

        row = {
            "epoch": epoch,
            "train_loss": float(totals[0] / max(1, seen)),
            "task_loss": float(totals[1] / max(1, seen)),
            "distill_loss": float(totals[2] / max(1, seen)),
            "validation": validation,
            "seconds": round(time.time() - epoch_start, 3),
            "reweight_active": reweight_active,
            "noisy_count": noisy_count,
            "mean_sample_weight": float(
                current_weights.mean().item()
            ),
            "mean_batch_weight": float(total_weight / max(1, seen)),
        }
        history.append(row)

        if cfg.use_small_loss:
            sample_weights, sample_loss = update_small_loss_weights(
                sample_loss_sum,
                sample_seen,
                sample_loss,
                train_targets,
                sample_weights,
                cfg,
                epoch,
            )

        save_torch(
            last_path,
            checkpoint_payload(
                epoch,
                vision,
                head,
                optimizer,
                scheduler,
                scaler,
                ema,
                cfg_dict,
                validation,
                history,
                best_macro,
                "raw_resume",
                early_stop_counter,
                sample_weights,
                sample_loss,
            ),
        )
        atomic_json(cfg.output_dir / "history.json", history)
        atomic_json(
            cfg.output_dir / "status.json",
            {
                "epoch": epoch,
                "epochs": cfg.epochs,
                "state": "complete" if epoch == cfg.epochs else "running",
                "latest": row,
                "best_macro": best_macro,
                "early_stop_counter": early_stop_counter,
            },
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if (
            validation is not None
            and cfg.early_stop_patience is not None
            and early_stop_counter >= cfg.early_stop_patience
        ):
            atomic_json(
                cfg.output_dir / "status.json",
                {
                    "epoch": epoch,
                    "epochs": cfg.epochs,
                    "state": "early_stopped",
                    "latest": row,
                    "best_macro": best_macro,
                    "early_stop_counter": early_stop_counter,
                },
            )
            print(
                f"[Early Stop] no macro improvement for {cfg.early_stop_patience} epochs; "
                f"best_macro={best_macro:.6f}",
                flush=True,
            )
            break

        if cfg.stop_after_epoch is not None and epoch >= cfg.stop_after_epoch:
            if epoch < cfg.epochs:
                atomic_json(
                    cfg.output_dir / "status.json",
                    {"epoch": epoch, "epochs": cfg.epochs, "state": "paused"},
                )
            break

    print(f"[Done] checkpoints and logs: {cfg.output_dir}", flush=True)


def F_cosine(x, y, dim=1):
    return torch.nn.functional.cosine_similarity(x, y, dim=dim)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = apply_args(Config(), args)
    set_seed(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run(cfg, device)


if __name__ == "__main__":
    main()
