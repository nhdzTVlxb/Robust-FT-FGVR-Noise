"""LoRA fine-tuning of CLIP ViT-B/32 on the kNN-denoised training set (thin entry).

Pipeline (fully scripted, reproducible, strict validation):
  1. Create the stratified train/val split BEFORE any label-driven step.
  2. kNN agreement, keep mask and teacher training use the training partition
     only; validation samples only query the training gallery.
  3. Recover dropped samples solely via teacher-kNN consensus pseudo-labels;
     kept samples get continuous reliability weights.
  4. Inject LoRA into the last N attention blocks of the ViT; train LoRA +
     cosine head (warm-started from the teacher) on images with augmentation,
     compact smoothed-label targets and sample weights.
  5. Evaluate per epoch on the held-out noisy val split (banded metrics).
  6. Save resumable checkpoints; --predict requires an explicit checkpoint
     policy (best / last / full).

Run:  python finetune_lora.py --epochs 15          (train, 90/10 split)
      python finetune_lora.py --epochs 15 --holdout-frac 0.1
                                                   (8:1:1; final unbiased holdout report)
      python finetune_lora.py --full --epochs 15   (train on 100% -> full.pt)
      python finetune_lora.py --predict --checkpoint full
      python finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.denoise import (
    confident_learning_keep,
    fit_gmm_1d,
    gmm_divide_select,
    knn_agreement,
    knn_majority_prediction,
    label_propagation,
    per_class_topk_keep,
    sinkhorn_balance,
)
from robustft.engine import (
    elr_regularizer,
    seed_everything,
    stratified_split,
    stratified_three_way_split,
    select_trusted_idx,
    teacher_stats,
    train_head,
    trust_project_grad,
)
from robustft.models import MODEL_NAME, build_lora_model
from robustft.robust_utils import choose_checkpoint, validate_disjoint_split
from robustft.submission import save_predictions, zip_submission


def _loss_vec(logits: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """Per-sample loss for the chosen robust objective (no soft-alpha; used by the
    mixup wrapper and the gce/sce/apl branches). Computed in fp32 for log stability."""
    if args.robust_loss == "gce":
        # Generalized Cross-Entropy (Zhang & Sabuncu 2018): L_q=(1-p_y^q)/q.
        p_y = logits.float().softmax(1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return (1.0 - p_y.pow(args.gce_q)) / args.gce_q
    if args.robust_loss == "sce":
        # Symmetric Cross-Entropy (Wang 2019): alpha*CE + beta*RCE; RCE is robust.
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=args.label_smoothing)
        p = logits.float().softmax(1).clamp(1e-7, 1.0)
        oh = F.one_hot(target, logits.size(1)).float().clamp_min(1e-4)
        rce = -(p * oh.log()).sum(1)
        return args.sce_alpha * ce + args.sce_beta * rce
    if args.robust_loss == "apl":
        # Active Passive Loss (Ma 2020): normalized CE (active) + RCE (passive),
        # theoretically robust to symmetric noise.
        lp = logits.float().log_softmax(1)
        nce = (-lp.gather(1, target.unsqueeze(1)).squeeze(1)) / (-lp.sum(1)).clamp_min(1e-6)
        p = lp.exp().clamp(1e-7, 1.0)
        oh = F.one_hot(target, logits.size(1)).float().clamp_min(1e-4)
        rce = -(p * oh.log()).sum(1)
        return args.apl_alpha * nce + args.apl_beta * rce
    return F.cross_entropy(logits, target, reduction="none", label_smoothing=args.label_smoothing)


def prepare_targets(args, device, train_idx: torch.Tensor, val_idx: torch.Tensor) -> dict:
    """Frozen-feature stage: kNN keep mask + teacher + compact targets/weights.

    Split-first: every label-driven statistic is computed from the training
    partition only. Returns everything index-aligned with ImageFolder ordering.
    """
    cache = torch.load(Path(args.cache_dir) / "train_features.pt", map_location="cpu")
    feats, labels, class_names = cache["features"], cache["labels"], cache["class_names"]
    image_names = cache["image_names"]
    num_classes = len(class_names)
    validate_disjoint_split(train_idx.tolist(), val_idx.tolist())
    f16 = feats.to(device)
    f32 = feats.to(device, dtype=torch.float32)
    y = labels.to(device)
    tr = train_idx.to(device)
    va = val_idx.to(device)
    ftr, ytr = f16[tr], y[tr]

    seed_everything(args.seed)
    agree = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    agree[tr] = knn_agreement(ftr, ytr, ftr, ytr, k=args.knn_k, exclude_self=True).to(device)
    if va.numel():
        agree[va] = knn_agreement(f16[va], y[va], ftr, ytr, k=args.knn_k, exclude_self=False).to(device)
    keep = torch.zeros_like(y, dtype=torch.bool)
    keep_tr = per_class_topk_keep(agree[tr], ytr, num_classes, keep_ratio=args.keep_ratio)
    keep_tr |= agree[tr] >= args.high_agreement_floor
    keep[tr] = keep_tr
    idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
    print(f"kNN filter keeps {idx_keep.numel()}/{y.numel()} ({idx_keep.numel() / y.numel():.2%})")

    teacher = train_head(f32, y, num_classes, idx_keep, smoothing=0.1, epochs=args.teacher_epochs,
                         batch_size=8192, device=device)
    p_label, preds_t, pmax_t, margins_t = teacher_stats(teacher, f32, y)
    # teacher_stats runs under inference_mode; clone its outputs before the OOF
    # branch replaces training-partition entries in place.
    p_label = p_label.clone()
    preds_t = preds_t.clone()
    pmax_t = pmax_t.clone()
    margins_t = margins_t.clone()
    teacher_probs_tr = None
    if args.teacher_oof_folds > 1:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all()
        numpy_rng_state = np.random.get_state()
        fold_ids = torch.empty(tr.numel(), dtype=torch.long, device=device)
        fold_gen = torch.Generator(device=device).manual_seed(args.seed + 1701)
        for cls in range(num_classes):
            cls_pos = torch.nonzero(ytr == cls, as_tuple=False).squeeze(1)
            order = cls_pos[torch.randperm(cls_pos.numel(), generator=fold_gen, device=device)]
            fold_ids[order] = torch.arange(order.numel(), device=device) % args.teacher_oof_folds
        global_to_tr = torch.full((y.numel(),), -1, dtype=torch.long, device=device)
        global_to_tr[tr] = torch.arange(tr.numel(), device=device)
        keep_pos = global_to_tr[idx_keep]
        teacher_probs_tr = torch.empty(tr.numel(), num_classes, dtype=torch.float32, device=device)
        for fold in range(args.teacher_oof_folds):
            held_pos = torch.nonzero(fold_ids == fold, as_tuple=False).squeeze(1)
            held_global = tr[held_pos]
            fold_train = idx_keep[fold_ids[keep_pos] != fold]
            fold_teacher = train_head(
                f32, y, num_classes, fold_train, smoothing=0.1,
                epochs=args.teacher_epochs, batch_size=8192, device=device,
            )
            with torch.no_grad():
                teacher_probs_tr[held_pos] = fold_teacher(f32[held_global]).softmax(1)
            del fold_teacher
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(numpy_rng_state)
        oof_top2 = teacher_probs_tr.topk(2, dim=1)
        preds_t[tr] = oof_top2.indices[:, 0]
        pmax_t[tr] = oof_top2.values[:, 0]
        margins_t[tr] = oof_top2.values[:, 0] - oof_top2.values[:, 1]
        p_label[tr] = teacher_probs_tr.gather(1, ytr[:, None]).squeeze(1)
        print(f"teacher OOF probabilities ready: {args.teacher_oof_folds} stratified folds", flush=True)
    knn_preds_tr = knn_majority_prediction(
        ftr, ftr, ytr, args.knn_k, num_classes, exclude_self=True)

    cl_relabel_info = None
    if args.denoise in ("cleanlab", "cleanlab_knn", "cleanlab_relabel", "ot", "otknn"):
        # Confident Learning on teacher probs. "ot": first Sinkhorn-balance the
        # probs to the uniform test prior (exploit balanced-test) before selection.
        P = teacher_probs_tr if teacher_probs_tr is not None else teacher(f32[tr]).softmax(1)
        if args.denoise in ("ot", "otknn"):
            P = sinkhorn_balance(P, iters=args.ot_iters)
        ck, cp, cc, ca = confident_learning_keep(P, ytr, num_classes)
        if args.denoise in ("cleanlab_knn", "otknn"):
            ck = ck & keep_tr  # stricter: agree AND confident
        if args.denoise in ("cleanlab_relabel", "ot", "otknn"):
            cl_relabel_info = (cp, cc, ca & (cp != ytr))  # relabel confident-wrong
        keep_tr = ck
    elif args.denoise == "labelprop":
        # propagate labels on the feature kNN graph from high-agreement anchors;
        # keep where propagation agrees with the label, relabel confident-disagree.
        anchors = torch.nonzero(agree[tr] >= args.lp_anchor_agree, as_tuple=False).squeeze(1)
        Z = label_propagation(F.normalize(f32[tr], dim=1), ytr, anchors, num_classes,
                              k=args.knn_k, alpha=args.lp_alpha)
        zp, zc = Z.argmax(1), Z.max(1).values
        has = Z.sum(1) > 1e-6
        keep_tr = ((zp == ytr) & has) | (~has)
        cl_relabel_info = (zp, zc, has & (zp != ytr))
    elif args.denoise == "divmix":
        # DivideMix-lite: GMM on per-sample teacher loss splits clean/noisy; noisy
        # samples are relabelled to the teacher prediction (semi-supervised co-refine).
        P = teacher(f32[tr]).softmax(1)
        ce = -(P.gather(1, ytr[:, None]).squeeze(1).clamp_min(1e-8).log())
        clean_prob = fit_gmm_1d(ce)
        keep_tr = clean_prob >= args.divmix_thresh
        tp, tc = P.argmax(1), P.max(1).values
        cl_relabel_info = (tp, tc * (1 - clean_prob), (clean_prob < args.divmix_thresh) & (tp != ytr))
    if args.denoise != "knn":
        keep[tr] = keep_tr
        idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
        print(f"denoise[{args.denoise}] keeps {int(keep_tr.sum())}/{tr.numel()} "
              f"({keep_tr.float().mean():.2%})")

    class_total = torch.bincount(ytr, minlength=num_classes).clamp_min(1).float()
    class_kept = torch.bincount(ytr[keep_tr], minlength=num_classes).float()
    class_clean_ratio = class_kept / class_total

    target_labels = y.clone()
    weights = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    # Composite reliability signals: (signal, weight). Combined either by product
    # (mul, aggressive: any low factor -> ~0 weight) or weighted average (sum,
    # conservative: no single factor can zero a sample out).
    sig_terms = [
        (agree[tr].clamp(0, 1), args.agreement_power),
        (p_label[tr].clamp(0, 1), args.confidence_power),
        (margins_t[tr].clamp(0, 1), args.margin_power),
    ]
    if args.proto_power > 0:
        # per-class visual prototype = centroid of kept samples; similarity is a
        # global per-class signal complementing local kNN agreement.
        ftr32 = f32[tr]
        kept_pos = torch.nonzero(keep_tr, as_tuple=False).squeeze(1)
        proto = torch.zeros(num_classes, ftr32.size(1), device=device)
        proto.index_add_(0, ytr[kept_pos], ftr32[kept_pos])
        cnt = torch.bincount(ytr[kept_pos], minlength=num_classes).clamp_min(1).unsqueeze(1)
        proto = F.normalize(proto / cnt, dim=1)
        proto_sim = (F.normalize(ftr32, dim=1) * proto[ytr]).sum(1).clamp(0, 1)
        sig_terms.append((proto_sim, args.proto_power))
    if args.aug_consist_power > 0 and Path(args.aug_consist_path).exists():
        ac = torch.load(args.aug_consist_path, map_location="cpu")["consistency"].to(device)
        sig_terms.append((ac[tr].clamp(0, 1), args.aug_consist_power))
    if args.reliability_mode == "sum":
        num = sum(w * s for s, w in sig_terms)
        den = sum(w for _, w in sig_terms)
        reliability = (num / max(den, 1e-6)).clamp(0, 1)
    else:
        reliability = torch.ones(tr.numel(), device=device)
        for s, w in sig_terms:
            reliability = reliability * s.pow(w)
    weights[tr[keep_tr]] = args.min_sample_weight + (1.0 - args.min_sample_weight) * reliability[keep_tr]
    pseudo_tr = (
        (~keep_tr)
        & (preds_t[tr] == knn_preds_tr)
        & (pmax_t[tr] >= args.pseudo_thresh)
        & (margins_t[tr] >= args.pseudo_margin)
    )
    soft_alpha = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    if pseudo_tr.any():
        pseudo_idx = tr[pseudo_tr]
        target_labels[pseudo_idx] = preds_t[pseudo_idx]
        weights[pseudo_idx] = pmax_t[pseudo_idx] * margins_t[pseudo_idx].sqrt()
        # soft fusion: keep alpha of the original label instead of hard-replacing
        # (conservative: don't fully trust the pseudo-label).
        soft_alpha[pseudo_idx] = args.pseudo_soft_alpha
    relabel_count = 0
    if cl_relabel_info is not None:
        # Confident-Learning relabel: confident-wrong samples get the CL-predicted
        # class with a confidence weight (overrides the consensus-pseudo path above
        # for these samples). Recovers corrected data instead of discarding it.
        cl_pred_tr, cl_conf_tr, relabel_tr = cl_relabel_info
        if relabel_tr.any():
            ridx = tr[relabel_tr]
            target_labels[ridx] = cl_pred_tr[relabel_tr]
            weights[ridx] = cl_conf_tr[relabel_tr].clamp(0, 1)
            soft_alpha[ridx] = 0.0  # hard-use the corrected label
            relabel_count = int(relabel_tr.sum())
    if args.class_balance:
        # counter noise-induced class imbalance: scale weights by inverse class
        # frequency (of the final used set) to the cb_power.
        used_m = weights > 0
        cnt = torch.bincount(target_labels[used_m], minlength=num_classes).clamp_min(1).float()
        cls_w = (cnt.mean() / cnt).pow(args.cb_power)
        weights = weights * cls_w[target_labels]
        print(f"class-balance(power={args.cb_power}): weight ratio max/min = "
              f"{cls_w.max() / cls_w.min():.2f}")
    pseudo_count = int(pseudo_tr.sum())
    print(f"targets ready: {int((weights > 0).sum())} usable samples "
          f"({pseudo_count} consensus pseudo-labelled, {relabel_count} CL-relabelled)")
    return {
        "target_labels": target_labels.cpu(), "weights": weights.cpu(), "labels": labels,
        "class_names": class_names, "image_names": image_names,
        "teacher_head_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
        "agree": agree.cpu(), "teacher_preds": preds_t.cpu(), "soft_alpha": soft_alpha.cpu(),
        "class_clean_ratio": class_clean_ratio.cpu(),
        "target_stats": {
            "kept": int(keep.sum()), "pseudo": pseudo_count,
            "mean_kept_weight": round(float(weights[keep].mean()), 6),
        },
    }


@torch.inference_mode()
def evaluate_images(model, loader, labels_gpu, agree_gpu, device, trusted_mask=None) -> dict:
    model.eval()
    correct = torch.zeros_like(labels_gpu, dtype=torch.bool)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        correct[idx.to(device)] = logits.argmax(1) == labels_gpu[idx.to(device)]
    bands = {
        "noisy_all": torch.ones_like(labels_gpu, dtype=torch.bool),
        "low_lt03": agree_gpu < 0.3,
        "mid_03_06": (agree_gpu >= 0.3) & (agree_gpu < 0.6),
        "high_ge06": agree_gpu >= 0.6,
    }
    if trusted_mask is not None:
        bands["trusted"] = trusted_mask  # high-agreement & teacher-consensus subset
    return {k: round((correct & m).sum().item() / max(1, int(m.sum())), 4) for k, m in bands.items()}


def train(args) -> None:
    device = torch.device("cuda")
    ckpt_dir = Path(args.work_dir) / "lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(Path(args.cache_dir) / "train_features.pt", map_location="cpu")
    labels = cache["labels"]
    ho_idx = torch.arange(0)
    if args.full:
        tr_idx = torch.arange(labels.numel())
        va_idx = torch.arange(0)
    elif args.holdout_frac > 0:
        tr_idx, va_idx, ho_idx = stratified_three_way_split(
            labels, args.val_frac, args.holdout_frac, args.seed)
    else:
        tr_idx, va_idx = stratified_split(labels, args.val_frac, args.seed)
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed)
        tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=generator)[:5000]].sort().values
        va_idx = va_idx[torch.randperm(va_idx.numel(), generator=generator)[:2000]].sort().values if va_idx.numel() else va_idx
        ho_idx = ho_idx[torch.randperm(ho_idx.numel(), generator=generator)[:1000]].sort().values if ho_idx.numel() else ho_idx
        args.epochs = 2

    # holdout samples are query-only, exactly like val: never in gallery/teacher
    prep = prepare_targets(args, device, tr_idx, torch.cat([va_idx, ho_idx]))
    class_names = prep["class_names"]
    num_classes = len(class_names)
    target_labels_all = prep["target_labels"].to(device)
    weights_all = prep["weights"].to(device)
    labels_all = prep["labels"].to(device)
    agree_all = prep["agree"].to(device)
    teacher_preds_all = prep["teacher_preds"].to(device)
    soft_alpha_all = prep["soft_alpha"].to(device)
    class_clean_ratio = prep["class_clean_ratio"].to(device)

    base = ImageFolder(args.train_dir)
    paths = [p for p, _ in base.samples]
    names_check = [Path(p).name for p in paths]
    assert names_check == prep["image_names"], "ImageFolder order mismatch with feature cache"

    usable = (weights_all[tr_idx.to(device)] > 0).cpu()
    recover_pool = None
    if args.ssl_recover:
        # keep cleanlab-dropped samples in the loader (weight 0 until pseudo-recovered)
        recover_pool = tr_idx[~usable].clone()
        print(f"train images {tr_idx.numel()} (usable {int(usable.sum())} + "
              f"ssl-recoverable {recover_pool.numel()})  val images {va_idx.numel()}")
    elif args.dynamic_divide:
        # keep the FULL training partition in the loader; the clean/noisy split is
        # re-decided each epoch by the GMM divide (weights below are overwritten).
        print(f"train images {tr_idx.numel()} (dynamic-divide: full partition, re-split each "
              f"epoch after {args.divide_warmup}ep static-clean warmup)  val images {va_idx.numel()}")
    else:
        tr_idx = tr_idx[usable]
        print(f"train images {tr_idx.numel()}  val images {va_idx.numel()}")

    model = build_lora_model(num_classes, args.lora_rank, args.lora_alpha, args.lora_dropout,
                             prep["teacher_head_state"], device, lora_blocks=args.lora_blocks,
                             lora_target=args.lora_target, img_size=args.img_size, peft=args.peft,
                             feat_fuse=args.feat_fuse, attn_pool=args.attn_pool,
                             pos_resample=args.pos_resample,
                             lora_front_rank=args.lora_front_rank,
                             lora_front_alpha=args.lora_front_alpha,
                             lora_split_block=args.lora_split_block,
                             lora_mid_rank=args.lora_mid_rank,
                             lora_mid_alpha=args.lora_mid_alpha,
                             lora_second_split_block=args.lora_second_split_block,
                             global_local_pool=args.global_local_pool)
    train_resize_scales = (
        [int(size) for size in args.train_resize_scales.split(",") if size.strip()]
        if args.train_resize_scales else None
    )
    train_tf, eval_tf = build_finetune_transforms(
        model.backbone,
        args.crop_min_scale,
        img_size=args.img_size,
        randaug=args.randaug,
        train_resize_scales=train_resize_scales,
    )
    if train_resize_scales:
        print(f"training resize scales: {train_resize_scales} -> model input {args.img_size}")
    if args.random_erasing:
        # RandomErasing (Zhong 2020): occlude a random rectangle -> occlusion-robust.
        train_tf = transforms.Compose([train_tf, transforms.RandomErasing(p=0.25)])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.2f}M")
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        # EMA is used as a teacher/evaluator; keep it deterministic while the
        # student model switches between train/eval modes.
        ema_model.eval()

    pin = not args.no_pin
    train_ds = IndexedImageDataset([paths[i] for i in tr_idx.tolist()], train_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    recover_loader = recover_pool_gpu = None
    if recover_pool is not None and recover_pool.numel():
        # weak-aug (eval_tf) view of the dropped pool, for EMA pseudo-labelling each epoch
        recover_ds = IndexedImageDataset([paths[i] for i in recover_pool.tolist()], eval_tf)
        recover_loader = DataLoader(recover_ds, batch_size=args.batch_size * 2, shuffle=False,
                                    num_workers=0, pin_memory=pin)
        recover_pool_gpu = recover_pool.to(device)
    divide_loader = None
    if args.dynamic_divide:
        # weak-aug (eval_tf) view of the WHOLE training partition; re-scored each epoch
        # to recompute the GMM clean/noisy split (order == tr_idx order).
        divide_ds = IndexedImageDataset([paths[i] for i in tr_idx.tolist()], eval_tf)
        divide_loader = DataLoader(divide_ds, batch_size=args.batch_size * 2, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=pin)
    val_loader = None
    if va_idx.numel():
        val_ds = IndexedImageDataset([paths[i] for i in va_idx.tolist()], eval_tf)
        # Windows: workers are full processes (~1GB each); keep val pool small and
        # persistent so it is spawned once instead of every epoch.
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=pin, persistent_workers=True)

    tr_idx_gpu = tr_idx.to(device)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = list(model.head.parameters())
    # TGP projects the gradient of ALL trainable params (LoRA + cosine head).
    tgp_params = [p for p in model.parameters() if p.requires_grad] if args.tgp else None
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-2},
    ])
    steps_per_epoch = len(train_loader)
    schedule_epochs = args.schedule_epochs or args.epochs
    if schedule_epochs < args.epochs:
        raise ValueError("--schedule-epochs must be >= --epochs")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lora_lr, args.head_lr], total_steps=schedule_epochs * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    if args.resume:
        resume = torch.load(Path(args.resume), map_location=device)
        model.load_state_dict(resume.get("raw_model", resume["model"]))
        if ema_model is not None:
            ema_model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        print(f"resuming {args.resume} from epoch {start_epoch}")

    def checkpoint_payload(epoch: int, bands: dict | None = None) -> dict:
        eval_model = ema_model if ema_model is not None else model
        payload_extra = {"raw_model": model.state_dict()} if ema_model is not None else {}
        return {
            "model": eval_model.state_dict(), "class_names": class_names,
            **payload_extra,
            "epoch": epoch, "bands": bands, "args": vars(args),
            "target_stats": prep["target_stats"],
            "train_idx": tr_idx, "val_idx": va_idx, "holdout_idx": ho_idx,
            "model_name": MODEL_NAME, "single_model": True,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
        }

    va_labels = labels_all[va_idx.to(device)] if va_idx.numel() else None
    va_agree = agree_all[va_idx.to(device)] if va_idx.numel() else None
    trusted_va = None
    if args.trusted_agree > 0 and va_idx.numel():
        # trustworthy mini-val: high kNN agreement AND teacher prediction == given
        # label -> labels here are very likely correct, a cleaner proxy for the
        # clean test set than the full noisy val.
        tp_va = teacher_preds_all[va_idx.to(device)]
        trusted_va = (va_agree >= args.trusted_agree) & (tp_va == va_labels)
    history = []
    best_mid = -1.0
    stale_epochs = 0
    # ELR per-sample running target (EMA of softmax predictions), indexed by global idx
    elr_target = torch.zeros(labels_all.numel(), num_classes, device=device) if args.elr else None
    # A-lite memory: after a short warmup, use previous-epoch per-sample loss to
    # gently downweight the highest-loss tail inside each class. This keeps the
    # original Series-B cleanlab/mixup trajectory, but softens likely noisy labels
    # instead of hard-dropping them.
    small_loss_memory = (
        torch.full((labels_all.numel(),), float("inf"), device=device)
        if args.small_loss_lite else None
    )
    ema_prev_pred = (
        torch.full((labels_all.numel(),), -1, dtype=torch.long, device=device)
        if args.small_loss_ema_stability_epochs > 0 else None
    )
    ema_stable_count = (
        torch.zeros(labels_all.numel(), dtype=torch.int16, device=device)
        if args.small_loss_ema_stability_epochs > 0 else None
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_seen = 0.0, 0
        tgp_fire = tgp_proj = 0  # TGP observability: clean-pass runs / projection actually changed grad
        small_clean_all = None
        small_loss_active = False
        if args.small_loss_lite and epoch > args.small_loss_start_epoch:
            finite = torch.isfinite(small_loss_memory[tr_idx_gpu])
            if finite.any():
                small_loss_active = True
                small_clean_all = torch.ones(labels_all.numel(), dtype=torch.bool, device=device)
                usable_tr = tr_idx_gpu[(weights_all[tr_idx_gpu] > 0) & finite]
                noisy_total = 0
                for cls in range(num_classes):
                    cls_idx = usable_tr[labels_all[usable_tr] == cls]
                    if cls_idx.numel() < 4:
                        continue
                    keep_ratio = args.small_loss_keep_ratio
                    if args.small_loss_adaptive_class:
                        keep_ratio = float(class_clean_ratio[cls].clamp(
                            args.small_loss_keep_min, args.small_loss_keep_max
                        ))
                    thresh = torch.quantile(small_loss_memory[cls_idx], keep_ratio)
                    cls_noisy = small_loss_memory[cls_idx] > thresh
                    small_clean_all[cls_idx[cls_noisy]] = False
                    noisy_total += int(cls_noisy.sum())
                print(
                    f"  small-loss-lite@ep{epoch}: noisy_tail={noisy_total}/{usable_tr.numel()} "
                    f"keep={'adaptive' if args.small_loss_adaptive_class else f'{args.small_loss_keep_ratio:.2f}'} "
                    f"noisy_w={args.small_loss_noisy_weight:.2f} "
                    f"ema_kl={args.small_loss_ema_kl_weight:.2f}",
                    flush=True,
                )
        # reliability curriculum: per-epoch threshold drops from a high quantile
        # (train only on the most reliable samples first) to 0 (use all by the end).
        thr_e = 0.0
        if args.curriculum:
            pos_w = weights_all[tr_idx_gpu]
            pos_w = pos_w[pos_w > 0]
            frac = args.curriculum_start * (1.0 - (epoch - 1) / max(1, args.epochs - 1))
            thr_e = float(torch.quantile(pos_w, frac)) if frac > 1e-6 else 0.0
        if recover_loader is not None and epoch >= args.ssl_warmup and (epoch - args.ssl_warmup) % args.ssl_every == 0:
            # self-training refresh: the improving EMA model pseudo-labels the dropped pool
            # (weak aug); confident-enough ones rejoin training at ssl_weight. Re-decided each
            # refresh so the recovered set tracks the model (FixMatch-style, early-learning gated).
            rec_model = ema_model if ema_model is not None else model
            rec_model.eval()
            weights_all[recover_pool_gpu] = 0.0
            n_rec = 0
            with torch.no_grad():
                for rimages, ridx in recover_loader:
                    rimages = rimages.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        rp = rec_model(rimages).float().softmax(1)
                    top2 = rp.topk(2, dim=1)
                    conf, pred = top2.values[:, 0], top2.indices[:, 0]
                    margin = top2.values[:, 0] - top2.values[:, 1]
                    gi = recover_pool_gpu[ridx.to(device)]
                    sel = (conf >= args.ssl_thresh) & (margin >= args.ssl_margin)
                    if sel.any():
                        target_labels_all[gi[sel]] = pred[sel]
                        weights_all[gi[sel]] = args.ssl_weight
                        n_rec += int(sel.sum())
            model.train()
            print(f"  ssl-recover@ep{epoch}: {n_rec}/{recover_pool.numel()} dropped pseudo-labelled "
                  f"(thr={args.ssl_thresh}, margin={args.ssl_margin}, w={args.ssl_weight})", flush=True)
        if divide_loader is not None and epoch >= args.divide_warmup and (epoch - args.divide_warmup) % args.divide_every == 0:
            # DivideMix self-cleaning: re-split the WHOLE partition from the EMA model's
            # per-sample loss. clean -> own label (w=1); noisy & confident -> pseudo-label
            # (w=divide_noisy_weight, treated as unlabelled); noisy & unsure -> dropped this
            # epoch. Re-decided every refresh so the clean set tracks the improving model.
            div_model = ema_model if ema_model is not None else model
            div_model.eval()
            n = tr_idx_gpu.numel()
            d_loss = torch.empty(n, device=device)
            d_conf = torch.empty(n, device=device)
            d_pred = torch.empty(n, dtype=torch.long, device=device)
            with torch.no_grad():
                for dimages, didx in divide_loader:
                    dimages = dimages.to(device, non_blocking=True)
                    didx = didx.to(device)
                    with torch.autocast("cuda", dtype=torch.float16):
                        dl = div_model(dimages).float()
                    gy = labels_all[tr_idx_gpu[didx]]
                    d_loss[didx] = F.cross_entropy(dl, gy, reduction="none")
                    dp = dl.softmax(1).max(1)
                    d_conf[didx] = dp.values
                    d_pred[didx] = dp.indices
            orig_tr = labels_all[tr_idx_gpu]
            new_t, new_w, w_clean = gmm_divide_select(
                d_loss, d_conf, d_pred, orig_tr,
                clean_thresh=args.divide_thresh, conf_gate=args.divide_conf_gate,
                clean_weight=1.0, noisy_weight=args.divide_noisy_weight)
            target_labels_all[tr_idx_gpu] = new_t
            weights_all[tr_idx_gpu] = new_w
            n_clean = int((w_clean >= args.divide_thresh).sum())
            n_pseudo = int(((new_w > 0) & (new_t != orig_tr)).sum())
            model.train()
            print(f"  divide@ep{epoch}: clean={n_clean}/{n} ({n_clean / n:.1%})  "
                  f"pseudo-relabelled noisy={n_pseudo}  dropped={int((new_w == 0).sum())}", flush=True)
        for images, idx in train_loader:
            images = images.to(device, non_blocking=True)
            need_stability = (
                args.small_loss_ema_stability_epochs > 0
                and epoch > args.small_loss_start_epoch - args.small_loss_ema_stability_epochs
            )
            clean_images = images.clone() if args.elr or need_stability else None
            gidx = tr_idx_gpu[idx.to(device)]
            tb = target_labels_all[gidx]
            wb = weights_all[gidx]
            if args.ssl_recover:
                active = wb > 0
                if not active.any():
                    continue
                images = images[active]
                gidx = gidx[active]
                tb = tb[active]
                wb = wb[active]
                if clean_images is not None:
                    clean_images = clean_images[active]
            if thr_e > 0:
                wb = wb * (wb >= thr_e).to(wb.dtype)
            opt.zero_grad(set_to_none=True)
            ema_probs_clean = None
            if need_stability:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    ema_probs_clean = ema_model(clean_images).float().softmax(1)
                ema_pred = ema_probs_clean.argmax(1)
                same = ema_prev_pred[gidx] == ema_pred
                ema_stable_count[gidx] = torch.where(
                    same, ema_stable_count[gidx] + 1, torch.ones_like(ema_stable_count[gidx])
                )
                ema_prev_pred[gidx] = ema_pred
            if args.tgp:
                tgp_fire += 1
                # TGP-v3 (TrustCLIP-style noisy-vs-trusted): on the PRE-mixup batch,
                # split into trusted (top-weight) and noisy (rest) subsets, take each
                # subset's gradient (fp32, no scaler -> numerically clean), remove the
                # noisy gradient's component that conflicts with the trusted direction,
                # then step on their sum. Engages by construction (noisy vs clean DO
                # disagree), unlike v2's full-vs-clean which never conflicted.
                ci = select_trusted_idx(wb, args.tgp_clean_frac, args.tgp_min_clean)
                nmask = wb > 0
                if ci.numel():
                    nmask = nmask.clone()
                    nmask[ci] = False
                ni = nmask.nonzero(as_tuple=True)[0]

                def _subset_grad(sub):
                    lv = F.cross_entropy(model(images[sub]), tb[sub], reduction="none",
                                         label_smoothing=args.label_smoothing)
                    l = (lv * wb[sub]).sum() / wb[sub].sum().clamp_min(1e-6)
                    parts = torch.autograd.grad(l, tgp_params, allow_unused=True)
                    flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                      for g, p in zip(parts, tgp_params)])
                    return l.detach(), flat

                if ci.numel() >= args.tgp_min_clean and ni.numel() >= 1:
                    cl, g_clean = _subset_grad(ci)
                    nl, g_noisy = _subset_grad(ni)
                    g_noisy_proj = trust_project_grad(g_noisy, g_clean, mode=args.tgp_mode)
                    if g_noisy_proj is not g_noisy:
                        tgp_proj += 1
                    g_total = g_clean + g_noisy_proj
                    loss_val = 0.5 * (cl.item() + nl.item())
                else:
                    pos = (wb > 0).nonzero(as_tuple=True)[0]
                    cl, g_total = _subset_grad(pos)
                    loss_val = cl.item()
                if torch.isfinite(g_total).all():
                    off = 0
                    for p in tgp_params:
                        nele = p.numel()
                        p.grad = g_total[off:off + nele].view_as(p).to(p.dtype).clone()
                        off += nele
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                sched.step()
                if ema_model is not None:
                    with torch.no_grad():
                        for pe, pm in zip(ema_model.parameters(), model.parameters()):
                            if pm.requires_grad:
                                pe.mul_(args.ema_decay).add_(pm, alpha=1.0 - args.ema_decay)
                tot_loss += loss_val * images.size(0)
                tot_seen += images.size(0)
                continue
            perm = None
            mixup_active = args.mixup_alpha > 0 and (
                args.mixup_end_epoch <= 0 or epoch <= args.mixup_end_epoch
            )
            if mixup_active and args.manifold_mixup == 0:
                # Mixup (Zhang 2018): convex-combine inputs + targets -> a strong
                # label-noise smoother, orthogonal to the kNN denoising pipeline.
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1.0 - lam) * images[perm]
            elif args.cutmix > 0 and args.manifold_mixup == 0:
                # CutMix (Yun 2019): paste a random patch from another image; the
                # label mix is the patch area fraction.
                lam = float(np.random.beta(args.cutmix, args.cutmix))
                perm = torch.randperm(images.size(0), device=device)
                H, W = images.shape[-2:]
                r = math.sqrt(1.0 - lam)
                rh, rw = int(H * r), int(W * r)
                cy, cx = np.random.randint(H), np.random.randint(W)
                y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
                x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
                images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
                lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
            with torch.autocast("cuda", dtype=torch.float16):
                elr_reg = 0.0
                if args.elr:
                    # clean (pre-mixup) forward -> running EMA target -> anchor term.
                    p_clean = model(clean_images).float().softmax(1)
                    with torch.no_grad():
                        cur = args.elr_beta * elr_target[gidx] + (1.0 - args.elr_beta) * p_clean
                        elr_target[gidx] = cur / cur.sum(1, keepdim=True).clamp_min(1e-8)
                    elr_reg = elr_regularizer(p_clean, elr_target[gidx], args.elr_lambda)
                if args.manifold_mixup > 0:
                    # Manifold Mixup (Verma 2019): mix the *feature* representation
                    # instead of the input -> flatter, noise-robust feature manifold.
                    lam = float(np.random.beta(args.manifold_mixup, args.manifold_mixup))
                    mperm = torch.randperm(images.size(0), device=device)
                    feat = model.extract_feat(images)
                    feat = lam * feat + (1.0 - lam) * feat[mperm]
                    logits = model.head(F.normalize(feat.float(), dim=-1))
                    loss_vec = lam * _loss_vec(logits, tb, args) + (1.0 - lam) * _loss_vec(logits, tb[mperm], args)
                    wb = lam * wb + (1.0 - lam) * wb[mperm]
                else:
                    logits = model(images)
                    if perm is not None:
                        loss_vec = lam * _loss_vec(logits, tb, args) + (1.0 - lam) * _loss_vec(logits, tb[perm], args)
                        wb = lam * wb + (1.0 - lam) * wb[perm]
                    elif args.robust_loss != "ce":
                        loss_vec = _loss_vec(logits, tb, args)
                    else:
                        loss_vec = F.cross_entropy(
                            logits, tb, reduction="none", label_smoothing=args.label_smoothing)
                        if args.pseudo_soft_alpha > 0:
                            ab = soft_alpha_all[gidx]
                            ce_orig = F.cross_entropy(
                                logits, labels_all[gidx], reduction="none", label_smoothing=args.label_smoothing)
                            loss_vec = (1.0 - ab) * loss_vec + ab * ce_orig
                if args.small_loss_lite:
                    with torch.no_grad():
                        tracked = _loss_vec(logits.detach(), tb, args)
                        if perm is not None:
                            tracked = lam * tracked + (1.0 - lam) * _loss_vec(logits.detach(), tb[perm], args)
                        prev = small_loss_memory[gidx]
                        small_loss_memory[gidx] = torch.where(
                            torch.isfinite(prev),
                            args.small_loss_memory_momentum * prev
                            + (1.0 - args.small_loss_memory_momentum) * tracked.detach().float(),
                            tracked.detach().float(),
                        )
                if small_loss_active and small_clean_all is not None:
                    noisy_mix = (~small_clean_all[gidx]).to(loss_vec.dtype)
                    if perm is not None:
                        noisy_mix = lam * noisy_mix + (1.0 - lam) * (~small_clean_all[gidx[perm]]).to(loss_vec.dtype)
                    if ema_model is not None and args.small_loss_ema_kl_weight > 0:
                        if ema_probs_clean is not None:
                            ema_probs = ema_probs_clean
                            stable_mix = (
                                ema_stable_count[gidx] >= args.small_loss_ema_stability_epochs
                            ).to(loss_vec.dtype)
                            if perm is not None:
                                ema_probs = lam * ema_probs + (1.0 - lam) * ema_probs[perm]
                                stable_mix = lam * stable_mix + (1.0 - lam) * stable_mix[perm]
                        else:
                            with torch.no_grad():
                                ema_probs = ema_model(images).float().softmax(1)
                            stable_mix = torch.ones_like(noisy_mix)
                        temp = args.small_loss_temperature
                        ema_kl = F.kl_div(
                            F.log_softmax(logits.float() / temp, dim=1),
                            (ema_probs.clamp_min(1e-8).log() / temp).softmax(1),
                            reduction="none",
                        ).sum(1) * (temp * temp)
                        soft_loss = (
                            (1.0 - args.small_loss_ema_kl_weight) * loss_vec
                            + args.small_loss_ema_kl_weight * ema_kl
                        )
                        kl_gate = noisy_mix * stable_mix
                        loss_vec = (1.0 - kl_gate) * loss_vec + kl_gate * soft_loss
                    wb = wb * ((1.0 - noisy_mix) + noisy_mix * args.small_loss_noisy_weight)
                loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6) + elr_reg
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        if pm.requires_grad:
                            pe.mul_(args.ema_decay).add_(pm, alpha=1.0 - args.ema_decay)
            tot_loss += loss.item() * images.size(0)
            tot_seen += images.size(0)
        msg = f"epoch {epoch:02d}/{args.epochs} loss={tot_loss / max(1, tot_seen):.4f} time={time.time() - t0:.0f}s"
        if args.tgp:
            msg += f" tgp_fire={tgp_fire} tgp_proj={tgp_proj}"
        entry = {"epoch": epoch, "loss": round(tot_loss / max(1, tot_seen), 4)}
        if val_loader is not None:
            bands = evaluate_images(ema_model if ema_model is not None else model,
                                    val_loader, va_labels, va_agree, device, trusted_mask=trusted_va)
            entry.update(bands)
            msg += "  " + "  ".join(f"{k}={v}" for k, v in bands.items())
            sel_key = "trusted" if "trusted" in bands else "mid_03_06"
            if bands[sel_key] > best_mid:
                best_mid = bands[sel_key]
                stale_epochs = 0
                torch.save(checkpoint_payload(epoch, bands), ckpt_dir / "best.pt")
                msg += "  *best*"
                if args.snapshot_after > 0 and epoch >= args.snapshot_after:
                    torch.save(checkpoint_payload(epoch, bands), ckpt_dir / f"best_ep{epoch:02d}.pt")
                    msg += "  [snap]"
            else:
                stale_epochs += 1
        print(msg, flush=True)
        history.append(entry)
        torch.save(checkpoint_payload(epoch), ckpt_dir / ("full.pt" if args.full else "last.pt"))
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(checkpoint_payload(epoch), ckpt_dir / f"ep{epoch:02d}.pt")
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        if val_loader is not None and args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(f"early stopping after {stale_epochs} stale epochs")
            break

    if ho_idx.numel():
        # free the persistent train/val worker processes before spawning new ones
        del train_loader, val_loader
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        ho_ds = IndexedImageDataset([paths[i] for i in ho_idx.tolist()], eval_tf)
        ho_loader = DataLoader(ho_ds, batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=2, pin_memory=not args.no_pin)
        ho_labels = labels_all[ho_idx.to(device)]
        ho_agree = agree_all[ho_idx.to(device)]
        bands = evaluate_images(model, ho_loader, ho_labels, ho_agree, device)
        print(f"holdout eval (untouched {args.holdout_frac:.0%}, best epoch {ckpt.get('epoch')}): "
              + "  ".join(f"{k}={v}" for k, v in bands.items()), flush=True)
        history.append({"holdout_best_epoch": ckpt.get("epoch"),
                        **{f"holdout_{k}": v for k, v in bands.items()}})
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
    print("done")


@torch.inference_mode()
def predict(args) -> None:
    device = torch.device("cuda")
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    targs = ckpt.get("args", {})
    model = build_lora_model(len(class_names), targs.get("lora_rank", args.lora_rank),
                             targs.get("lora_alpha", args.lora_alpha), 0.0, None, device,
                             lora_blocks=targs.get("lora_blocks", args.lora_blocks),
                             lora_target=targs.get("lora_target", args.lora_target),
                             img_size=targs.get("img_size", args.img_size),
                             peft=targs.get("peft", args.peft),
                             feat_fuse=targs.get("feat_fuse", args.feat_fuse),
                             attn_pool=targs.get("attn_pool", args.attn_pool),
                             pos_resample=targs.get("pos_resample", "timm"),
                             lora_front_rank=targs.get("lora_front_rank", 0),
                             lora_front_alpha=targs.get("lora_front_alpha", 0.0),
                             lora_split_block=targs.get("lora_split_block", 6),
                             lora_mid_rank=targs.get("lora_mid_rank", 0),
                             lora_mid_alpha=targs.get("lora_mid_alpha", 0.0),
                             lora_second_split_block=targs.get("lora_second_split_block", 8),
                             global_local_pool=targs.get("global_local_pool", False))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    _, eval_tf = build_finetune_transforms(model.backbone, targs.get("crop_min_scale", args.crop_min_scale),
                                           img_size=targs.get("img_size", args.img_size))
    test_dir = Path(args.test_dir)
    paths = sorted(str(p) for p in test_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ds = IndexedImageDataset(paths, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=not args.no_pin)
    preds = np.empty(len(paths), dtype=np.int64)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            if not args.no_flip_tta:
                logits = logits + model(torch.flip(images, dims=[3]))
        preds[idx.numpy()] = logits.argmax(1).cpu().numpy()

    out_csv = Path(args.output_csv)
    save_predictions(out_csv, [Path(p).name for p in paths], preds.tolist(), class_names)
    zip_path = zip_submission(out_csv)
    print(f"saved {out_csv} and {zip_path} ({len(paths)} predictions)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=str(config.TRAIN_DIR))
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR))
    p.add_argument("--cache-dir", default=None,
                   help="feature cache dir; defaults to <work-dir>/cache. Set explicitly "
                        "when using an isolated --work-dir for tests.")
    p.add_argument("--output-csv", default=str(config.DEFAULT_LORA_OUTPUT_CSV))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="carve out an untouched stratified test partition before val "
                        "(0.1 -> 8:1:1 split with an unbiased final report)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--schedule-epochs", type=int, default=None,
                   help="OneCycleLR horizon; defaults to --epochs. Set larger to stop early "
                        "while preserving a longer-run learning-rate trajectory.")
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=16)
    p.add_argument("--keep-ratio", type=float, default=0.75)
    p.add_argument("--denoise", default="knn",
                   choices=("knn", "cleanlab", "cleanlab_knn", "cleanlab_relabel",
                            "ot", "otknn", "labelprop", "divmix"),
                   help="clean-set selection: knn; cleanlab(*); ot=Sinkhorn-balance probs then CL; "
                        "otknn=ot∩knn; labelprop=feature-graph propagation; divmix=GMM loss-split co-refine")
    p.add_argument("--ot-iters", type=int, default=30, help="Sinkhorn iterations for ot/otknn")
    p.add_argument("--lp-alpha", type=float, default=0.8, help="label-propagation diffusion weight")
    p.add_argument("--lp-anchor-agree", type=float, default=0.6, help="kNN-agreement floor to be a propagation anchor")
    p.add_argument("--divmix-thresh", type=float, default=0.5, help="GMM clean-posterior threshold for divmix keep")
    p.add_argument("--ssl-recover", action="store_true",
                   help="FixMatch/self-training: instead of discarding cleanlab-dropped samples, keep "
                        "them as unlabeled; each epoch the (improving) EMA model pseudo-labels them with "
                        "a high-confidence gate and the confident ones rejoin training. Uses all data + "
                        "iterative self-training (orthogonal to the one-shot cleanlab drop).")
    p.add_argument("--ssl-thresh", type=float, default=0.95,
                   help="pseudo-label confidence gate for ssl-recover (only very-confident dropped samples rejoin)")
    p.add_argument("--ssl-margin", type=float, default=0.0,
                   help="minimum top1-top2 probability margin for ssl-recover")
    p.add_argument("--ssl-weight", type=float, default=0.5,
                   help="loss weight assigned to recovered samples (vs ~1.0 for clean)")
    p.add_argument("--ssl-warmup", type=int, default=2,
                   help="epochs of clean-only training before recovery starts (exploit early-learning)")
    p.add_argument("--ssl-every", type=int, default=1, help="refresh recovered pseudo-labels every K epochs")
    p.add_argument("--dynamic-divide", action="store_true",
                   help="DivideMix self-cleaning: each epoch re-split the WHOLE training partition into "
                        "clean/noisy with a GMM on the (EMA) training model's per-sample loss (not the "
                        "static frozen-feature teacher). Clean -> own label; noisy -> treated as unlabelled, "
                        "rejoining only via a high-confidence pseudo-label gate. Requires --ema-decay>0. "
                        "Mutually exclusive with --ssl-recover.")
    p.add_argument("--divide-warmup", type=int, default=2,
                   help="epochs trained on the static clean set before dynamic re-splitting starts "
                        "(exploit early-learning before trusting the model's own loss)")
    p.add_argument("--divide-every", type=int, default=1, help="re-run the GMM divide every K epochs")
    p.add_argument("--divide-thresh", type=float, default=0.5,
                   help="GMM clean-posterior threshold; >= -> clean")
    p.add_argument("--divide-conf-gate", type=float, default=0.95,
                   help="a noisy sample rejoins (as a pseudo-labelled unlabelled sample) only if its "
                        "model confidence >= this gate")
    p.add_argument("--divide-noisy-weight", type=float, default=0.5,
                   help="loss weight for confidently pseudo-labelled noisy samples (vs 1.0 for clean)")
    p.add_argument("--elr", action="store_true",
                   help="Early-Learning Regularization (Liu 2020): keep ALL data; add a term that anchors "
                        "each sample's prediction to its running EMA target (captured before the net memorises "
                        "noise). Single extra loss term, composes with mixup/cleanlab. Uses one extra clean "
                        "(pre-mixup) forward per step.")
    p.add_argument("--elr-lambda", type=float, default=3.0, help="ELR regularization weight (paper default 3)")
    p.add_argument("--elr-beta", type=float, default=0.7, help="ELR target EMA momentum (paper default 0.7)")
    p.add_argument("--tgp", action="store_true",
                   help="Trust-aligned Gradient Projection (TrustCLIP, ACM MM 2025), adapted: keep ALL data; "
                        "each step also compute a trusted gradient from the high-weight (clean) samples and "
                        "project the full-batch LoRA gradient so its component conflicting with the trusted "
                        "direction is removed. Neutralises noise-driven gradients without dropping data. "
                        "Uses one extra (clean-subset, pre-mixup) backward per step.")
    p.add_argument("--tgp-mode", choices=("conflict", "onto"), default="conflict",
                   help="conflict=PCGrad-style remove only conflicting component; onto=GPM-style keep only the "
                        "component inside the trusted subspace")
    p.add_argument("--tgp-clean-frac", type=float, default=0.5,
                   help="fraction of each batch (top by per-sample weight) used as the trusted clean subset")
    p.add_argument("--tgp-min-clean", type=int, default=4,
                   help="minimum trusted samples in a batch to run the projection (else that step is left untouched)")
    p.add_argument("--class-balance", action="store_true", help="scale sample weights by inverse class frequency")
    p.add_argument("--cb-power", type=float, default=0.5, help="class-balance exponent (0=off..1=full inverse-freq)")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix Beta(a,a) strength (0=off; mutually exclusive w/ mixup)")
    p.add_argument("--random-erasing", action="store_true", help="append RandomErasing(p=0.25) to train augmentation")
    p.add_argument("--pseudo-thresh", type=float, default=0.7)
    p.add_argument("--pseudo-margin", type=float, default=0.2)
    p.add_argument("--high-agreement-floor", type=float, default=0.7)
    p.add_argument("--min-sample-weight", type=float, default=0.2)
    p.add_argument("--agreement-power", type=float, default=1.0)
    p.add_argument("--confidence-power", type=float, default=0.5)
    p.add_argument("--margin-power", type=float, default=0.5)
    p.add_argument("--proto-power", type=float, default=0.0,
                   help="visual-class-prototype similarity factor in reliability (0=off)")
    p.add_argument("--aug-consist-power", type=float, default=0.0,
                   help="augmentation-consistency factor in reliability (0=off; needs aug-consist file)")
    p.add_argument("--aug-consist-path", default="outputs/cache/aug_consistency.pt")
    p.add_argument("--trusted-agree", type=float, default=0.0,
                   help="val mode: build a trusted-val (agree>=t & teacher==label) and select best by it (0=off)")
    p.add_argument("--reliability-mode", choices=("mul", "sum"), default="mul",
                   help="combine reliability signals by product (aggressive) or weighted avg (sum=conservative)")
    p.add_argument("--pseudo-soft-alpha", type=float, default=0.0,
                   help="soft pseudo-label: keep this fraction of the original label vs hard-replace (conservative)")
    p.add_argument("--teacher-epochs", type=int, default=20)
    p.add_argument("--teacher-oof-folds", type=int, default=1,
                   help="cross-fit frozen-feature teacher probabilities over this many stratified folds")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--robust-loss", choices=("ce", "gce", "sce", "apl"), default="ce",
                   help="gce=Generalized CE; sce=Symmetric CE; apl=Active-Passive (NCE+RCE)")
    p.add_argument("--gce-q", type=float, default=0.7)
    p.add_argument("--sce-alpha", type=float, default=0.1, help="SCE: CE weight")
    p.add_argument("--sce-beta", type=float, default=1.0, help="SCE: reverse-CE weight")
    p.add_argument("--apl-alpha", type=float, default=1.0, help="APL: normalized-CE weight")
    p.add_argument("--apl-beta", type=float, default=1.0, help="APL: reverse-CE weight")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Mixup Beta(a,a) strength (0=off; 0.2 mild, 0.4 stronger)")
    p.add_argument("--mixup-end-epoch", type=int, default=0,
                   help="last epoch using MixUp; 0 keeps MixUp active for all epochs")
    p.add_argument("--peft", choices=("lora", "dora"), default="lora",
                   help="dora = weight-decomposed LoRA (Liu 2024), strictly generalises lora")
    p.add_argument("--feat-fuse", type=int, default=0,
                   help="fuse CLS tokens of the last K transformer blocks (0=off=last layer only)")
    p.add_argument("--attn-pool", action="store_true",
                   help="pool last-block patch tokens with a learned attention query instead of CLS")
    p.add_argument("--global-local-pool", action="store_true",
                   help="retain global CLS and add gated attention-pooled patch evidence")
    p.add_argument("--manifold-mixup", type=float, default=0.0,
                   help="Manifold Mixup Beta(a,a) strength on features (0=off; mutually exclusive with --mixup-alpha)")
    p.add_argument("--curriculum", action="store_true",
                   help="reliability curriculum: train on most-reliable samples first, widen each epoch")
    p.add_argument("--curriculum-start", type=float, default=0.5,
                   help="curriculum: fraction of lowest-reliability samples excluded at epoch 1 (ramps to 0)")
    p.add_argument("--small-loss-lite", action="store_true",
                   help="A-lite: after warmup, classwise high-loss tail is gently downweighted and taught by EMA soft labels")
    p.add_argument("--small-loss-start-epoch", type=int, default=6,
                   help="activate small-loss-lite after this epoch (6 => starts at epoch 7)")
    p.add_argument("--small-loss-keep-ratio", type=float, default=0.90,
                   help="per-class fraction kept at normal weight; the high-loss tail is treated as noisy")
    p.add_argument("--small-loss-noisy-weight", type=float, default=0.70,
                   help="gentle multiplier for the noisy high-loss tail")
    p.add_argument("--small-loss-ema-kl-weight", type=float, default=0.20,
                   help="KL interpolation weight toward EMA soft labels for noisy samples")
    p.add_argument("--small-loss-temperature", type=float, default=1.0,
                   help="temperature for EMA soft-label KL")
    p.add_argument("--small-loss-memory-momentum", type=float, default=0.70,
                   help="EMA momentum for per-sample loss memory used by classwise small-loss")
    p.add_argument("--small-loss-adaptive-class", action="store_true",
                   help="derive each class keep ratio from its Cleanlab keep rate")
    p.add_argument("--small-loss-keep-min", type=float, default=0.88,
                   help="minimum per-class keep ratio in adaptive mode")
    p.add_argument("--small-loss-keep-max", type=float, default=0.96,
                   help="maximum per-class keep ratio in adaptive mode")
    p.add_argument("--small-loss-ema-stability-epochs", type=int, default=0,
                   help="apply noisy-sample EMA KL only after this many stable EMA predictions")
    p.add_argument("--save-every", type=int, default=0,
                   help="also snapshot ep<N>.pt every N epochs (periodic anchors)")
    p.add_argument("--snapshot-after", type=int, default=0,
                   help="val mode: snapshot best_ep<N>.pt on each new val-best after epoch N")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-front-rank", type=int, default=0,
                   help="optional lower rank for blocks before --lora-split-block; 0 keeps uniform LoRA")
    p.add_argument("--lora-front-alpha", type=float, default=0.0,
                   help="LoRA alpha for blocks before --lora-split-block")
    p.add_argument("--lora-split-block", type=int, default=6,
                   help="zero-based boundary between front and rear LoRA ranks")
    p.add_argument("--lora-mid-rank", type=int, default=0,
                   help="optional middle-band rank between the first and second split blocks")
    p.add_argument("--lora-mid-alpha", type=float, default=0.0,
                   help="LoRA alpha for the optional middle band")
    p.add_argument("--lora-second-split-block", type=int, default=8,
                   help="zero-based boundary between middle and rear LoRA ranks")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-blocks", type=int, default=12, choices=(4, 6, 12))
    p.add_argument("--lora-target", choices=("attn", "attn_mlp"), default="attn",
                   help="attn_mlp also adapts mlp fc1/fc2 (~3x trainable params)")
    p.add_argument("--img-size", type=int, default=224,
                   help="input resolution; >224 resamples CLIP pos-embeds (e.g. 448 -> 196 patch tokens)")
    p.add_argument("--pos-resample", choices=("timm", "aligned"), default="timm",
                   help="CLIP pos-embed resampling: 'timm'=default bicubic+antialias; "
                        "'aligned'=bilinear+align_corners=True (lossless anchors on odd grids, "
                        "e.g. 13x13@416 / 19x19@608)")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA of trainable weights for eval/checkpoints (try 0.999)")
    p.add_argument("--randaug", action="store_true",
                   help="RandAugment(2, 7) instead of ColorJitter")
    p.add_argument("--crop-min-scale", type=float, default=0.8)
    p.add_argument("--train-resize-scales", default="",
                   help="optional comma-separated random short-side resize scales before the training crop")
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--checkpoint", choices=("best", "last", "full"), default="best")
    p.add_argument("--resume", default=None, help="resume from a training checkpoint using the same schedule")
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--no-pin", action="store_true",
                   help="disable pinned host memory (low-RAM safety; pinned alloc can "
                        "deadlock when Windows is under memory pressure)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--predict", action="store_true")
    args = p.parse_args()
    # validate periodic-refresh divisors (used as modulo divisors in the epoch loop)
    if args.ssl_every <= 0:
        p.error("--ssl-every must be > 0")
    if not (0.0 <= args.ssl_margin <= 1.0):
        p.error("--ssl-margin must be in [0, 1]")
    if args.divide_every <= 0:
        p.error("--divide-every must be > 0")
    if args.dynamic_divide:
        if args.ssl_recover:
            p.error("--dynamic-divide and --ssl-recover are mutually exclusive")
        if args.ema_decay <= 0:
            p.error("--dynamic-divide requires --ema-decay > 0 (re-scores with the EMA model)")
    if args.small_loss_lite:
        if args.ema_decay <= 0 and args.small_loss_ema_kl_weight > 0:
            p.error("--small-loss-lite with EMA KL requires --ema-decay > 0")
        if not (0.0 < args.small_loss_keep_ratio <= 1.0):
            p.error("--small-loss-keep-ratio must be in (0, 1]")
        if not (0.0 <= args.small_loss_noisy_weight <= 1.0):
            p.error("--small-loss-noisy-weight must be in [0, 1]")
        if not (0.0 <= args.small_loss_ema_kl_weight <= 1.0):
            p.error("--small-loss-ema-kl-weight must be in [0, 1]")
        if args.small_loss_temperature <= 0:
            p.error("--small-loss-temperature must be > 0")
        if not (0.0 < args.small_loss_keep_min <= args.small_loss_keep_max <= 1.0):
            p.error("adaptive small-loss keep bounds must satisfy 0 < min <= max <= 1")
        if args.small_loss_ema_stability_epochs < 0:
            p.error("--small-loss-ema-stability-epochs must be >= 0")
    if args.mixup_end_epoch < 0:
        p.error("--mixup-end-epoch must be >= 0")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        p.error("--lora-rank and --lora-alpha must be > 0")
    if args.lora_front_rank < 0 or args.lora_front_alpha < 0:
        p.error("--lora-front-rank and --lora-front-alpha must be >= 0")
    if args.lora_front_rank > 0 and args.lora_front_alpha <= 0:
        p.error("--lora-front-alpha must be > 0 when layered LoRA is enabled")
    if not (0 <= args.lora_split_block <= 12):
        p.error("--lora-split-block must be in [0, 12]")
    if args.lora_mid_rank < 0 or args.lora_mid_alpha < 0:
        p.error("--lora-mid-rank/alpha must be non-negative")
    if args.lora_mid_rank > 0 and args.lora_mid_alpha <= 0:
        p.error("--lora-mid-alpha must be > 0 when --lora-mid-rank is enabled")
    if not (args.lora_split_block <= args.lora_second_split_block <= 12):
        p.error("--lora-second-split-block must be between --lora-split-block and 12")
    if args.teacher_oof_folds < 1:
        p.error("--teacher-oof-folds must be >= 1")
    if args.train_resize_scales:
        try:
            train_resize_scales = [int(size) for size in args.train_resize_scales.split(",") if size.strip()]
        except ValueError:
            p.error("--train-resize-scales must be comma-separated positive integers")
        if not train_resize_scales or any(size <= 0 for size in train_resize_scales):
            p.error("--train-resize-scales must be comma-separated positive integers")
    if args.tgp:
        if args.elr:
            p.error("--tgp and --elr are mutually exclusive (both add an extra backward; run separately)")
        if args.dynamic_divide:
            p.error("--tgp and --dynamic-divide are mutually exclusive")
        if not (0.0 < args.tgp_clean_frac <= 1.0):
            p.error("--tgp-clean-frac must be in (0, 1]")
        if args.tgp_min_clean <= 0:
            p.error("--tgp-min-clean must be > 0")
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.work_dir) / "cache")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)
