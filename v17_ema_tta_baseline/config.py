from pathlib import Path
from typing import Optional


class Config:
    # =========================
    # Path
    # =========================
    train_dir = Path("/home/cyp/speedsci/A4/train")
    pretrained_path = Path("/home/cyp/speedsci/A4/ViT-B-32.pt")
    checkpoint_dir = Path("/home/cyp/speedsci/A4/checkpoint_v17_ema_tta_baseline")
    log_path = checkpoint_dir / "log.txt"
    version_name = "v17_ema_tta_baseline"

    # Normal non-leaky validation run. Train/val are split 9:1.
    train_on_all = False

    # epoch 1-8: classifier only, LoRA frozen
    # epoch 9+: LoRA + classifier
    head_warmup_epochs = 8

    # MLP classifier head: 512 -> 1024 -> GELU -> Dropout(0.2) -> num_classes
    classifier_head = "mlp"
    mlp_hidden_dim = 1024
    mlp_dropout = 0.2

    # =========================
    # Model
    # =========================
    clip_model_name = "ViT-B/32"
    num_classes = 500

    # 输入尺寸
    image_size = 512

    # LoRA
    lora_rank = 8
    lora_alpha = 16
    lora_target = "attn+mlp"  # attn / mlp / attn+mlp

    # 所有层都加 LoRA
    # CLIP ViT-B/32 一共 12 层
    lora_last_n_layers = 12

    # =========================
    # Training
    # =========================
    seed = 42
    epochs = 200
    batch_size = 64
    num_workers = 8

    learning_rate = 2e-4       # LoRA lr
    lr_classifier = 1e-3       # classifier lr
    weight_decay = 0.01
    warmup_epochs = 3
    label_smoothing = 0.1
    grad_clip = 1.0

    use_amp = True

    # =========================
    # EMA
    # =========================
    use_ema = True
    ema_decay = 0.9995

    # 额外的一致性正则，基线里关闭
    use_feature_consistency = False


    # =========================
    # CLIP Prototype Cleaning
    # =========================
    # 用原始冻结 CLIP 特征做每类 prototype，给疑似噪声样本降权。
    use_prototype_cleaning = False
    prototype_cache_path = checkpoint_dir / "prototype_weights.pt"
    prototype_force_recompute = False
    prototype_image_size = 512
    prototype_batch_size = 48

    # 每类 prototype 相似度排名靠前的样本保留权重 1.0，靠后的样本降权。
    prototype_clean_ratio = 0.85
    prototype_clean_weight = 1.0
    prototype_noisy_weight = 0.5

    # 先用全类均值算初始 prototype，再保留相似度最高的部分样本重算 refined prototype。
    prototype_refine = True
    prototype_refine_ratio = 0.80

    # =========================
    # Small-loss Reweight
    # =========================
    use_loss_reweight = False

    # 前 5 轮不筛样本，第 6 轮开始用上一轮 loss 权重
    reweight_warmup_epochs = 8

    # 每类保留 loss 最小的 80% 样本
    clean_ratio = 0.8

    # Curriculum clean ratio:
    # epoch 1-5: no reweight
    # epoch 6-10: 0.9
    # epoch 11-20: 0.85
    # epoch >20: 0.8
    use_clean_ratio_schedule = True
    clean_ratio_epoch_6_10 = 0.9
    clean_ratio_epoch_11_20 = 0.85
    clean_ratio_epoch_after_20 = 0.8

    # 每类 loss 较大的 20% 样本权重
    noisy_weight = 0.3

    # 干净样本权重
    clean_weight = 1.0

    # Noisy samples: 0.3 * hard-label CE + 0.7 * EMA soft-label KL
    use_noisy_ema_soft = False
    noisy_hard_ce_weight = 0.3
    noisy_ema_kl_weight = 0.7
    ema_soft_temperature = 1.0

    # MixUp / CutMix
    use_mix_aug = False
    mixup_alpha = 0.1
    cutmix_alpha = 0.5
    mix_aug_prob = 0.3

    # =========================
    # Validation
    # =========================
    val_ratio = 0.1

    # =========================
    # EMA + TTA Validation
    # =========================
    # 只记录 ema_tta_acc，不保存 checkpoint_best_tta.pt
    use_ema_tta_val = True
    tta_sizes = [448, 512]
    tta_hflip = True

    # =========================
    # Early Stopping
    # =========================
    early_stop_patience = 20
    early_stop_metric = "ema_acc"

    # Resume
    resume_from: Optional[str] = None
