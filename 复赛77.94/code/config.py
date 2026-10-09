from pathlib import Path


class Config:
    # 路径由 train.py / predict.py 的命令行参数覆盖
    data_root = Path("/my-data/aic4")
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    clip_model = Path("/my-data/aic4/models/clip-vit-base-patch32")
    output_dir = Path("/my-data/aic4/checkpoint_exp22_750_b512_50ep")

    # 任务
    num_classes = 750
    seed = 42
    class_map = None

    image_size = 512
    epochs = 50
    batch_size = 96
    num_workers = 8
    val_split = 0.0
    amp_dtype = "float16"

    # LoRA：覆盖全部 12 个 ViT block 的 q/k/v/out 与 fc1/fc2
    lora_rank = 128
    lora_alpha = 256
    lora_dropout = 0.2
    lora_targets = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )

    # 优化器与学习率
    learning_rate = 1e-4
    weight_decay = 0.05
    warmup_epochs = 5
    grad_clip = 1.0

    # 增强与损失
    crop_min = 0.5
    randaugment_num_ops = 4
    randaugment_magnitude = 18
    mixup_alpha = 0.25
    cutmix_alpha = 0.25
    distill_weight = 1.0

    # EMA
    ema_decay = 0.9998

    # Small-loss reweighting + EMA soft-label correction.
    use_small_loss = True
    small_loss_warmup_epochs = 8
    clean_ratio = 0.8
    use_clean_ratio_schedule = True
    clean_ratio_epoch_6_10 = 0.95
    clean_ratio_epoch_11_25 = 0.9
    clean_ratio_epoch_after_25 = 0.85
    clean_weight = 1.0
    noisy_weight = 0.3
    use_ema_soft_label = True
    noisy_hard_ce_weight = 0.3
    noisy_ema_kl_weight = 0.7
    ema_soft_temperature = 1.0

    # 推理
    tta_mode = "legacy14"
    tta_hflip = True

    # 工程
    resume = False
    stop_after_epoch = None
    early_stop_patience = 7
