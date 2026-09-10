"""
Prediction script for CLIP ViT-B/32 + LoRA training code.

适配当前训练代码：
- 后 12 层 LoRA
- checkpoint_best_ema_tta.pt
- EMA 权重推理
- EMA + TTA 推理
- 输出格式：filename,label  中间无空格

默认：
test_dir:
/home/cyp/speedsci/A4/test

checkpoint:
/home/cyp/speedsci/A4/checkpoint_v18_dynamic_robust/checkpoint_best_ema_tta.pt

output:
/home/cyp/speedsci/A4/checkpoint_v18_dynamic_robust/pred_v18_dynamic_robust.csv
"""

import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
from tqdm import tqdm

from config import Config
from model_utils import build_clip_lora

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================================
# Transforms
# ============================================================================
def get_val_transform(size: int):
    resize_size = int(size * 256 / 224)

    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def parse_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================================
# Dataset
# ============================================================================
class TestDataset(Dataset):
    def __init__(self, test_dir: str, recursive: bool = False):
        self.test_dir = Path(test_dir)

        if not self.test_dir.exists():
            raise FileNotFoundError(f"测试目录不存在: {self.test_dir}")

        iterator = self.test_dir.rglob("*") if recursive else self.test_dir.iterdir()

        self.image_paths = sorted([
            p for p in iterator
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(f"测试目录没有找到图片: {self.test_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"读取图片失败: {path}, error={e}")

        return image, path.name


def pil_collate_fn(batch):
    images, names = zip(*batch)
    return list(images), list(names)


# ============================================================================
# Label mapping
# ============================================================================
def build_id_to_class(cfg: Config) -> Dict[int, str]:
    """
    ImageFolder 训练时类别按文件夹名排序。
    推理时要把预测 index 映射回文件夹名，例如 0000 / 0001 / 0123。
    """
    train_dir = Path(cfg.train_dir)

    if train_dir.exists():
        classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

        if len(classes) == cfg.num_classes:
            return {i: cls_name for i, cls_name in enumerate(classes)}

        print(
            f"[Warning] train_dir 类别数={len(classes)} 与 cfg.num_classes={cfg.num_classes} 不一致，"
            f"将使用 pred:04d 输出。"
        )
    else:
        print(f"[Warning] train_dir 不存在: {train_dir}，将使用 pred:04d 输出。")

    return {i: f"{i:04d}" for i in range(cfg.num_classes)}


# ============================================================================
# Checkpoint loading
# ============================================================================
def copy_ema_shadow_to_model(model, classifier, ema_state):
    """
    checkpoint_best_ema.pt 里保存了：
    - model_state: 当前 raw 权重
    - classifier_state: 当前 raw 分类头
    - ema_state: EMA shadow 权重

    如果推理想用 EMA，需要把 ema_state['shadow'] 拷贝回模型。
    """
    if ema_state is None or "shadow" not in ema_state:
        return False

    shadow = ema_state["shadow"]

    named_params = {}

    for name, p in model.named_parameters():
        named_params["model." + name] = p

    for name, p in classifier.named_parameters():
        named_params["classifier." + name] = p

    copied = 0
    missing = []

    with torch.no_grad():
        for name, ema_tensor in shadow.items():
            if name in named_params:
                named_params[name].copy_(
                    ema_tensor.to(
                        device=named_params[name].device,
                        dtype=named_params[name].dtype,
                    )
                )
                copied += 1
            else:
                missing.append(name)

    print(f"[EMA] copied shadow params: {copied}")

    if missing:
        print(f"[EMA Warning] missing params: {len(missing)}")

    return copied > 0


def load_checkpoint(model, classifier, checkpoint_path: str, device: str, use_ema: bool = True):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    print(f"[Checkpoint] Loading: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    if "model_state" not in state or "classifier_state" not in state:
        raise KeyError("checkpoint 中没有 model_state / classifier_state，请确认这是训练脚本保存的权重。")

    model.load_state_dict(state["model_state"], strict=False)
    classifier.load_state_dict(state["classifier_state"], strict=True)

    used_ema = False

    if use_ema and "ema_state" in state:
        used_ema = copy_ema_shadow_to_model(
            model=model,
            classifier=classifier,
            ema_state=state["ema_state"],
        )

    print(f"[Checkpoint] epoch: {state.get('epoch', 'N/A')}")
    print(f"[Checkpoint] best_ema_acc: {state.get('best_ema_acc', 'N/A')}")
    print(f"[Checkpoint] best_epoch: {state.get('best_epoch', 'N/A')}")
    print(f"[Checkpoint] best_metric_value: {state.get('best_metric_value', 'N/A')}")

    best_info = state.get("best_metrics_info", None)
    if isinstance(best_info, dict):
        print(f"[Checkpoint] best_metrics_info.ema_acc: {best_info.get('ema_acc', 'N/A')}")
        print(f"[Checkpoint] best_metrics_info.ema_tta_acc: {best_info.get('ema_tta_acc', 'N/A')}")
        print(f"[Checkpoint] best_metrics_info.val_acc: {best_info.get('val_acc', 'N/A')}")

    print(f"[Checkpoint] infer weights: {'EMA shadow' if used_ema else 'raw model_state'}")

    return state


# ============================================================================
# Inference
# ============================================================================
@torch.no_grad()
def infer_one_batch(model, classifier, pil_images, sizes, hflip, device, use_amp):
    logits_sum = None
    n_view = 0

    for size in sizes:
        transform = get_val_transform(size)
        images = torch.stack([transform(img.convert("RGB")) for img in pil_images], dim=0)
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            features = model.encode_image(images)
            logits = classifier(features)

        logits_sum = logits if logits_sum is None else logits_sum + logits
        n_view += 1

        if hflip:
            images_flip = torch.flip(images, dims=[3])

            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                features_flip = model.encode_image(images_flip)
                logits_flip = classifier(features_flip)

            logits_sum = logits_sum + logits_flip
            n_view += 1

    logits_avg = logits_sum / max(1, n_view)
    preds = logits_avg.argmax(dim=1)

    return preds


# ============================================================================
# Main
# ============================================================================
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, default="/home/cyp/speedsci/A4/test")
    parser.add_argument("--checkpoint", type=str, default="/home/cyp/speedsci/A4/checkpoint_v18_dynamic_robust/checkpoint_best_ema_tta.pt")
    parser.add_argument("--output", type=str, default="/home/cyp/speedsci/A4/checkpoint_v18_dynamic_robust/pred_v18_dynamic_robust.csv")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # 选择推理尺寸：
    # 默认使用 cfg.tta_sizes，例如 [256, 448]
    # 单尺度 324：--no_tta --sizes 324
    # 指定多尺度：--sizes 324,448
    parser.add_argument("--sizes", type=str, default=None, help="例如: 324 或 256,448")
    parser.add_argument("--no_tta", action="store_true", help="关闭 TTA，默认用 cfg.tta_sizes + hflip")
    parser.add_argument("--no_hflip", action="store_true")

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_ema", action="store_true", help="即使 checkpoint 有 ema_state，也不用 EMA shadow")
    parser.add_argument("--recursive", action="store_true", help="测试图片如果在子文件夹里，开启这个")
    parser.add_argument("--with_header", action="store_true", help="输出 CSV 表头 filename,label；默认不加表头")

    # 如果你临时想改 LoRA 后几层数量，必须和训练 checkpoint 一致
    parser.add_argument("--lora_last_n_layers", type=int, default=None)

    args = parser.parse_args()

    cfg = Config()
    cfg.test_dir = Path(args.test_dir)

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    if args.no_amp:
        cfg.use_amp = False

    if args.lora_last_n_layers is not None:
        cfg.lora_last_n_layers = args.lora_last_n_layers

    # 兼容旧 config：如果没有 tta_sizes，就默认 [256, 448]
    if not hasattr(cfg, "tta_sizes"):
        cfg.tta_sizes = [256, 448]

    if not hasattr(cfg, "tta_hflip"):
        cfg.tta_hflip = True

    if args.sizes is not None:
        infer_sizes = parse_sizes(args.sizes)
    elif args.no_tta:
        infer_sizes = [cfg.image_size]
    else:
        infer_sizes = list(cfg.tta_sizes)

    hflip = (not args.no_hflip) and (not args.no_tta) and bool(cfg.tta_hflip)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {device}")

    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    print(f"[Config] test_dir: {cfg.test_dir}")
    print(f"[Config] checkpoint: {args.checkpoint}")
    print(f"[Config] output: {args.output}")
    print(f"[Config] num_classes: {cfg.num_classes}")
    print(f"[Config] image_size: {cfg.image_size}")
    print(f"[Config] lora_target: {cfg.lora_target}")
    print(f"[Config] lora_last_n_layers: {getattr(cfg, 'lora_last_n_layers', 'N/A')}")
    print(f"[Config] infer_sizes: {infer_sizes}, hflip={hflip}, amp={cfg.use_amp}")
    print(f"[Config] use_ema: {not args.no_ema}")

    model, classifier = build_clip_lora(cfg, device=device)
    model = model.to(device)
    classifier = classifier.to(device)

    load_checkpoint(
        model=model,
        classifier=classifier,
        checkpoint_path=args.checkpoint,
        device=device,
        use_ema=(not args.no_ema),
    )

    model.eval()
    classifier.eval()

    id_to_class = build_id_to_class(cfg)

    test_dataset = TestDataset(cfg.test_dir, recursive=args.recursive)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=pil_collate_fn,
    )

    print(f"[Data] Test samples: {len(test_dataset)}")

    all_names = []
    all_labels = []

    for pil_images, names in tqdm(test_loader, desc="[Inference]"):
        preds = infer_one_batch(
            model=model,
            classifier=classifier,
            pil_images=pil_images,
            sizes=infer_sizes,
            hflip=hflip,
            device=device,
            use_amp=cfg.use_amp,
        )

        all_names.extend(names)
        all_labels.extend([id_to_class[int(p)] for p in preds.cpu().tolist()])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if args.with_header:
            f.write("filename,label\n")

        for name, label in zip(all_names, all_labels):
            # 注意：提交格式中间不要有空格
            f.write(f"{name},{label}\n")

    print(f"\n[Done] Saved predictions to: {output_path}")
    print(f"[Done] Total: {len(all_labels)}")
    print("[Preview]")

    for name, label in list(zip(all_names, all_labels))[:5]:
        print(f"{name},{label}")


if __name__ == "__main__":
    main()
