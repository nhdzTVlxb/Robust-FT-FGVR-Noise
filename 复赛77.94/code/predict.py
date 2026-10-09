from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

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
    """可被 DataLoader worker pickle 的历史裁剪视图。"""

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
        T.Compose([T.Resize(larger), CropView(size, top, left, flip), T.Resize(size)] + normalize)
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
    parser.add_argument("--output", type=str, default="submission_v19")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tta", choices=["legacy14", "center2"], default="legacy14")
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
        # Exp21 stores EMA only for trainable LoRA/head parameters; frozen
        # CLIP weights are already identical to the raw checkpoint.
        vision.load_state_dict(state["vision_state_dict"], strict=True)
        head.load_state_dict(state["head_state_dict"], strict=True)
        combined.load_state_dict(state["ema_state_dict"], strict=False)
    else:
        vision.load_state_dict(state["vision_state_dict"], strict=True)
        head.load_state_dict(state["head_state_dict"], strict=True)

    vision = vision.to(device).eval()
    head = head.to(device).eval()
    return state, config, vision, head, actual_size


@torch.inference_mode()
def predict(args):
    cfg = Config()
    if args.test_dir is None:
        test_dir = cfg.test_dir
    else:
        test_dir = Path(args.test_dir)
    clip_model = Path(args.clip_model) if args.clip_model else cfg.clip_model

    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pred_results.csv"
    zip_path = output_dir / "submission.zip"
    if csv_path.exists() or zip_path.exists():
        raise FileExistsError(f"output already exists in {output_dir}; use a new directory")

    inspect_model(str(clip_model))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    state, config, vision, head, actual_size = load_checkpoint(
        checkpoint, clip_model, str(device)
    )
    files = sorted(
        path
        for path in Path(test_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not files:
        raise RuntimeError(f"No test images found in {test_dir}")
    if any("," in path.name or "\n" in path.name for path in files):
        raise ValueError("Image filenames cannot contain commas or newlines")

    views = build_views(actual_size, args.tta)
    dataset = TTADataset(files, views)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    totals = torch.zeros(len(files), config["num_classes"], dtype=torch.float32)
    counts = torch.zeros(len(files), dtype=torch.long)
    amp_on = device.type == "cuda" and config.get("amp", "float16") != "float32"
    amp_dtype = getattr(torch, config.get("amp", "float16"))

    for step, (images, indices) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp_on, dtype=amp_dtype):
            scores = clip_forward(vision, head, images)
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite prediction scores")
        totals.index_add_(0, indices, scores.float().cpu())
        counts.index_add_(0, indices, torch.ones_like(indices))
        if step % 100 == 0:
            print(f"[Inference] batch {step + 1}/{len(loader)}", flush=True)

    if not torch.all(counts == len(views)):
        raise RuntimeError("Missing or duplicated TTA views")

    predictions = (totals / len(views)).argmax(dim=1).tolist()
    submission_ids = config["submission_ids"]
    lines = []
    for path, prediction in zip(files, predictions):
        separator = "," if not args.with_header else ","
        # 竞赛规范使用逗号后无空格；保留原包的字段含义但不引入空格。
        lines.append(f"{path.name}{separator}{submission_ids[prediction]}\n")

    with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
        if args.with_header:
            handle.write("filename,label\n")
        handle.writelines(lines)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, "pred_results.csv")

    report = {
        "images": len(files),
        "rows": len(predictions),
        "classes": len(submission_ids),
        "tta": args.tta,
        "views": len(views),
        "checkpoint_epoch": state["epoch"],
        "header": bool(args.with_header),
        "weights_kind": "EMA",
        "csv": str(csv_path),
        "zip": str(zip_path),
    }
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    print(f"[Done] saved {csv_path} and {zip_path}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(4)
    predict(args)


if __name__ == "__main__":
    main()
