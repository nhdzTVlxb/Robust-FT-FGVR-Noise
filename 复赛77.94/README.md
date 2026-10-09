# Exp22（4090 / 750 类）

这是复赛 750 类数据集上的 Exp22 整理包。模型为 CLIP ViT-B/32 + 全 12 个 ViT block 的 LoRA，输入 512 px，LoRA rank/alpha 为 `128/256`，全量训练 50 epoch。

训练启用了 RandAugment、MixUp/CutMix、冻结 CLIP 特征蒸馏、EMA、small-loss 和 EMA soft-label。详细参数和流程见 `配置.txt`。

权重采用紧凑推理格式，只保存 LoRA 与分类头的 EMA 参数；预测时需要相同的赛题允许 CLIP 基座：

```bash
python predict_balanced.py \
  --checkpoint ../weights/best_lora_inference.pth \
  --clip-model /path/to/clip-vit-base-patch32 \
  --test-dir /path/to/test \
  --tta legacy14 \
  --variants raw range30_70
```

完整训练运行参数和元数据在 `metadata/`，代码在 `code/`。
