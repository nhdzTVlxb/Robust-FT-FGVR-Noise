# Exp57（750 类复赛数据）最小工程

这是当前服务器上 Exp57 训练流程的精简快照，只保留运行所需代码，不包含数据、特征缓存、权重、checkpoint、日志或历史实验结果。

## 当前实验配置

- 官方 CLIP ViT-B/32 backbone
- 750 类 ImageFolder 训练集
- 输入尺寸 `576`，位置编码 `aligned`
- batch size `96`；显存不足时启动脚本自动回退到 `80`
- 训练 `12` epoch，OneCycle 调度 `16` epoch
- 三段式 LoRA：
  - block 0–3：rank/alpha `32/64`
  - block 4–7：`96/192`
  - block 8–11：`192/384`
- Cleanlab + teacher-kNN 清洗
- RandAugment、MixUp `0.2`
- small-loss 从第 6 轮启用
- EMA `0.9995`
- SWA：`epoch 4–12`
- TTA：`512/576/640` + horizontal flip
- soft balance strength：`1.0`

## 文件说明

### 顶层文件

- `config.py`：项目默认路径、类别和提交文件路径等基础配置。
- `main.py`：冻结 CLIP 特征提取入口，生成训练集/测试集特征缓存；类别数从训练目录自动读取，因此支持 750 类。
- `finetune_lora.py`：主训练入口，执行 Cleanlab/kNN 标签处理、三段式 LoRA 微调、small-loss、EMA、保存每轮 checkpoint。
- `run_exp57_new750.sh`：服务器一键流程。先构建新数据特征缓存，再训练、生成 SWA、执行 TTA 预测。它还包含 batch 96 OOM 时自动退回 batch 80 的逻辑。
- `requirements.txt`：Python 依赖列表。

### `robustft/`

- `data.py`：ImageFolder 数据读取、训练增强和测试集读取。
- `denoise.py`：Cleanlab、kNN 共识、伪标签、样本筛选和权重计算。
- `engine.py`：训练循环、EMA/教师头、损失函数、优化和评估辅助函数。
- `features.py`：冻结 CLIP 特征提取、缓存读写和缓存一致性检查。
- `models.py`：CLIP ViT-B/32、LoRA 注入、三段式 rank 配置、分类头和位置编码重采样。
- `robust_utils.py`：checkpoint 选择、数据划分和通用鲁棒训练工具。
- `submission.py`：生成 CSV、校验类别编号、压缩提交 ZIP。
- `__init__.py`：Python 包标记文件。

### `tools/`

- `swa_soup.py`：对同一训练轨迹的 epoch 4–12 checkpoint 做参数平均，生成 `full_swa_ep04_12.pt`。
- `tta_predict.py`：加载 SWA 权重，执行 512/576/640 多尺度 + 翻转 TTA，并生成原始和 soft-balance 提交文件。

## 运行方式

服务器上需要准备：

```text
/my-data/aic4/train/<class_id>/*.jpg
/my-data/aic4/test/*.jpg
/my-data/aic4/models/clip-vit-base-patch32/
```

然后执行：

```bash
bash run_exp57_new750.sh
```

输出位置：

```text
/my-data/aic4/exp_shared_cache/
/my-data/aic4/checkpoint_exp57_new750_b96_12ep/
/my-data/aic4/submissions/pred_exp57_new750_swa_ep04_12_tta_balanced.zip
```

## 合规说明

本工程最终使用的 backbone 是官方 CLIP ViT-B/32；训练数据由比赛提供。工程快照不包含任何旧数据或训练权重。若赛事要求披露第三方代码来源，应按赛事规则补充来源声明。
