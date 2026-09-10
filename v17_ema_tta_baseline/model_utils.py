import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


# ============================================================================
# CLIP ViT positional embedding interpolation
# ============================================================================
def patch_clip_visual_forward_for_any_size(visual: nn.Module):
    """
    让 OpenAI CLIP ViT-B/32 支持 324 / 448 这类非 224 输入。
    否则 positional_embedding 会因为 patch 数不同报错。
    """

    def interpolate_positional_embedding(self, x, new_h, new_w):
        pos = self.positional_embedding.to(dtype=x.dtype, device=x.device)

        cls_pos = pos[:1]
        patch_pos = pos[1:]

        old_grid = int(math.sqrt(patch_pos.shape[0]))
        if old_grid * old_grid != patch_pos.shape[0]:
            raise ValueError(f"Invalid positional embedding shape: {patch_pos.shape}")

        if old_grid == new_h and old_grid == new_w:
            return pos

        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_h * new_w, -1)

        return torch.cat([cls_pos, patch_pos], dim=0)

    def forward_any_size(self, x: torch.Tensor):
        x = self.conv1(x)  # [B, C, H, W]
        new_h, new_w = x.shape[-2], x.shape[-1]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # [B, HW, C]

        cls_token = self.class_embedding.to(dtype=x.dtype, device=x.device)
        cls_token = cls_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)

        pos = interpolate_positional_embedding(self, x, new_h, new_w)
        x = x + pos.unsqueeze(0)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [L, B, C]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, C]

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    visual.forward = types.MethodType(forward_any_size, visual)


# ============================================================================
# LoRA
# ============================================================================
class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original

        for p in self.original.parameters():
            p.requires_grad_(False)

        self.lora = LoRA(original.in_features, original.out_features, rank, alpha)

    def forward(self, x: torch.Tensor):
        return self.original(x) + self.lora(x)


def replace_linear_with_lora(module, rank=8, alpha=16, target_names=None):
    target_names = target_names or set()

    for name, child in module.named_children():
        if name in target_names and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            replace_linear_with_lora(child, rank, alpha, target_names)


class LoRAAttention(nn.Module):
    def __init__(self, orig_attn, d_model: int, n_head: int, lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        in_w = orig_attn.in_proj_weight.data
        in_b = orig_attn.in_proj_bias.data if orig_attn.in_proj_bias is not None else None

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = orig_attn.out_proj

        with torch.no_grad():
            self.q_proj.weight.copy_(in_w[:d_model])
            self.k_proj.weight.copy_(in_w[d_model:2 * d_model])
            self.v_proj.weight.copy_(in_w[2 * d_model:])

            if in_b is not None:
                self.q_proj.bias.copy_(in_b[:d_model])
                self.k_proj.bias.copy_(in_b[d_model:2 * d_model])
                self.v_proj.bias.copy_(in_b[2 * d_model:])

        for p in self.q_proj.parameters():
            p.requires_grad_(False)
        for p in self.k_proj.parameters():
            p.requires_grad_(False)
        for p in self.v_proj.parameters():
            p.requires_grad_(False)
        for p in self.out_proj.parameters():
            p.requires_grad_(False)

        self.lora_q = LoRA(d_model, d_model, lora_rank, lora_alpha)
        self.lora_v = LoRA(d_model, d_model, lora_rank, lora_alpha)
        self.lora_out = LoRA(d_model, d_model, lora_rank, lora_alpha)

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None, **kwargs):
        x = query
        s, b, c = x.shape

        q = self.q_proj(x) + self.lora_q(x)
        k = self.k_proj(x)
        v = self.v_proj(x) + self.lora_v(x)

        q = q.view(s, b, self.n_head, self.head_dim).permute(1, 2, 0, 3).contiguous()
        k = k.view(s, b, self.n_head, self.head_dim).permute(1, 2, 0, 3).contiguous()
        v = v.view(s, b, self.n_head, self.head_dim).permute(1, 2, 0, 3).contiguous()

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        out = out.permute(2, 0, 1, 3).contiguous().view(s, b, c)
        out = self.out_proj(out) + self.lora_out(out)

        return out, None


# ============================================================================
# Classifier heads
# ============================================================================
class MLPClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ============================================================================
# Build model
# ============================================================================
def build_clip_lora(cfg, device="cpu"):
    if not cfg.pretrained_path.exists():
        raise FileNotFoundError(f"CLIP 权重不存在: {cfg.pretrained_path}")

    model, _ = clip.load(str(cfg.pretrained_path), device=device, jit=False)
    model = model.float()

    patch_clip_visual_forward_for_any_size(model.visual)

    for p in model.parameters():
        p.requires_grad_(False)

    visual = model.visual
    d_model = visual.transformer.width
    n_head = d_model // 64

    resblocks = visual.transformer.resblocks
    total_layers = len(resblocks)

    # 默认只给最后 6 层加 LoRA
    lora_last_n_layers = getattr(cfg, "lora_last_n_layers", 6)
    lora_start_layer = max(0, total_layers - lora_last_n_layers)

    lora_layers = []

    for block_idx, block in enumerate(resblocks):
        if block_idx < lora_start_layer:
            continue

        lora_layers.append(block_idx)

        if "attn" in cfg.lora_target:
            block.attn = LoRAAttention(
                orig_attn=block.attn,
                d_model=d_model,
                n_head=n_head,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
            )

        if "mlp" in cfg.lora_target:
            replace_linear_with_lora(
                block.mlp,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                target_names={"c_fc", "c_proj"},
            )

    feature_dim = 512
    classifier_head = getattr(cfg, "classifier_head", "linear")

    if classifier_head == "mlp":
        classifier = MLPClassifier(
            feature_dim=feature_dim,
            hidden_dim=getattr(cfg, "mlp_hidden_dim", 1024),
            num_classes=cfg.num_classes,
            dropout=getattr(cfg, "mlp_dropout", 0.2),
        )
    else:
        classifier = nn.Linear(feature_dim, cfg.num_classes)
        nn.init.normal_(classifier.weight, std=0.01)
        nn.init.zeros_(classifier.bias)

    trainable_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_cls = sum(p.numel() for p in classifier.parameters())

    print(f"[Model] CLIP weight: {cfg.pretrained_path}")
    print(f"[Model] image_size: {cfg.image_size}")
    print(f"[Model] LoRA target: {cfg.lora_target}")
    print(f"[Model] LoRA rank: {cfg.lora_rank}, alpha: {cfg.lora_alpha}")
    print(f"[Model] LoRA last N layers: {lora_last_n_layers}")
    print(f"[Model] LoRA layer indices: {lora_layers}")
    print(f"[Model] Classifier head: {classifier_head}")
    print(f"[Model] Trainable LoRA params: {trainable_lora:,}")
    print(f"[Model] Classifier params: {trainable_cls:,}")

    return model, classifier


def build_original_clip(cfg, device="cpu"):
    model, _ = clip.load(str(cfg.pretrained_path), device=device, jit=False)
    model = model.float().eval()
    patch_clip_visual_forward_for_any_size(model.visual)

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================================
# EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model: nn.Module, classifier: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, p in self.named_trainable_params(model, classifier):
            self.shadow[name] = p.detach().clone()

    def named_trainable_params(self, model, classifier):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield "model." + name, p

        for name, p in classifier.named_parameters():
            yield "classifier." + name, p

    @torch.no_grad()
    def update(self, model, classifier):
        for name, p in self.named_trainable_params(model, classifier):
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model, classifier):
        self.backup = {}

        for name, p in self.named_trainable_params(model, classifier):
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model, classifier):
        for name, p in self.named_trainable_params(model, classifier):
            p.data.copy_(self.backup[name].data)

        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ============================================================================
# Loss
# ============================================================================
class LabelSmoothingCELoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


class FeatureConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, ft_features, orig_features):
        ft_norm = F.normalize(ft_features, dim=1)
        orig_norm = F.normalize(orig_features, dim=1)
        return self.weight * F.mse_loss(ft_norm, orig_norm)
