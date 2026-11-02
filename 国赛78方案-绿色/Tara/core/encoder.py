from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
TIMM_ID = 'vit_base_patch32_clip_quickgelu_224.openai'
PATCH = 32

class TaraClassifier(nn.Module):

    def __init__(self, n_classes: int, side: int, clip_file: Optional[Path]=None) -> None:
        super().__init__()
        import timm
        extra = {'pretrained_cfg_overlay': {'file': str(clip_file)}} if clip_file else {}
        self.tower = timm.create_model(TIMM_ID, pretrained=clip_file is not None, img_size=side, weight_init='skip', **extra)
        self.tower.patch_embed.strict_img_size = False
        self.readout = nn.Linear(self.tower.num_classes, n_classes)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer('class_codes', torch.arange(n_classes, dtype=torch.int64))
        self.register_buffer('init_temperature', torch.zeros(()))
        self.grid = side // PATCH

    def position_table(self, g: int) -> torch.Tensor:
        (table, grid) = (self.tower.pos_embed, self.grid)
        body = F.interpolate(table[:, 1:].unflatten(1, (grid, grid)).movedim(-1, 1).float(), size=(g, g), mode='bicubic', align_corners=False, antialias=g < grid) if g != grid else None
        return table if body is None else torch.cat([table[:, :1], body.to(table.dtype).movedim(1, -1).flatten(1, 2)], 1)

    def embed(self, x: torch.Tensor, swaps: Optional[torch.Tensor]=None) -> torch.Tensor:
        t = self.tower
        h = t.patch_embed(x)
        h = torch.cat([t.cls_token.expand(len(h), -1, -1), h], 1) + self.position_table(x.shape[-1] // PATCH)
        h = t.norm_pre(h)
        for (depth, block) in enumerate(t.blocks):
            h = block(h)
            if swaps is not None:
                if swaps.ndim == 5 and depth < swaps.shape[1]:
                    mask = swaps[:, depth, 0]
                    h = torch.where(mask, h.flip(0), h)
                elif swaps.ndim == 4 and depth < len(swaps):
                    h = torch.where(swaps[depth], h.flip(0), h)
        return t.head(t.norm(h)[:, 0])

    def forward(self, x: torch.Tensor, swaps: Optional[torch.Tensor]=None):
        e = self.embed(x, swaps)
        return (self.readout(e), e)

def restore(weights_file: Path) -> TaraClassifier:
    state = torch.load(weights_file, map_location='cpu', weights_only=True)
    net = TaraClassifier(len(state['class_codes']), round((state['tower.pos_embed'].shape[1] - 1) ** 0.5) * PATCH)
    result = net.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {'init_temperature'} or result.unexpected_keys:
        raise RuntimeError('checkpoint does not match the Tara inference model')
    return net
