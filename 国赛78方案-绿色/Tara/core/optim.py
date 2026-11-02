import collections
import math
from typing import Callable, Dict, List, Optional
import torch
import torch.nn.functional as F
class NoiseAwareObjective:

    def __init__(self, n_classes: int, label_smoothing: float, gce_fraction: float, gce_power: float) -> None:
        self.n = n_classes
        self.label_smoothing = label_smoothing
        self.gce_fraction = gce_fraction
        self.gce_power = gce_power

    def smoothed(self, y: torch.Tensor) -> torch.Tensor:
        rows = torch.full((len(y), self.n), self.label_smoothing / self.n, device=y.device)
        return rows.scatter_(1, y[:, None], 1.0 - self.label_smoothing + self.label_smoothing / self.n)

    def __call__(self, logits: torch.Tensor, y: torch.Tensor, lam: torch.Tensor, weights=None) -> torch.Tensor:
        own = self.smoothed(y)
        logp = F.log_softmax(logits.float(), 1)
        own_ce = -(own * logp).sum(1)
        other_ce = -(own.flip(0) * logp).sum(1)
        own_gce = (1.0 - logp.gather(1, y[:, None]).squeeze(1).mul(self.gce_power).exp()) / self.gce_power
        other_y = y.flip(0)
        other_gce = (1.0 - logp.gather(1, other_y[:, None]).squeeze(1).mul(self.gce_power).exp()) / self.gce_power
        own_loss = (1.0 - self.gce_fraction) * own_ce + self.gce_fraction * own_gce
        other_loss = (1.0 - self.gce_fraction) * other_ce + self.gce_fraction * other_gce
        if weights is None:
            return (lam * own_loss + (1.0 - lam) * other_loss).mean()
        (own_w, other_w) = (weights, weights.flip(0))
        total = lam * own_w + (1.0 - lam) * other_w
        return (lam * own_w * own_loss + (1.0 - lam) * other_w * other_loss).sum() / total.sum().clamp_min(1e-08)

def depth_rank(name: str, n_blocks: int) -> Optional[int]:
    if name.startswith('readout.'):
        return None
    if name.startswith('tower.blocks.'):
        return int(name.split('.')[2]) + 1
    return n_blocks + 1 if name.startswith(('tower.norm.', 'tower.head.')) else 0

def layerwise_groups(net: torch.nn.Module, opt) -> List[dict]:
    n_blocks = len(net.tower.blocks)
    groups: 'collections.OrderedDict[tuple, dict]' = collections.OrderedDict()
    for (name, p) in net.named_parameters():
        rank = depth_rank(name, n_blocks)
        decays = p.ndim >= 2 and (not name.endswith(('pos_embed', 'cls_token')))
        rate = opt.head_lr if rank is None else opt.tower_lr * opt.layer_decay ** (n_blocks + 1 - rank)
        g = groups.setdefault((rank, decays), {'params': [], 'lr': rate, 'weight_decay': opt.weight_decay if decays else 0.0})
        g['params'].append(p)
    return list(groups.values())

def warm_cosine(total: int, warm: int, floor: float=0.01) -> Callable[[int], float]:

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return factor

