from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
GRID = np.geomspace(1.0, 300.0, 41)

def class_offsets(feats: torch.Tensor, labels: torch.Tensor, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    sums = feats.new_zeros(n_classes, feats.shape[1]).index_add_(0, labels, feats)
    counts = torch.bincount(labels, minlength=n_classes).to(feats.dtype).clamp_min(1).unsqueeze(1)
    means = sums / counts
    return (F.normalize(means - feats.mean(0), dim=1), feats.mean(0))

@torch.no_grad()
def initialise_readout(feats: torch.Tensor, labels: torch.Tensor, n_classes: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    radius = feats.norm(dim=1).mean()
    half = torch.as_tensor(np.random.default_rng((seed, 23)).random(len(labels)) < 0.5, device=feats.device)
    grid = torch.as_tensor(GRID, dtype=torch.float32, device=feats.device)
    held_ce = torch.zeros(len(grid), device=feats.device)
    usable = 0
    for fold in (half, ~half):
        if fold.sum() == 0 or (~fold).sum() == 0:
            continue
        (d, centre) = class_offsets(feats[fold], labels[fold], n_classes)
        z = (feats[~fold] - centre) / radius @ d.T
        scaled = (grid[:, None, None] * z).flatten(0, 1)
        held_ce += F.cross_entropy(scaled, labels[~fold].repeat(len(grid)), reduction='none').view(len(grid), -1).mean(1)
        usable += 1
    temperature = grid[held_ce.argmin()] if usable else grid[len(grid) // 2]
    (d, centre) = class_offsets(feats, labels, n_classes)
    weight = temperature / radius * d
    return (weight, -(weight @ centre), float(temperature))
