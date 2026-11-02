import enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
Box = Tuple[int, int, int, int]

class InteractionKind(enum.IntEnum):
    CLEAN = 0
    SWAP = 1
    MIXUP = 2
    CUTMIX = 3

def centred_span(rng: np.random.Generator, extent: int, edge: int) -> Tuple[int, int]:
    c = int(rng.integers(0, extent))
    return (min(max(c - edge // 2, 0), extent), min(max(c + edge - edge // 2, 0), extent))

def edge_for(kept: float, extent: int) -> int:
    return int(round(extent * math.sqrt(max(0.0, 1.0 - kept))))

class InteractionAgenda:

    def __init__(self, cfg, seed: int, epoch: int, train_indices: np.ndarray, steps: int, grid: int, side: int) -> None:
        swap, blend = cfg.token_swap, cfg.image_blend
        self.order = np.random.default_rng((seed, epoch, 5)).permutation(train_indices)
        depth_rng = np.random.default_rng((seed, epoch, 7))
        depth = depth_rng.integers(1, swap.candidate_blocks + 1, steps)
        kept = depth_rng.beta(swap.beta, swap.beta, steps)
        self.kind, self.lam, self.box = self._image_plan(
            np.random.default_rng((seed, epoch, 9)), steps, swap.step_share, kept, blend, side
        )
        depth[self.kind != InteractionKind.SWAP] = 0
        self.swap = self._token_plan(
            np.random.default_rng((seed, epoch, 13)), depth, grid, swap.candidate_blocks, swap.aspect_jitter
        )

    def _image_plan(self, rng: np.random.Generator, steps: int, share: float, kept: np.ndarray, blend, side: int):
        kind = np.full(steps, InteractionKind.CLEAN, np.int64)
        lam = np.ones(steps)
        box: List[Optional[Box]] = [None] * steps
        swapped = rng.random(steps) < share
        kind[swapped], lam[swapped] = InteractionKind.SWAP, kept[swapped]
        for s in np.flatnonzero(~swapped):
            if rng.random() >= blend.rate:
                continue
            if rng.random() < blend.mixup_share:
                kind[s], lam[s] = InteractionKind.MIXUP, float(rng.beta(blend.mixup_beta, blend.mixup_beta))
                continue
            edge = edge_for(float(rng.beta(blend.cutmix_beta, blend.cutmix_beta)), side)
            (r0, r1) = centred_span(rng, side, edge)
            (c0, c1) = centred_span(rng, side, edge)
            kind[s], lam[s], box[s] = InteractionKind.CUTMIX, 1.0 - (r1 - r0) * (c1 - c0) / float(side * side), (r0, r1, c0, c1)
        return (kind, lam, box)

    def _token_plan(self, rng: np.random.Generator, depth: np.ndarray, grid: int, blocks: int, aspect_jitter: float) -> torch.Tensor:
        table = np.zeros((len(depth), blocks, 1, 1 + grid * grid, 1), bool)
        for s in np.flatnonzero(depth):
            area = max(1.0, grid * grid * (1.0 - self.lam[s]))
            ratio = math.exp(rng.uniform(-aspect_jitter, aspect_jitter))
            rows = min(grid, max(1, int(round(math.sqrt(area / ratio)))))
            cols = min(grid, max(1, int(round(math.sqrt(area * ratio)))))
            r0, r1 = centred_span(rng, grid, rows)
            c0, c1 = centred_span(rng, grid, cols)
            cells = np.zeros((grid, grid), bool)
            cells[r0:r1, c0:c1] = True
            table[s, depth[s] - 1, 0, 1:, 0] = cells.ravel()
            self.lam[s] = 1.0 - cells.sum() / float(grid * grid)
        return torch.from_numpy(table)

    def blend_pixels(self, x: torch.Tensor, s: int) -> torch.Tensor:
        if self.kind[s] == InteractionKind.MIXUP:
            return torch.lerp(x.flip(0), x, float(self.lam[s]))
        if self.kind[s] == InteractionKind.CUTMIX:
            (r0, r1, c0, c1) = self.box[s]
            x[..., r0:r1, c0:c1] = x.flip(0)[..., r0:r1, c0:c1]
        return x

    def tally(self) -> Dict[str, int]:
        return {k.name.lower(): int((self.kind == k).sum()) for k in InteractionKind}
