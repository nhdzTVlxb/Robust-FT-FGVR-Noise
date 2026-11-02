from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .images import ClipPixelNormalizer, EvaluationPhotoDataset, quiet_worker

def photo_loader(dataset: Dataset, batch: int, workers: int) -> DataLoader:
    extra = {'prefetch_factor': 4} if workers else {}
    return DataLoader(dataset, batch_size=batch, num_workers=workers, pin_memory=True, worker_init_fn=quiet_worker, **extra)

@torch.no_grad()
def validate_sides(net: torch.nn.Module, paths: Sequence[Path], side: int, batch: int, workers: int, long_cap: int=0) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    norm = ClipPixelNormalizer(next(net.parameters()).device)
    probs = np.empty((len(paths), net.readout.out_features), np.float16)
    emb = np.empty((len(paths), net.readout.in_features), np.float16)
    at = 0
    for block in photo_loader(EvaluationPhotoDataset(paths, side, long_cap), batch, workers):
        got = len(block)
        if got < batch:
            block = torch.cat([block, block[-1:].expand(batch - got, -1, -1, -1)])
        x = norm(block)
        with torch.autocast('cuda', torch.bfloat16):
            (logits_a, emb_a) = net(x)
            (logits_b, emb_b) = net(x.flip(3))
        p = (logits_a.float().softmax(1) + logits_b.float().softmax(1)) / 2
        e = F.normalize(F.normalize(emb_a.float(), dim=1) + F.normalize(emb_b.float(), dim=1), dim=1)
        probs[at:at + got] = p[:got].cpu().numpy()
        emb[at:at + got] = e[:got].cpu().numpy()
        at += got
    assert at == len(paths), 'loader returned %d of %d pictures' % (at, len(paths))
    return (probs, emb)
