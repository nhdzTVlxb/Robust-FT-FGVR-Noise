"""Weight-average (SWA / model soup) several same-architecture checkpoints into
ONE model. The output is a single set of weights -> single forward pass at
inference, i.e. still a single model (competition-compliant; this is a training
technique, not an output-level ensemble/vote/fusion).

Only valid for checkpoints with identical architecture (same lora-rank/target/
img-size). Backbone (frozen) weights are identical across ckpts; LoRA + head
weights get averaged.

Usage:
  python tools/swa_soup.py --checkpoints a.pt b.pt c.pt --out work/lora/full.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # stream-accumulate one checkpoint at a time (RAM-safe for many large ckpts).
    base = torch.load(args.checkpoints[0], map_location="cpu")
    keys = list(base["model"].keys())
    acc = {k: base["model"][k].float().clone() if base["model"][k].is_floating_point() else base["model"][k]
           for k in keys}
    n = 1
    for c in args.checkpoints[1:]:
        s = torch.load(c, map_location="cpu")
        assert set(s["model"].keys()) == set(keys), "arch mismatch across checkpoints"
        for k in keys:
            if acc[k].is_floating_point():
                acc[k] += s["model"][k].float()
        n += 1
        del s
    avg = {k: (acc[k] / n).to(base["model"][k].dtype) if acc[k].is_floating_point() else acc[k]
           for k in keys}
    base["model"] = avg
    base["epoch"] = "swa"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(f"souped {n} checkpoints -> {out}")
    print("  sources:", ", ".join(Path(c).name for c in args.checkpoints))


if __name__ == "__main__":
    main()
