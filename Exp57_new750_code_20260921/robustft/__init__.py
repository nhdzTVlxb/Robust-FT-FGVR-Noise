"""Robust fine-tuning of CLIP ViT-B/32 under noisy labels.

Core library for the competition project. Pure library code: no argparse,
no side effects beyond `PIL.ImageFile.LOAD_TRUNCATED_IMAGES` (set in
`robustft.data`, required for the competition images).
"""

__version__ = "1.0.0"
