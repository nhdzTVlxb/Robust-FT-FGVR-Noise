import argparse
import os
import sys
from pathlib import Path

from core import engine, settings


def train_command(arguments):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, default=Path(os.environ.get('TARA_TRAIN_TREE', '/mnt/aic/train')))
    parser.add_argument('--clip', type=Path, default=Path(os.environ.get('TARA_CLIP', '/mnt/aic/open_clip_model.pth')))
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('TARA_OUTPUT_DIR', root)))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    values = parser.parse_args(arguments)
    cfg = settings.load({'optim': {'batch_size': values.batch_size, 'epochs': values.epochs}})
    engine.train(values.tree, values.clip, values.output, cfg, workers=values.workers, report=print)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'train':
        raise SystemExit('usage: python main.py train [options]')
    train_command(sys.argv[2:])


if __name__ == '__main__':
    main()
