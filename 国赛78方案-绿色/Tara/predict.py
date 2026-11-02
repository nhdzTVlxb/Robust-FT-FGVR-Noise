import os
from pathlib import Path

import numpy as np
import torch

from tool import restore, image_files, infer_labels, submission_zip


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for inference')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    root = Path(__file__).resolve().parent
    default_weights = root / 'best.pth'
    if not default_weights.is_file():
        default_weights = root / 'weights' / 'best.pth'
    weights = Path(os.environ.get('TARA_WEIGHTS', default_weights))
    train_tree = Path(os.environ.get('TARA_TRAIN_TREE', '/mnt/aic/train'))
    test_tree = Path(os.environ.get('TARA_TEST_TREE', '/mnt/aic/test'))
    class_names = sorted(path.name for path in train_tree.iterdir() if path.is_dir())
    if not class_names:
        raise FileNotFoundError('no class folders found in %s' % train_tree)
    paths = image_files(test_tree)
    if not paths:
        raise FileNotFoundError('no test images found in %s' % test_tree)
    model = restore(weights).to(device).eval().requires_grad_(False)
    codes = model.class_codes.detach().cpu().tolist()
    if [int(name) for name in class_names] != codes:
        raise RuntimeError('class folder order does not match best.pth')
    labels = infer_labels(model, paths, device, batch_size=704, workers=8)
    destination = Path.cwd() / 'submission.zip'
    submission_zip(destination, paths, labels, class_names)
    counts = np.bincount(labels, minlength=len(class_names))
    print('Tara inference complete | images=%d classes=%d count=%d..%d | output=%s' %
          (len(paths), len(class_names), int(counts.min()), int(counts.max()), destination))


if __name__ == '__main__':
    main()
