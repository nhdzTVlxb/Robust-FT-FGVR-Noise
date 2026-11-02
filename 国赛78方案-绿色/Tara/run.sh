#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TARA_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
"$PYTHON_BIN" "$ROOT/main.py" train \
  --tree "${TARA_TRAIN_TREE:-/mnt/aic/train}" \
  --clip "${TARA_CLIP:-/mnt/aic/open_clip_model.pth}" \
  --output "${TARA_OUTPUT_DIR:-$ROOT}" \
  --workers "${TARA_WORKERS:-12}" \
  --batch-size "${TARA_BATCH_SIZE:-192}" \
  --epochs "${TARA_EPOCHS:-40}" >> "$ROOT/logs/train.log" 2>&1
