#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/my-data/aic4/pydeps_timm:/my-data/aic4/pydeps_cu128
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLIP_MODEL_DIR=/my-data/aic4/models/clip-vit-base-patch32

ROOT=/root/aic4/CLIP_76_TrainingKit_v18_style_exp37_38_from_exp33
CODE="$ROOT/code"
DATA=/my-data/aic4
CACHE="$DATA/exp_shared_cache"
BUILD="$DATA/exp57_new750_cache_build"
OUT="$DATA/checkpoint_exp57_new750_b96_12ep"
SUB="$DATA/submissions"
LOG="$DATA/exp57_new750.log"

mkdir -p "$CACHE" "$SUB"

if [ ! -f "$CACHE/train_features.pt" ] || [ ! -f "$CACHE/test_features.pt" ]; then
  echo "[cache] building fresh 750-class feature cache" | tee -a "$LOG"
  rm -rf "$BUILD"
  mkdir -p "$BUILD"
  (
    cd "$CODE"
    python3 -u main.py \
      --work-dir "$BUILD" \
      --train-dir "$DATA/train" \
      --test-dir "$DATA/test" \
      --device cuda \
      --num-workers 16 \
      --extract-batch-size 256 \
      --head-batch-size 8192 \
      --rebuild-cache
  ) 2>&1 | tee -a "$LOG"
  cp -f "$BUILD/cache/train_features.pt" "$CACHE/train_features.pt"
  cp -f "$BUILD/cache/test_features.pt" "$CACHE/test_features.pt"
fi

if [ ! -f "$OUT/training_complete" ]; then
  mkdir -p "$OUT"
  if [ ! -f "$OUT/lora/ep12.pt" ]; then
    set +e
    (
      cd "$CODE"
      python3 -u finetune_lora.py \
        --work-dir "$OUT" \
        --cache-dir "$CACHE" \
        --train-dir "$DATA/train" \
        --test-dir "$DATA/test" \
        --img-size 576 \
        --pos-resample aligned \
        --batch-size 96 \
        --epochs 12 \
        --schedule-epochs 16 \
        --num-workers 16 \
        --seed 42 \
        --full \
        --lora-rank 192 \
        --lora-alpha 384 \
        --lora-front-rank 32 \
        --lora-front-alpha 64 \
        --lora-split-block 4 \
        --lora-mid-rank 96 \
        --lora-mid-alpha 192 \
        --lora-second-split-block 8 \
        --lora-dropout 0.05 \
        --lora-target attn_mlp \
        --lora-blocks 12 \
        --keep-ratio 0.90 \
        --knn-k 16 \
        --denoise cleanlab \
        --teacher-epochs 20 \
        --pseudo-thresh 0.60 \
        --pseudo-margin 0.05 \
        --ema-decay 0.9995 \
        --randaug \
        --crop-min-scale 0.8 \
        --label-smoothing 0.1 \
        --mixup-alpha 0.2 \
        --lora-lr 1.30e-4 \
        --head-lr 1.25e-3 \
        --save-every 1 \
        --no-pin \
        --small-loss-lite \
        --small-loss-start-epoch 6 \
        --small-loss-keep-ratio 0.90 \
        --small-loss-noisy-weight 0.70 \
        --small-loss-ema-kl-weight 0.20 \
        --small-loss-temperature 1.0 \
        --small-loss-memory-momentum 0.70
    ) 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    set -e
    if [ "$status" -ne 0 ]; then
      if grep -qiE 'CUDA out of memory|OutOfMemoryError' "$LOG"; then
        echo "[train] batch96 OOM; retrying with batch80" | tee -a "$LOG"
        rm -rf "$OUT"
        mkdir -p "$OUT"
        (
          cd "$CODE"
          python3 -u finetune_lora.py \
            --work-dir "$OUT" --cache-dir "$CACHE" \
            --train-dir "$DATA/train" --test-dir "$DATA/test" \
            --img-size 576 --pos-resample aligned --batch-size 80 \
            --epochs 12 --schedule-epochs 16 --num-workers 16 --seed 42 --full \
            --lora-rank 192 --lora-alpha 384 --lora-front-rank 32 --lora-front-alpha 64 \
            --lora-split-block 4 --lora-mid-rank 96 --lora-mid-alpha 192 \
            --lora-second-split-block 8 --lora-dropout 0.05 --lora-target attn_mlp \
            --lora-blocks 12 --keep-ratio 0.90 --knn-k 16 --denoise cleanlab \
            --teacher-epochs 20 --pseudo-thresh 0.60 --pseudo-margin 0.05 \
            --ema-decay 0.9995 --randaug --crop-min-scale 0.8 --label-smoothing 0.1 \
            --mixup-alpha 0.2 --lora-lr 1.30e-4 --head-lr 1.25e-3 --save-every 1 \
            --no-pin --small-loss-lite --small-loss-start-epoch 6 \
            --small-loss-keep-ratio 0.90 --small-loss-noisy-weight 0.70 \
            --small-loss-ema-kl-weight 0.20 --small-loss-temperature 1.0 \
            --small-loss-memory-momentum 0.70
        ) 2>&1 | tee -a "$LOG"
      else
        exit "$status"
      fi
    fi
  fi

  checkpoints=()
  for epoch in $(seq 4 12); do
    checkpoints+=("$OUT/lora/ep$(printf '%02d' "$epoch").pt")
  done
  (
    cd "$CODE"
    python3 -u tools/swa_soup.py \
      --checkpoints "${checkpoints[@]}" \
      --out "$OUT/lora/full_swa_ep04_12.pt"
  ) 2>&1 | tee -a "$LOG"
  ln -f "$OUT/lora/full_swa_ep04_12.pt" "$OUT/lora/full.pt"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT/training_complete"
fi

PREFIX="$SUB/pred_exp57_new750_swa_ep04_12"
if [ ! -f "${PREFIX}_tta_balanced.zip" ]; then
  (
    cd "$CODE"
    python3 -u tools/tta_predict.py \
      --work-dir "$OUT" \
      --test-dir "$DATA/test" \
      --out-prefix "$PREFIX" \
      --scales 512,576,640 \
      --balance-strength 1.0 \
      --batch-size 48 \
      --num-workers 4 \
      --no-pin \
      --save-logits
  ) 2>&1 | tee -a "$LOG"
fi

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT/pipeline_complete"
echo "[done] Exp57 new750 complete" | tee -a "$LOG"
