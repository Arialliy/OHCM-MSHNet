#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
GPU="${GPU:-2}"
DATASET="${DATASET:?DATASET is required}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-4}"
THRESHOLD="${THRESHOLD:-0.5}"
RESUME="${RESUME:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"
OUT="$ROOT/results/step0_baseline/$RUN_ID/$DATASET/seed_$SEED"
OUT_IN_CONTAINER="$ROOT_IN_CONTAINER/results/step0_baseline/$RUN_ID/$DATASET/seed_$SEED"
NAME_DATASET="$(printf '%s' "$DATASET" | tr -c '[:alnum:]' '_')"
CONTAINER_NAME="ohcm_step0_${NAME_DATASET}_s${SEED}_${RUN_ID}"
RESUME_ARG=()
if [[ -n "$RESUME" ]]; then
  RESUME_IN_CONTAINER="$RESUME"
  if [[ "$RESUME" == "$ROOT/"* ]]; then
    RESUME_IN_CONTAINER="$ROOT_IN_CONTAINER/${RESUME#"$ROOT/"}"
  fi
  RESUME_ARG=(--resume "$RESUME_IN_CONTAINER")
fi

mkdir -p "$OUT"

docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=${GPU}\"" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    mkdir -p '$OUT_IN_CONTAINER'
    python train.py \
      --model_names MSHNet \
      --dataset_names '$DATASET' \
      --batchSize '$BATCH_SIZE' \
      --patchSize 256 \
      --nEpochs '$EPOCHS' \
      --optimizer_name Adagrad \
      --threads 1 \
      --intervals 10 \
      --seed '$SEED' \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      --save '$OUT_IN_CONTAINER/checkpoints' \
      ${RESUME_ARG[*]} \
      2>&1 | tee -a '$OUT_IN_CONTAINER/train_console.log'

    python tools/export_step0_predictions.py \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --checkpoint '$OUT_IN_CONTAINER/checkpoints/$DATASET/MSHNet_${EPOCHS}.pth.tar' \
      --output_dir '$OUT_IN_CONTAINER/exports' \
      --threshold '$THRESHOLD' \
      --seed '$SEED' \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      2>&1 | tee -a '$OUT_IN_CONTAINER/export_console.log'

    chown -R 1004:1004 '$OUT_IN_CONTAINER'"
