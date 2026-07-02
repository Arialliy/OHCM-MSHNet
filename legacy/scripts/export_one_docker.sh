#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
GPU="${GPU:-}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
DATASET="${DATASET:?DATASET is required}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUT="${OUT:?OUT is required}"
EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS:-}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"

checkpoint_in_container="$CHECKPOINT"
if [[ "$CHECKPOINT" == "$ROOT/"* ]]; then
  checkpoint_in_container="$ROOT_IN_CONTAINER/${CHECKPOINT#"$ROOT/"}"
fi

out_in_container="$OUT"
if [[ "$OUT" == "$ROOT/"* ]]; then
  out_in_container="$ROOT_IN_CONTAINER/${OUT#"$ROOT/"}"
fi

mkdir -p "$OUT"

name_dataset="$(printf '%s' "$DATASET" | tr -c '[:alnum:]' '_')"
name_model="$(printf '%s' "$MODEL_NAME" | tr -c '[:alnum:]' '_')"
container_name="ohcm_export_${name_model}_${name_dataset}_s${SEED}_$(date +%Y%m%d%H%M%S)"

gpu_args=()
if [[ -n "$GPU" ]]; then
  gpu_args=(--gpus "\"device=${GPU}\"")
fi

docker run --rm \
  --name "$container_name" \
  "${gpu_args[@]}" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    mkdir -p '$out_in_container'
    python tools/export_step0_predictions.py \
      --model_name '$MODEL_NAME' \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --train_dataset_name '$TRAIN_DATASET' \
      --checkpoint '$checkpoint_in_container' \
      --output_dir '$out_in_container' \
      --seed '$SEED' \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      $EXPORT_EXTRA_ARGS \
      2>&1 | tee '$out_in_container/export_console.log'
    chown -R 1004:1004 '$out_in_container'"
