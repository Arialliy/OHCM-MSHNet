#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
DATASET="${DATASET:?DATASET is required}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET}"
SEED="${SEED:-42}"
METHOD="${METHOD:-}"
EXPORTS="${EXPORTS:?EXPORTS is required}"
OUT="${OUT:?OUT is required}"
IMAGE_LIST="${IMAGE_LIST:-}"
THRESHOLD="${THRESHOLD:-0.5}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"

to_container_path() {
  local path="$1"
  if [[ "$path" == "$ROOT/"* ]]; then
    printf '%s/%s' "$ROOT_IN_CONTAINER" "${path#"$ROOT/"}"
  else
    printf '%s' "$path"
  fi
}

exports_in_container="$(to_container_path "$EXPORTS")"
out_in_container="$(to_container_path "$OUT")"
image_list_arg=()
if [[ -n "$IMAGE_LIST" ]]; then
  image_list_arg=(--image_list "$(to_container_path "$IMAGE_LIST")")
fi

mkdir -p "$OUT"

name_dataset="$(printf '%s' "$DATASET" | tr -c '[:alnum:]' '_')"
container_name="ohcm_eval_${name_dataset}_s${SEED}_$(date +%Y%m%d%H%M%S)"

docker run --rm \
  --name "$container_name" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    mkdir -p '$out_in_container'
    python tools/evaluate_prediction_exports.py \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --train_dataset_name '$TRAIN_DATASET' \
      --exports_dir '$exports_in_container' \
      --output_dir '$out_in_container' \
      --method '$METHOD' \
      --seed '$SEED' \
      --threshold '$THRESHOLD' \
      ${image_list_arg[*]} \
      2>&1 | tee '$out_in_container/evaluate_console.log'
    chown -R 1004:1004 '$out_in_container'"
