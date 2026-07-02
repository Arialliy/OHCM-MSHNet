#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
DATASET="${DATASET:?DATASET is required}"
SEED="${SEED:-42}"
EXPORTS="${EXPORTS:?EXPORTS is required}"
OUT="${OUT:?OUT is required}"
THRESHOLD="${THRESHOLD:-0.5}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"

exports_in_container="$EXPORTS"
if [[ "$EXPORTS" == "$ROOT/"* ]]; then
  exports_in_container="$ROOT_IN_CONTAINER/${EXPORTS#"$ROOT/"}"
fi

out_in_container="$OUT"
if [[ "$OUT" == "$ROOT/"* ]]; then
  out_in_container="$ROOT_IN_CONTAINER/${OUT#"$ROOT/"}"
fi

mkdir -p "$OUT"

name_dataset="$(printf '%s' "$DATASET" | tr -c '[:alnum:]' '_')"
container_name="ohcm_step1_${name_dataset}_s${SEED}_$(date +%Y%m%d%H%M%S)"

docker run --rm \
  --name "$container_name" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    mkdir -p '$out_in_container'
    python tools/analyze_step1_hard_clutter.py \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --exports_dir '$exports_in_container' \
      --output_dir '$out_in_container' \
      --seed '$SEED' \
      --threshold '$THRESHOLD' \
      2>&1 | tee '$out_in_container/step1_console.log'
    chown -R 1004:1004 '$out_in_container'"
