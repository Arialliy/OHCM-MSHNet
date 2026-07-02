#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
DATASET="${DATASET:?DATASET is required}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
THRESHOLD="${THRESHOLD:-0.5}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
ROOT_IN_CONTAINER="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/step0_baseline/$RUN_ID"
RUN_ROOT_IN_CONTAINER="$ROOT_IN_CONTAINER/results/step0_baseline/$RUN_ID"
EXPORTS="$RUN_ROOT/$DATASET/seed_$SEED/exports"
EXPORTS_IN_CONTAINER="$RUN_ROOT_IN_CONTAINER/$DATASET/seed_$SEED/exports"
OUT="$RUN_ROOT/$DATASET/seed_$SEED/step1"
OUT_IN_CONTAINER="$RUN_ROOT_IN_CONTAINER/$DATASET/seed_$SEED/step1"
NAME_DATASET="$(printf '%s' "$DATASET" | tr -c '[:alnum:]' '_')"
CONTAINER_NAME="ohcm_step1_${NAME_DATASET}_s${SEED}_${RUN_ID}"

if [[ ! -f "$EXPORTS/summary_metrics.json" ]]; then
  printf 'Missing Step0 export summary: %s\n' "$EXPORTS/summary_metrics.json" >&2
  exit 2
fi

mkdir -p "$OUT"

docker run --rm \
  --name "$CONTAINER_NAME" \
  -v /home/ly:/home \
  -w "$ROOT_IN_CONTAINER" \
  "$IMAGE" \
  bash -lc "set -euo pipefail
    python tools/analyze_step1_hard_clutter.py \
      --dataset_dir ./datasets \
      --dataset_name '$DATASET' \
      --exports_dir '$EXPORTS_IN_CONTAINER' \
      --output_dir '$OUT_IN_CONTAINER' \
      --seed '$SEED' \
      --threshold '$THRESHOLD' \
      2>&1 | tee '$OUT_IN_CONTAINER/step1_console.log'
    chown -R 1004:1004 '$OUT_IN_CONTAINER'"
