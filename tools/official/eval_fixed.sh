#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CHECKPOINT MODEL_NAME OUTPUT_DIR [IMAGE_LIST]" >&2
  exit 2
fi

CHECKPOINT="$1"
MODEL_NAME="$2"
OUTPUT_DIR="$3"
IMAGE_LIST="${4:-}"
DATASET="${DATASET:-NUDT-SIRST}"
DATASET_DIR="${DATASET_DIR:-./datasets}"
METHOD="${METHOD:-${MODEL_NAME}}"
SEED="${SEED:-}"

cmd=(python tools/official/evaluate_checkpoint_direct.py
  --dataset_dir "${DATASET_DIR}"
  --dataset_name "${DATASET}"
  --model_name "${MODEL_NAME}"
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --method "${METHOD}"
  --threshold "${THRESHOLD:-0.5}"
  --thresholds "${THRESHOLDS:-0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95}")

if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ -n "${IMAGE_LIST}" ]]; then
  cmd+=(--image_list "${IMAGE_LIST}")
fi

"${cmd[@]}"
