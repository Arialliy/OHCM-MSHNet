#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATASET_DIR=${DATASET_DIR:-"${ROOT}/datasets"}
HCVAL_IMAGE_LIST=${HCVAL_IMAGE_LIST:-"${ROOT}/docs/internal/hc_protocol/hcval_NUDT-SIRST.txt"}
if [[ ! -d "${DATASET_DIR}/NUDT-SIRST" ]]; then
  echo "Missing NUDT-SIRST dataset under ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${HCVAL_IMAGE_LIST}" ]]; then
  echo "Missing HC-Val image list: ${HCVAL_IMAGE_LIST}" >&2
  exit 1
fi

TWA_CKPT="${ROOT}/docs/internal/twa/seed42_nudt/twa_seed42_250_300_350_400.pth.tar"
OUT_DIR="${ROOT}/docs/internal/twa/seed42_nudt/eval_hcval_twa_no_bn"

python tools/official/evaluate_twa_checkpoint.py \
  --model_name MSHNetOHEM \
  --checkpoint "${TWA_CKPT}" \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name NUDT-SIRST \
  --split hcval \
  --image_list "${HCVAL_IMAGE_LIST}" \
  --output_dir "${OUT_DIR}" \
  --method TWAOHEM-noBN \
  --seed 42 \
  --threshold 0.5
