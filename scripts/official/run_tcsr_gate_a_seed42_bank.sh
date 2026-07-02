#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

DATASET=${DATASET:-NUDT-SIRST}
BASE_DIR="${ROOT}/docs/internal/tcsr/seed42_nudt"
TWA_BASE="${ROOT}/docs/internal/twa/seed42_nudt"
CKPT_DIR="${TWA_BASE}/source_checkpoints"
BANK_DIR="${BASE_DIR}/bank_train"
SUMMARY="${BASE_DIR}/tcsr_bank_summary.json"
GATE_SUMMARY="${BASE_DIR}/gate_tcsr_a_bank_summary.json"

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "Missing required file: ${p}" >&2
    exit 2
  fi
}

require_file "${CKPT_DIR}/MSHNetOHEM_250.pth.tar"
require_file "${CKPT_DIR}/MSHNetOHEM_300.pth.tar"
require_file "${CKPT_DIR}/MSHNetOHEM_350.pth.tar"
require_file "${CKPT_DIR}/MSHNetOHEM_400.pth.tar"

mkdir -p "${BASE_DIR}"

if [[ -f "${SUMMARY}" ]]; then
  echo "[skip] ${SUMMARY} already exists"
else
  python "${ROOT}/tools/official/build_tcsr_sparse_bank.py" \
    --dataset_dir "${ROOT}/datasets" \
    --dataset_name "${DATASET}" \
    --split train \
    --model_name MSHNetOHEM \
    --ohem_checkpoint "${CKPT_DIR}/MSHNetOHEM_400.pth.tar" \
    --tce_checkpoints \
      "${CKPT_DIR}/MSHNetOHEM_250.pth.tar" \
      "${CKPT_DIR}/MSHNetOHEM_300.pth.tar" \
      "${CKPT_DIR}/MSHNetOHEM_350.pth.tar" \
      "${CKPT_DIR}/MSHNetOHEM_400.pth.tar" \
    --output_dir "${BANK_DIR}"
fi

python "${ROOT}/tools/official/check_tcsr_bank_gate_a.py" \
  --summary "${SUMMARY}" \
  --output "${GATE_SUMMARY}" \
  --min_images_with_neg 50 \
  --min_neg_pixels_total 500 \
  --max_target_leakage_pixels 0 \
  --max_neg_protect_overlap_pixels 0

echo "Wrote ${GATE_SUMMARY}"
