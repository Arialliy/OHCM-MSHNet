#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATASET=${DATASET:-NUDT-SIRST}
THRESHOLD=${THRESHOLD:-0.5}

BASE_DIR="${ROOT}/docs/internal/twa/seed42_nudt"
CKPT_DIR="${BASE_DIR}/source_checkpoints"

OHEM_FULL_SUMMARY=${OHEM_FULL_SUMMARY:-"${BASE_DIR}/eval_full_ohem/summary_metrics.json"}
OHEM_HCVAL_SUMMARY=${OHEM_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_ohem/summary_metrics.json"}
TWA_FULL_SUMMARY=${TWA_FULL_SUMMARY:-"${BASE_DIR}/eval_full_twa4_no_bn/summary_metrics.json"}
TWA_HCVAL_SUMMARY=${TWA_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_twa4_no_bn/summary_metrics.json"}
GATE_E_SUMMARY=${GATE_E_SUMMARY:-"${BASE_DIR}/gate_twa_e_summary.json"}
EP250_GATE_A_SUMMARY=${EP250_GATE_A_SUMMARY:-"${BASE_DIR}/gate_late_snapshot_ep250_a_summary.json"}

EP250_FULL_SUMMARY=${EP250_FULL_SUMMARY:-"${BASE_DIR}/eval_full_single_ep250/summary_metrics.json"}
EP250_HCVAL_SUMMARY=${EP250_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_single_ep250/summary_metrics.json"}
EP300_FULL_SUMMARY=${EP300_FULL_SUMMARY:-"${BASE_DIR}/eval_full_single_ep300/summary_metrics.json"}
EP300_HCVAL_SUMMARY=${EP300_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_single_ep300/summary_metrics.json"}
EP350_FULL_SUMMARY=${EP350_FULL_SUMMARY:-"${BASE_DIR}/eval_full_single_ep350/summary_metrics.json"}
EP350_HCVAL_SUMMARY=${EP350_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_single_ep350/summary_metrics.json"}
OUTPUT=${OUTPUT:-"${BASE_DIR}/gate_twa_e2_fullsafe_single_control_summary.json"}

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "Missing required file: ${p}" >&2
    exit 2
  fi
}

eval_single_if_missing() {
  local epoch="$1"
  local split="$2"
  local summary_path="$3"
  local out_dir
  out_dir="$(dirname "${summary_path}")"

  if [[ -f "${summary_path}" ]]; then
    echo "[skip] ${summary_path} already exists"
    return 0
  fi

  local ckpt="${CKPT_DIR}/MSHNetOHEM_${epoch}.pth.tar"
  require_file "${ckpt}"

  echo "[eval] ep${epoch} split=${split} -> ${out_dir}"
  python "${ROOT}/tools/official/evaluate_twa_checkpoint.py" \
    --model_name MSHNetOHEM \
    --checkpoint "${ckpt}" \
    --dataset_dir "${ROOT}/datasets" \
    --dataset_name "${DATASET}" \
    --split "${split}" \
    --output_dir "${out_dir}" \
    --threshold "${THRESHOLD}" \
    --method "single_ep${epoch}" \
    --seed 42

  require_file "${summary_path}"
}

require_file "${OHEM_FULL_SUMMARY}"
require_file "${OHEM_HCVAL_SUMMARY}"
require_file "${TWA_FULL_SUMMARY}"
require_file "${TWA_HCVAL_SUMMARY}"
require_file "${GATE_E_SUMMARY}"
require_file "${EP250_GATE_A_SUMMARY}"
require_file "${EP250_FULL_SUMMARY}"
require_file "${EP250_HCVAL_SUMMARY}"

eval_single_if_missing 300 full "${EP300_FULL_SUMMARY}"
eval_single_if_missing 300 hcval "${EP300_HCVAL_SUMMARY}"
eval_single_if_missing 350 full "${EP350_FULL_SUMMARY}"
eval_single_if_missing 350 hcval "${EP350_HCVAL_SUMMARY}"

python "${ROOT}/tools/official/check_twa_gate_e2_fullsafe_single_control.py" \
  --gate_e_summary "${GATE_E_SUMMARY}" \
  --ep250_gate_a_summary "${EP250_GATE_A_SUMMARY}" \
  --ohem_full "${OHEM_FULL_SUMMARY}" \
  --ohem_hcval "${OHEM_HCVAL_SUMMARY}" \
  --twa_full "${TWA_FULL_SUMMARY}" \
  --twa_hcval "${TWA_HCVAL_SUMMARY}" \
  --snapshot "250:${EP250_FULL_SUMMARY}:${EP250_HCVAL_SUMMARY}" \
  --snapshot "300:${EP300_FULL_SUMMARY}:${EP300_HCVAL_SUMMARY}" \
  --snapshot "350:${EP350_FULL_SUMMARY}:${EP350_HCVAL_SUMMARY}" \
  --output "${OUTPUT}"

echo "Wrote ${OUTPUT}"
