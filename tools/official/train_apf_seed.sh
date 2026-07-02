#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_FAILED_APF_TRAINING:-0}" != "1" ]; then
  echo "[BLOCKED] APF-OHEM is stopped after Gate-A candidate audit failure."
  echo "Reason: candidate_to_budget_ratio_mean too low and flat_bg_ratio_mean too high."
  echo "Do not run APF seed42/43/44 for AAAI decisions."
  echo "Set ALLOW_FAILED_APF_TRAINING=1 only for explicit failure-analysis reruns."
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-/home/AAAI/OHCM-MSHNet-main}"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

APF_AUDIT="${APF_AUDIT:-docs/internal/apf_candidate_audit_seed42_train/summary.json}"
ANCHOR_DIR="${ANCHOR_DIR:-docs/internal/ohem_anchor_maps/seed42_train}"
CANDIDATE_DIR="${CANDIDATE_DIR:-docs/internal/apf_candidates/seed42_train}"
OHEM_CKPT="${OHEM_CKPT:-/home/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/NUDT-SIRST/seed_42/checkpoints/NUDT-SIRST/MSHNetOHEM_400.pth.tar}"

python tools/official/check_apf_ready.py \
  --summary "${APF_AUDIT}" \
  --anchor_dir "${ANCHOR_DIR}" \
  --candidate_dir "${CANDIDATE_DIR}" \
  --checkpoint "${OHEM_CKPT}"

echo "[BLOCKED] APF training command is intentionally not implemented at Gate-A No-Go stage."
exit 1
