#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AUDIT_JSON="${AUDIT_JSON:-}"

cd "${PROJECT_DIR}"

failed_route_args=(--model_name ERDMSHNet)
if [[ "${ALLOW_FAILED_ROUTE:-0}" == "1" ]]; then
  failed_route_args+=(--allow_failed_route --reason "${FAILED_ROUTE_REASON:-failure-analysis rerun only}")
fi
python tools/official/check_failed_routes_blocked.py "${failed_route_args[@]}"

if [[ -z "${AUDIT_JSON}" ]]; then
  echo "ERROR: AUDIT_JSON is required. Run audit_online_gate_candidates.py first."
  exit 1
fi

python tools/official/check_erd_ready.py --audit_json "${AUDIT_JSON}"

python train.py \
  --model_names ERDMSHNet \
  --dataset_names "${DATASET_NAME:-NUDT-SIRST}" \
  --dataset_dir "${DATASET_DIR:-/home/AAAI/OHCM-MSHNet/datasets}" \
  --batchSize "${BATCH_SIZE:-4}" \
  --patchSize "${PATCH_SIZE:-256}" \
  --nEpochs "${EPOCHS:-400}" \
  --optimizer_name "${OPTIMIZER:-Adagrad}" \
  --learning_rate "${LR:-0.05}" \
  --mshnet_warm_epoch "${MSHNET_WARM_EPOCH:-5}" \
  --mshnet_in_channels 1 \
  --ohem_ratio "${OHEM_RATIO:-0.01}" \
  --erd_rho "${ERD_RHO:-0.25}" \
  --erd_gamma_max "${ERD_GAMMA_MAX:-1.0}" \
  --erd_gate_start_epoch "${ERD_GATE_START_EPOCH:-20}" \
  --erd_gate_ramp_epochs "${ERD_GATE_RAMP_EPOCHS:-30}" \
  --erd_lambda_evidence "${ERD_LAMBDA_EVIDENCE:-0.2}" \
  --erd_lambda_gate_pos "${ERD_LAMBDA_GATE_POS:-0.05}" \
  --erd_lambda_gate_neg "${ERD_LAMBDA_GATE_NEG:-0.20}" \
  --erd_gate_target_radius "${ERD_GATE_TARGET_RADIUS:-2}" \
  --erd_gate_far_radius "${ERD_GATE_FAR_RADIUS:-5}" \
  --erd_gate_neg_q "${ERD_GATE_NEG_Q:-0.01}" \
  --erd_gate_neg_min_k "${ERD_GATE_NEG_MIN_K:-16}" \
  --erd_gate_neg_max_k "${ERD_GATE_NEG_MAX_K:-512}" \
  --erd_gate_audit_json "${AUDIT_JSON}" \
  --seed "${SEED}" \
  --save "${SAVE_ROOT:-/home/AAAI/OHCM-MSHNet/results/official/ERDMSHNet}/seed${SEED}"
