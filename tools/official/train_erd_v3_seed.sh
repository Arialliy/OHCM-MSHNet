#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
PROJECT_DIR="${PROJECT_DIR:-/home/AAAI/OHCM-MSHNet-main}"
DATASET_NAME="${DATASET_NAME:-NUDT-SIRST}"
DATASET_DIR="${DATASET_DIR:-/home/AAAI/OHCM-MSHNet/datasets}"
SAVE_ROOT="${SAVE_ROOT:-/home/AAAI/OHCM-MSHNet/results/official/ERDMSHNetV3}"
OHEM_CKPT="${OHEM_CKPT:-/home/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/NUDT-SIRST/seed_42/checkpoints/NUDT-SIRST/MSHNetOHEM_400.pth.tar}"
CANDIDATE_GATE_JSON="${CANDIDATE_GATE_JSON:-docs/internal/erd_v3_candidate_audit_train/gate_pass.json}"

cd "${PROJECT_DIR}"

failed_route_args=(--model_name ERDMSHNetV3)
if [[ "${ALLOW_FAILED_ROUTE:-0}" == "1" ]]; then
  failed_route_args+=(--allow_failed_route --reason "${FAILED_ROUTE_REASON:-failure-analysis rerun only}")
fi
python tools/official/check_failed_routes_blocked.py "${failed_route_args[@]}"

python tools/official/check_erd_v3_ready.py --candidate_gate_json "${CANDIDATE_GATE_JSON}"

CUDA_VISIBLE_DEVICES=2,3 python train.py \
  --model_names ERDMSHNetV3 \
  --dataset_names "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --batchSize "${BATCH_SIZE:-4}" \
  --patchSize "${PATCH_SIZE:-256}" \
  --nEpochs "${EPOCHS:-400}" \
  --optimizer_name "${OPTIMIZER:-Adagrad}" \
  --learning_rate "${LR:-0.05}" \
  --mshnet_warm_epoch "${MSHNET_WARM_EPOCH:-5}" \
  --mshnet_in_channels 1 \
  --ohem_ratio "${OHEM_RATIO:-0.01}" \
  --seed "${SEED}" \
  --erd_version v3_tpcs \
  --erd_pretrained_ohem "${OHEM_CKPT}" \
  --erd_aux_in_channels "${ERD_AUX_IN_CHANNELS:-16}" \
  --erd_hidden_channels "${ERD_HIDDEN_CHANNELS:-32}" \
  --erd_smax "${ERD_SMAX:-4.0}" \
  --erd_far_radius "${ERD_FAR_RADIUS:-7}" \
  --erd_target_protect_radius "${ERD_TARGET_PROTECT_RADIUS:-2}" \
  --erd_neg_topk_ratio "${ERD_NEG_TOPK_RATIO:-0.01}" \
  --erd_lambda_evidence "${ERD_LAMBDA_EVIDENCE:-0.2}" \
  --erd_lambda_protect_pos "${ERD_LAMBDA_PROTECT_POS:-0.5}" \
  --erd_lambda_protect_neg "${ERD_LAMBDA_PROTECT_NEG:-0.25}" \
  --erd_lambda_clutter_pos "${ERD_LAMBDA_CLUTTER_POS:-0.5}" \
  --erd_lambda_clutter_neg "${ERD_LAMBDA_CLUTTER_NEG:-0.25}" \
  --erd_lambda_preserve "${ERD_LAMBDA_PRESERVE:-0.5}" \
  --erd_preserve_margin "${ERD_PRESERVE_MARGIN:-0.02}" \
  --erd_require_candidate_audit \
  --erd_candidate_audit_json "${CANDIDATE_GATE_JSON}" \
  --save "${SAVE_ROOT}/seed${SEED}"
