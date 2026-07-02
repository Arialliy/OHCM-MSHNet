#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_FAILED_PFR_TRAINING:-0}" != "1" ]; then
  echo "[BLOCKED] PFR-MSHNet is stopped after seed42 Full Gate failure."
  echo "Reason: mIoU/Precision/FA/FP components failed on Full split."
  echo "Do not run seed43/44 or retune PFR for AAAI decisions."
  echo "Set ALLOW_FAILED_PFR_TRAINING=1 only for failure-analysis reruns."
  exit 1
fi

SEED=${1:-42}
DATASET=${DATASET:-NUDT-SIRST}
PROJECT_DIR=${PROJECT_DIR:-/home/AAAI/OHCM-MSHNet-main}
DATASET_DIR=${DATASET_DIR:-/home/AAAI/OHCM-MSHNet/datasets}
RESULT_DIR=${RESULT_DIR:-/home/AAAI/OHCM-MSHNet/results/official/PFRMSHNet/seed${SEED}}
PFR_AUDIT=${PFR_AUDIT:-docs/internal/pfr_candidate_audit_seed${SEED}_train/summary.json}
OHEM_CKPT=${OHEM_CKPT:-/home/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/${DATASET}/seed_${SEED}/checkpoints/${DATASET}/MSHNetOHEM_400.pth.tar}

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

python tools/official/check_failed_routes_blocked.py \
  --model_name PFRMSHNet \
  --allow_failed_route \
  --reason "failure-analysis rerun only"

python tools/official/check_pfr_ready.py --allow_failed_pfr --audit_summary "${PFR_AUDIT}"

python -u train.py \
  --model_names PFRMSHNet \
  --dataset_names "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --batchSize 4 \
  --patchSize 256 \
  --nEpochs 400 \
  --optimizer_name Adagrad \
  --learning_rate 0.05 \
  --mshnet_warm_epoch 5 \
  --mshnet_in_channels 1 \
  --ohem_ratio 0.01 \
  --seed "${SEED}" \
  --pfr_ready_summary "${PFR_AUDIT}" \
  --pfr_pretrained_ohem "${OHEM_CKPT}" \
  --pfr_beta 0.5 \
  --pfr_lambda_far_neg 0.5 \
  --pfr_lambda_target_protect 1.0 \
  --pfr_lambda_boundary_protect 0.5 \
  --pfr_lambda_residual_sparse 0.01 \
  --pfr_far_topk_ratio 0.005 \
  --pfr_target_dilate 3 \
  --pfr_far_dilate 9 \
  --save "${RESULT_DIR}"
