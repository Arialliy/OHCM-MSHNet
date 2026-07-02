#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
PROJECT_DIR="${PROJECT_DIR:-/home/AAAI/OHCM-MSHNet-main}"
DATASET_NAME="${DATASET_NAME:-NUDT-SIRST}"
DATASET_DIR="${DATASET_DIR:-/home/AAAI/OHCM-MSHNet/datasets}"
CKPT="${CKPT:-/home/AAAI/OHCM-MSHNet/results/official/ERDMSHNetV3/seed${SEED}/${DATASET_NAME}/ERDMSHNetV3_400.pth.tar}"
OUT_ROOT="${OUT_ROOT:-/home/AAAI/OHCM-MSHNet/results/official/ERDMSHNetV3/seed${SEED}/eval}"
HCVAL_LIST="${HCVAL_LIST:-/home/AAAI/OHCM-MSHNet/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hcval_NUDT-SIRST.txt}"

cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES=2,3 python tools/official/evaluate_checkpoint_direct.py \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --model_name ERDMSHNetV3 \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT_ROOT}/full" \
  --method ERDMSHNetV3_TPCS \
  --seed "${SEED}"

CUDA_VISIBLE_DEVICES=2,3 python tools/official/evaluate_checkpoint_direct.py \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --model_name ERDMSHNetV3 \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT_ROOT}/hcval" \
  --image_list "${HCVAL_LIST}" \
  --method ERDMSHNetV3_TPCS \
  --seed "${SEED}"
