#!/usr/bin/env bash
set -euo pipefail

SEED=${1:-42}
DATASET=${2:-NUDT-SIRST}

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
if [[ -d /home/AAAI/OHCM-MSHNet/datasets ]]; then
  DATA_ROOT=${DATA_ROOT:-/home/AAAI/OHCM-MSHNet}
else
  DATA_ROOT=${DATA_ROOT:-/home/ly/AAAI/OHCM-MSHNet}
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

EACF_CKPT=${EACF_CKPT:-"${DATA_ROOT}/results/official/EACFMSHNet/seed${SEED}/${DATASET}/EACFMSHNet_80.pth.tar"}
OHEM_CKPT=${OHEM_CKPT:-"${DATA_ROOT}/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/${DATASET}/seed_${SEED}/checkpoints/${DATASET}/MSHNetOHEM_400.pth.tar"}
if [[ ! -f "${OHEM_CKPT}" ]]; then
  OHEM_CKPT="${DATA_ROOT}/results/official/MSHNetOHEM/seed${SEED}/${DATASET}/MSHNetOHEM_400.pth.tar"
fi

python tools/official/evaluate_eacf_heads.py \
  --dataset_dir "${DATA_ROOT}/datasets" \
  --dataset_name "${DATASET}" \
  --split test \
  --model_name EACFMSHNet \
  --checkpoint "${EACF_CKPT}" \
  --ohem_checkpoint "${OHEM_CKPT}" \
  --threshold 0.5 \
  --output_dir "docs/internal/eacf_seed${SEED}_full_head_audit"
