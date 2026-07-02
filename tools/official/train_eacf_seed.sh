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

READY_JSON="${PROJECT_ROOT}/docs/internal/eacf_scale_consensus_seed${SEED}_train/summary.json"
OHEM_CKPT=${OHEM_CKPT:-"${DATA_ROOT}/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/${DATASET}/seed_${SEED}/checkpoints/${DATASET}/MSHNetOHEM_400.pth.tar"}
if [[ ! -f "${OHEM_CKPT}" ]]; then
  OHEM_CKPT="${DATA_ROOT}/results/official/MSHNetOHEM/seed${SEED}/${DATASET}/MSHNetOHEM_400.pth.tar"
fi

python tools/official/check_failed_routes_blocked.py --model_name EACFMSHNet
python tools/official/check_eacf_ready.py \
  --scale_consensus_summary "${READY_JSON}" \
  --ohem_checkpoint "${OHEM_CKPT}"

python train.py \
  --model_names EACFMSHNet \
  --dataset_names "${DATASET}" \
  --dataset_dir "${DATA_ROOT}/datasets" \
  --batchSize 4 \
  --patchSize 256 \
  --nEpochs 80 \
  --optimizer_name Adagrad \
  --learning_rate 0.005 \
  --lambda_variant 0.2 \
  --ohem_ratio 0.01 \
  --mshnet_warm_epoch 0 \
  --mshnet_in_channels 1 \
  --pretrained_ohem_checkpoint "${OHEM_CKPT}" \
  --eacf_freeze_backbone \
  --eacf_eta_max 0.5 \
  --eacf_lambda_anchor 0.5 \
  --eacf_lambda_scale_bg 0.05 \
  --eacf_lambda_scale_target 0.02 \
  --save "${DATA_ROOT}/results/official/EACFMSHNet/seed${SEED}"
