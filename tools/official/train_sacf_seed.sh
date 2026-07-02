#!/usr/bin/env bash
set -euo pipefail

SEED=${1:-42}
shift || true

DATASET="NUDT-SIRST"
MAX_EPOCHS=1
FREEZE_EVIDENCE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --freeze_evidence)
      FREEZE_EVIDENCE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
if [[ -d /home/AAAI/OHCM-MSHNet/datasets ]]; then
  DATA_ROOT=${DATA_ROOT:-/home/AAAI/OHCM-MSHNet}
else
  DATA_ROOT=${DATA_ROOT:-/home/ly/AAAI/OHCM-MSHNet}
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

OHEM_CKPT=${OHEM_CKPT:-"${DATA_ROOT}/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/${DATASET}/seed_${SEED}/checkpoints/${DATASET}/MSHNetOHEM_400.pth.tar"}
if [[ ! -f "${OHEM_CKPT}" ]]; then
  OHEM_CKPT="${DATA_ROOT}/results/official/MSHNetOHEM/seed${SEED}/${DATASET}/MSHNetOHEM_400.pth.tar"
fi

python tools/official/check_failed_routes_blocked.py --model_name SACFMSHNet

python train.py \
  --model_names SACFMSHNet \
  --dataset_names "${DATASET}" \
  --dataset_dir "${DATA_ROOT}/datasets" \
  --batchSize 4 \
  --patchSize 256 \
  --nEpochs "${MAX_EPOCHS}" \
  --intervals 1 \
  --optimizer_name Adagrad \
  --learning_rate 0.005 \
  --lambda_variant 0.2 \
  --ohem_ratio 0.01 \
  --mshnet_warm_epoch 0 \
  --mshnet_in_channels 1 \
  --ohem_checkpoint "${OHEM_CKPT}" \
  --freeze_evidence "${FREEZE_EVIDENCE}" \
  --sacf_delta_max 1.0 \
  --sacf_lambda_anchor 0.05 \
  --sacf_lambda_scale 0.20 \
  --sacf_lambda_disagree_bg 0.10 \
  --save "${DATA_ROOT}/results/official/SACFMSHNet/seed${SEED}"
