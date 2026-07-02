#!/usr/bin/env bash
set -euo pipefail

SEED=${1:-42}
shift || true

DATASET="NUDT-SIRST"
MAX_EPOCHS=1
STAGE="activation"

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
    --stage)
      STAGE="$2"
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

COMPONENT_AUDIT=${COMPONENT_AUDIT:-"docs/internal/cga_component_target_audit_seed${SEED}_train/summary.json"}
ACTIVATION_AUDIT=${ACTIVATION_AUDIT:-"docs/internal/cga_activation_seed${SEED}_epoch1/summary.json"}
OHEM_CKPT=${OHEM_CKPT:-"${DATA_ROOT}/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/${DATASET}/seed_${SEED}/checkpoints/${DATASET}/MSHNetOHEM_400.pth.tar"}

if [[ ! -f "${OHEM_CKPT}" ]]; then
  OHEM_CKPT="${DATA_ROOT}/results/official/MSHNetOHEM/seed${SEED}/${DATASET}/MSHNetOHEM_400.pth.tar"
fi
if [[ ! -f "${OHEM_CKPT}" ]]; then
  echo "Missing OHEM checkpoint: ${OHEM_CKPT}" >&2
  exit 2
fi

if [[ "${STAGE}" == "full" ]]; then
  python tools/official/check_cga_ready.py \
    --dataset_name "${DATASET}" \
    --seed "${SEED}" \
    --stage full \
    --component_audit "${COMPONENT_AUDIT}" \
    --activation_audit "${ACTIVATION_AUDIT}"
else
  python tools/official/check_cga_ready.py \
    --dataset_name "${DATASET}" \
    --seed "${SEED}" \
    --stage activation \
    --component_audit "${COMPONENT_AUDIT}"
fi

python train.py \
  --model_names CGAMSHNet \
  --dataset_names "${DATASET}" \
  --dataset_dir "${DATA_ROOT}/datasets" \
  --batchSize 4 \
  --patchSize 256 \
  --nEpochs "${MAX_EPOCHS}" \
  --intervals 1 \
  --optimizer_name Adam \
  --learning_rate 0.0005 \
  --lambda_variant 0.2 \
  --ohem_ratio 0.01 \
  --mshnet_warm_epoch 0 \
  --mshnet_in_channels 1 \
  --load_ohem_checkpoint "${OHEM_CKPT}" \
  --cga_train_mode decoder_aux \
  --save "${DATA_ROOT}/results/official/CGAMSHNet/seed${SEED}"
