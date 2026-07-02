#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-${SEED:-42}}"
DATASET="${DATASET:-NUDT-SIRST}"
DATASET_DIR="${DATASET_DIR:-./datasets}"
SAVE_ROOT="${SAVE_ROOT:-./results/official/MSHNetOHEM}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"

parallel_args=()
if [[ "${USE_PARALLEL:-0}" == "1" || "${GPU_IDS}" == *","* ]]; then
  parallel_args=(--use_parallel)
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python train.py \
  --model_names MSHNetOHEM \
  --dataset_names "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --batchSize "${BATCH_SIZE:-16}" \
  --patchSize "${PATCH_SIZE:-256}" \
  --nEpochs "${EPOCHS:-400}" \
  --optimizer_name "${OPTIMIZER:-Adagrad}" \
  --learning_rate "${LR:-0.05}" \
  --threads "${THREADS:-0}" \
  --seed "${SEED}" \
  "${parallel_args[@]}" \
  --mshnet_warm_epoch "${MSHNET_WARM_EPOCH:-5}" \
  --mshnet_in_channels 1 \
  --lambda_variant "${LAMBDA_VARIANT:-0.2}" \
  --ohem_ratio "${OHEM_RATIO:-0.01}" \
  --save "${SAVE_ROOT}/${DATASET}/seed_${SEED}/checkpoints"
