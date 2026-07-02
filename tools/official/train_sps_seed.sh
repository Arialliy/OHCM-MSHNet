#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-${SEED:-42}}"
DATASET="${DATASET:-NUDT-SIRST}"
DATASET_DIR="${DATASET_DIR:-./datasets}"
SAVE_ROOT="${SAVE_ROOT:-./results/official/MSHNetSPSOHEM}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

failed_route_args=(--model_name MSHNetSPSOHEM)
if [[ "${ALLOW_FAILED_ROUTE:-0}" == "1" ]]; then
  failed_route_args+=(--allow_failed_route --reason "${FAILED_ROUTE_REASON:-failure-analysis rerun only}")
fi
python tools/official/check_failed_routes_blocked.py "${failed_route_args[@]}"

if [[ -z "${PRETRAINED:-}" ]]; then
  echo "PRETRAINED must point to the paired MSHNetOHEM checkpoint." >&2
  exit 2
fi

target_safe_args=()
if [[ "${SPS_TARGET_SAFE:-0}" == "1" ]]; then
  target_safe_args=(
    --sps_target_safe
    --sps_target_safe_u_low "${SPS_TARGET_SAFE_U_LOW:-0.02}"
    --sps_target_safe_u_high "${SPS_TARGET_SAFE_U_HIGH:-0.08}"
    --sps_target_safe_conf_min "${SPS_TARGET_SAFE_CONF_MIN:-0.55}"
    --sps_target_safe_conf_floor "${SPS_TARGET_SAFE_CONF_FLOOR:-0.35}"
    --sps_target_safe_alpha_floor "${SPS_TARGET_SAFE_ALPHA_FLOOR:-0.0}"
  )
fi

two_view_args=(--sps_no_two_view_base)
if [[ "${SPS_TWO_VIEW_BASE:-0}" == "1" ]]; then
  two_view_args=(--sps_two_view_base)
fi

far_mask_args=()
if [[ "${SPS_DISABLE_FAR_MASK:-0}" == "1" ]]; then
  far_mask_args=(--sps_disable_far_mask)
fi

candidate_min_args=()
if [[ -n "${SPS_CANDIDATE_MIN_METRIC:-}" ]]; then
  candidate_min_args=(--sps_candidate_min_metric "${SPS_CANDIDATE_MIN_METRIC}")
fi

parallel_args=()
if [[ "${USE_PARALLEL:-0}" == "1" || "${GPU_IDS}" == *","* ]]; then
  parallel_args=(--use_parallel)
fi

candidate_min_conf_args=()
if [[ -n "${SPS_CANDIDATE_MIN_CONFIDENCE:-}" ]]; then
  candidate_min_conf_args=(--sps_candidate_min_confidence "${SPS_CANDIDATE_MIN_CONFIDENCE}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -u train.py \
  --model_names MSHNetSPSOHEM \
  --dataset_names "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --batchSize "${BATCH_SIZE:-16}" \
  --patchSize "${PATCH_SIZE:-256}" \
  --nEpochs "${EPOCHS:-150}" \
  --optimizer_name "${OPTIMIZER:-Adagrad}" \
  --learning_rate "${LR:-0.005}" \
  --threads "${THREADS:-0}" \
  --seed "${SEED}" \
  "${parallel_args[@]}" \
  --pretrained "${PRETRAINED}" \
  --mshnet_warm_epoch "${MSHNET_WARM_EPOCH:-5}" \
  --mshnet_in_channels 1 \
  --lambda_variant "${LAMBDA_VARIANT:-0.2}" \
  --ohem_ratio "${OHEM_RATIO:-0.01}" \
  --sps_lambda "${SPS_LAMBDA:-0.15}" \
  --sps_mode "${SPS_MODE:-sps}" \
  --sps_objective "${SPS_OBJECTIVE:-rerank}" \
  "${two_view_args[@]}" \
  --sps_start_epoch "${SPS_START_EPOCH:-50}" \
  --sps_end_epoch "${SPS_END_EPOCH:-150}" \
  --sps_perturbation "${SPS_PERTURBATION:-gain_offset}" \
  --sps_gain_min "${SPS_GAIN_MIN:-1.005}" \
  --sps_gain_max "${SPS_GAIN_MAX:-1.005}" \
  --sps_offset_abs "${SPS_OFFSET_ABS:-0.0}" \
  --sps_candidate_tau "${SPS_CANDIDATE_TAU:-0.7}" \
  --sps_candidate_topk_ratio "${SPS_CANDIDATE_TOPK_RATIO:-0.0}" \
  --sps_candidate_topk_metric "${SPS_CANDIDATE_TOPK_METRIC:-sps_score}" \
  "${candidate_min_args[@]}" \
  "${candidate_min_conf_args[@]}" \
  --sps_candidate_fallback_topk_ratio "${SPS_CANDIDATE_FALLBACK_TOPK_RATIO:-0.00001}" \
  --sps_candidate_expand_radius "${SPS_CANDIDATE_EXPAND_RADIUS:-0}" \
  --sps_candidate_expand_min_confidence "${SPS_CANDIDATE_EXPAND_MIN_CONFIDENCE:-0.0}" \
  --sps_target_margin_quantile "${SPS_TARGET_MARGIN_QUANTILE:-0.85}" \
  --sps_target_margin_temp "${SPS_TARGET_MARGIN_TEMP:-0.01}" \
  --sps_target_margin_min "${SPS_TARGET_MARGIN_MIN:-0.0}" \
  --sps_rerank_strict_fallback \
  "${far_mask_args[@]}" \
  "${target_safe_args[@]}" \
  --save "${SAVE_ROOT}/${DATASET}/seed_${SEED}/checkpoints"
