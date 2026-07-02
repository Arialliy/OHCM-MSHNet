#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/AAAI/OHCM-MSHNet}"
DATASET="${DATASET:-NUDT-SIRST}"
RUN_ID="${RUN_ID:-20260622_tsr_ohem_stage2}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LAMBDA_REGION="${LAMBDA_REGION:-0.05}"
STAGE01_RUN_ID="${STAGE01_RUN_ID:-20260622_tsr_ohem}"
STAGE01_ROOT="$ROOT/results/tsr_ohem/$STAGE01_RUN_ID"
STATS_JSON="${STATS_JSON:-$STAGE01_ROOT/stage0_target_scales/${DATASET}_target_scale_stats.json}"
RUN_ROOT="$ROOT/results/aaai_p0_paired/$RUN_ID"
HC_PROTOCOL_SRC="${HC_PROTOCOL_SRC:-$ROOT/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol}"

cd "$ROOT"
mkdir -p "$RUN_ROOT"
if [[ -d "$HC_PROTOCOL_SRC" && ! -d "$RUN_ROOT/hc_protocol" ]]; then
  cp -a "$HC_PROTOCOL_SRC" "$RUN_ROOT/hc_protocol"
fi

if [[ ! -f "$STATS_JSON" ]]; then
  mkdir -p "$(dirname "$STATS_JSON")"
  CUDA_VISIBLE_DEVICES="$GPUS" python tools/tsr_target_scale_stats.py \
    --dataset_dir ./datasets \
    --dataset_name "$DATASET" \
    --output_json "$STATS_JSON"
fi

SCALES="$(python -c "import json; print(json.load(open('$STATS_JSON'))['target_scales_arg'])")"
DILATE="$(python -c "import json; print(json.load(open('$STATS_JSON'))['safe_background_dilation_radius'])")"

{
  echo "RUN_ID=$RUN_ID"
  echo "dataset=$DATASET"
  echo "seed=$SEED"
  echo "gpus=$GPUS"
  echo "epochs=$EPOCHS"
  echo "batch_size=$BATCH_SIZE"
  echo "lambda_region=$LAMBDA_REGION"
  echo "target_scales=$SCALES"
  echo "safe_dilation_radius=$DILATE"
  echo "B0=existing MSHNetOHEM paired baseline"
  echo "R1=target-scale miner + region negative BCE"
  echo "R2=target-scale miner + region ranking"
} > "$RUN_ROOT/manifest_tsr_stage2.txt"

run_tsr_method() {
  local method="$1"
  local loss_mode="$2"
  local train_args
  train_args="--use_parallel --tsr_lambda_region $LAMBDA_REGION --tsr_region_start_epoch 60 --tsr_region_end_epoch 100 --tsr_target_scales $SCALES --tsr_region_loss_mode $loss_mode --tsr_beta 0.5 --tsr_topk 3 --tsr_nms_iou 0.3 --tsr_weight_temp 0.2 --tsr_target_temp 0.25 --tsr_hard_temp 0.25 --tsr_rank_temp 0.5 --tsr_margin 0.5 --tsr_topq 0.25 --tsr_dilate_radius $DILATE"
  RUN_ID="$RUN_ID" \
  METHOD="$method" \
  MODEL_NAME="MSHNetOHEM" \
  GPU="$GPUS" \
  DATASET="$DATASET" \
  SEED="$SEED" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  TRAIN_EXTRA_ARGS="$train_args" \
  EXPORT_EXTRA_ARGS="" \
  bash scripts/aaai_p0_run_one_3ca_inner.sh
}

run_tsr_method "TSR-OHEM-R1" "neg_bce"
run_tsr_method "TSR-OHEM-R2" "rank"

python tools/summarize_aaai_p0_paired.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds "$SEED"

python tools/analyze_threshold_matched.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds "$SEED"

chown -R 1004:1004 "$RUN_ROOT" || true
