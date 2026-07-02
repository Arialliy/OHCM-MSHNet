#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260623_pcar_stage400}"
STAGE200_RUN_ID="${STAGE200_RUN_ID:-20260623_pcar_stage200}"
METHOD="${METHOD:?METHOD is required}"
MODEL_NAME="${MODEL_NAME:-MSHNetOHEM}"
GPU="${GPU:?GPU is required}"
DATASET="${DATASET:-NUDT-SIRST}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-4}"
THRESHOLD="${THRESHOLD:-0.5}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS:-}"
ROOT="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/pcar_ohem/$RUN_ID"
STAGE200_ROOT="$ROOT/results/pcar_ohem/$STAGE200_RUN_ID"
OUT="$RUN_ROOT/$METHOD/$DATASET/seed_$SEED"
STAGE200_CKPT="$STAGE200_ROOT/$METHOD/$DATASET/seed_$SEED/checkpoints/$DATASET/${MODEL_NAME}_200.pth.tar"
FINAL_CKPT="$OUT/checkpoints/$DATASET/${MODEL_NAME}_${EPOCHS}.pth.tar"
HC_LIST="${HC_LIST:-$ROOT/results/step0_baseline/20260611_155232/step2_hcset/hcset_${DATASET}.txt}"
HC_VAL_LIST="${HC_VAL_LIST:-$ROOT/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hcval_${DATASET}.txt}"
HC_TEST_LIST="${HC_TEST_LIST:-$ROOT/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hctest_${DATASET}.txt}"
LOCK_ROOT="$RUN_ROOT/.locks"
LOCK_DIR="$LOCK_ROOT/${METHOD}_${DATASET}_seed_${SEED}.lock"

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
read -r -a train_extra_args <<< "$TRAIN_EXTRA_ARGS"
read -r -a export_extra_args <<< "$EXPORT_EXTRA_ARGS"

if [[ ! -f "$STAGE200_CKPT" ]]; then
  echo "Missing stage-200 checkpoint: $STAGE200_CKPT" >&2
  exit 1
fi

if [[ -f "$OUT/eval_full/summary_metrics.json" \
   && -f "$OUT/eval_hcval/summary_metrics.json" \
   && -f "$OUT/eval_dev_hc/summary_metrics.json" \
   && -f "$OUT/eval_hctest/summary_metrics.json" \
   && -f "$OUT/fp_analysis/step1_summary.json" ]]; then
  echo "[$(date '+%F %T')] SKIP completed method=$METHOD seed=$SEED"
  exit 0
fi

mkdir -p "$LOCK_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  lock_pid=""
  [[ -f "$LOCK_DIR/pid" ]] && lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ -n "$lock_pid" ]] && ! ps -p "$lock_pid" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] REMOVE stale lock method=$METHOD seed=$SEED lock=$LOCK_DIR"
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
  else
    echo "[$(date '+%F %T')] SKIP active lock method=$METHOD seed=$SEED lock=$LOCK_DIR pid=${lock_pid:-unknown}"
    exit 0
  fi
fi
trap 'rm -rf "$LOCK_DIR"' EXIT
printf '%s\n' "$$" > "$LOCK_DIR/pid"
printf '%s\n' "$(date '+%F %T')" > "$LOCK_DIR/started_at"
printf 'method=%s dataset=%s seed=%s gpu=%s epochs=%s resume=%s\n' "$METHOD" "$DATASET" "$SEED" "$GPU" "$EPOCHS" "$STAGE200_CKPT" > "$LOCK_DIR/task"

if [[ -f "$FINAL_CKPT" ]]; then
  echo "[$(date '+%F %T')] SKIP train existing checkpoint: $FINAL_CKPT" | tee -a "$OUT/train_console.log"
else
  python train.py \
    --model_names "$MODEL_NAME" \
    --dataset_names "$DATASET" \
    --batchSize "$BATCH_SIZE" \
    --patchSize 256 \
    --nEpochs "$EPOCHS" \
    --optimizer_name Adagrad \
    --threads 1 \
    --intervals 10 \
    --seed "$SEED" \
    --mshnet_warm_epoch 5 \
    --mshnet_in_channels 1 \
    --resume "$STAGE200_CKPT" \
    --save "$OUT/checkpoints" \
    "${train_extra_args[@]}" \
    2>&1 | tee -a "$OUT/train_console.log"
fi

python tools/export_step0_predictions.py \
  --model_name "$MODEL_NAME" \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --train_dataset_name "$DATASET" \
  --checkpoint "$FINAL_CKPT" \
  --output_dir "$OUT/exports" \
  --threshold "$THRESHOLD" \
  --seed "$SEED" \
  --mshnet_warm_epoch 5 \
  --mshnet_in_channels 1 \
  "${export_extra_args[@]}" \
  2>&1 | tee "$OUT/export_console.log"

python tools/evaluate_prediction_exports.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --train_dataset_name "$DATASET" \
  --exports_dir "$OUT/exports" \
  --output_dir "$OUT/eval_full" \
  --method "$METHOD" \
  --seed "$SEED" \
  --threshold "$THRESHOLD" \
  2>&1 | tee "$OUT/eval_full_console.log"

python tools/evaluate_prediction_exports.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --train_dataset_name "$DATASET" \
  --exports_dir "$OUT/exports" \
  --output_dir "$OUT/eval_hcval" \
  --image_list "$HC_VAL_LIST" \
  --method "$METHOD" \
  --seed "$SEED" \
  --threshold "$THRESHOLD" \
  2>&1 | tee "$OUT/eval_hcval_console.log"

python tools/evaluate_prediction_exports.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --train_dataset_name "$DATASET" \
  --exports_dir "$OUT/exports" \
  --output_dir "$OUT/eval_dev_hc" \
  --image_list "$HC_LIST" \
  --method "$METHOD" \
  --seed "$SEED" \
  --threshold "$THRESHOLD" \
  2>&1 | tee "$OUT/eval_dev_hc_console.log"

python tools/evaluate_prediction_exports.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --train_dataset_name "$DATASET" \
  --exports_dir "$OUT/exports" \
  --output_dir "$OUT/eval_hctest" \
  --image_list "$HC_TEST_LIST" \
  --method "$METHOD" \
  --seed "$SEED" \
  --threshold "$THRESHOLD" \
  2>&1 | tee "$OUT/eval_hctest_console.log"

python tools/analyze_step1_hard_clutter.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --exports_dir "$OUT/exports" \
  --output_dir "$OUT/fp_analysis" \
  --seed "$SEED" \
  --threshold "$THRESHOLD" \
  2>&1 | tee "$OUT/fp_analysis_console.log"

chown -R 1004:1004 "$OUT" || true
