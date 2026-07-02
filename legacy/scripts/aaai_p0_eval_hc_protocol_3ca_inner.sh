#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260617_aaai_p0_paired}"
DATASET="${DATASET:-NUDT-SIRST}"
ROOT="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/aaai_p0_paired/$RUN_ID"
BASELINE_ROOT="$ROOT/results/step0_baseline/20260611_155232"
STEP3_ROOT="$ROOT/results/step3_ohcm_light_gate/20260613_step3_gate"
HC_VAL_LIST="$RUN_ROOT/hc_protocol/hcval_${DATASET}.txt"
HC_TEST_LIST="$RUN_ROOT/hc_protocol/hctest_${DATASET}.txt"

cd "$ROOT"

eval_one() {
  local method="$1"
  local seed="$2"
  local exports="$3"
  local split="$4"
  local image_list="$5"
  local out="$RUN_ROOT/$method/$DATASET/seed_$seed/eval_$split"
  mkdir -p "$out"
  python tools/evaluate_prediction_exports.py \
    --dataset_dir ./datasets \
    --dataset_name "$DATASET" \
    --train_dataset_name "$DATASET" \
    --exports_dir "$exports" \
    --output_dir "$out" \
    --image_list "$image_list" \
    --method "$method" \
    --seed "$seed" \
    --threshold 0.5
}

for seed in 42 43 44; do
  exports="$BASELINE_ROOT/$DATASET/seed_$seed/exports"
  eval_one MSHNet "$seed" "$exports" hcval "$HC_VAL_LIST"
  eval_one MSHNet "$seed" "$exports" hctest "$HC_TEST_LIST"
done

for method in MSHNetFocal MSHNetOHEM MSHNetTopKNeg OHCM-light; do
  exports="$STEP3_ROOT/$method/$DATASET/seed_42/exports"
  eval_one "$method" 42 "$exports" hcval "$HC_VAL_LIST"
  eval_one "$method" 42 "$exports" hctest "$HC_TEST_LIST"
done

python tools/summarize_aaai_p0_paired.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds 42,43,44

chown -R 1004:1004 "$RUN_ROOT"
