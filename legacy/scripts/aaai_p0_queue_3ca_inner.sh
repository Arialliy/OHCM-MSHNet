#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
QUEUE="${QUEUE:?QUEUE is required}"
GPU="${GPU:?GPU is required}"
DATASET="${DATASET:-NUDT-SIRST}"
ROOT="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/aaai_p0_paired/$RUN_ID"
mkdir -p "$RUN_ROOT"
cd "$ROOT"

run_one() {
  local method="$1"
  local model="$2"
  local seed="$3"
  local train_extra="${4:-}"
  local export_extra="${5:-}"
  echo "[$(date '+%F %T')] START method=$method model=$model seed=$seed gpu=$GPU"
  RUN_ID="$RUN_ID" \
  METHOD="$method" \
  MODEL_NAME="$model" \
  DATASET="$DATASET" \
  SEED="$seed" \
  GPU="$GPU" \
  TRAIN_EXTRA_ARGS="$train_extra" \
  EXPORT_EXTRA_ARGS="$export_extra" \
  bash scripts/aaai_p0_run_one_3ca_inner.sh
  echo "[$(date '+%F %T')] DONE method=$method seed=$seed"
}

OHCM_ARGS="--ohcm_warm_epoch 60 --ohcm_tau 0.5 --ohcm_dilate_radius 5 --ohcm_topk 3 --ohcm_gamma_max 0.3 --ohcm_gamma_ramp_epochs 60 --ohcm_margin_m 0.1 --ohcm_margin_delta 0.5 --ohcm_mining_mode cc_area_lc_ms --lambda_clu 0.2 --lambda_sup 0.5 --lambda_margin 0.1 --lambda_proto 0.0"
OHCM_LATE_ARGS="$OHCM_ARGS --ohcm_inhibition_start_epoch 120"

case "$QUEUE" in
  seed43)
    run_one MSHNetFocal MSHNetFocal 43
    run_one MSHNetOHEM MSHNetOHEM 43
    run_one MSHNetTopKNeg MSHNetTopKNeg 43
    run_one OHCM-light OHCMMSHNet 43 "$OHCM_ARGS" "$OHCM_ARGS"
    run_one OHCM-late-inhibition OHCMMSHNet 42 "$OHCM_LATE_ARGS" "$OHCM_LATE_ARGS"
    run_one OHCM-late-inhibition OHCMMSHNet 44 "$OHCM_LATE_ARGS" "$OHCM_LATE_ARGS"
    ;;
  seed44)
    run_one MSHNetFocal MSHNetFocal 44
    run_one MSHNetOHEM MSHNetOHEM 44
    run_one MSHNetTopKNeg MSHNetTopKNeg 44
    run_one OHCM-light OHCMMSHNet 44 "$OHCM_ARGS" "$OHCM_ARGS"
    run_one OHCM-late-inhibition OHCMMSHNet 43 "$OHCM_LATE_ARGS" "$OHCM_LATE_ARGS"
    ;;
  *)
    echo "Unknown QUEUE=$QUEUE" >&2
    exit 2
    ;;
esac

python tools/summarize_aaai_p0_paired.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds 42,43,44 \
  2>&1 | tee "$RUN_ROOT/summarize_${QUEUE}.log" || true

chown -R 1004:1004 "$RUN_ROOT"
