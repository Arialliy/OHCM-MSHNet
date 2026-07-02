#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-3ca2917d9c0c}"
ROOT_HOST="${ROOT_HOST:-/home/ly/AAAI/OHCM-MSHNet}"
ROOT_CONTAINER="${ROOT_CONTAINER:-/home/AAAI/OHCM-MSHNet}"
RUN_ID="${RUN_ID:-20260622_tsr_ohem}"
MODE="${MODE:-stage0_stage1}"
LOG_DIR="$ROOT_HOST/results/tsr_ohem/$RUN_ID"

mkdir -p "$LOG_DIR"

if [[ "$MODE" == "stage0_stage1" ]]; then
  docker exec -d "$CONTAINER" bash -lc \
    "cd '$ROOT_CONTAINER' && RUN_ID='$RUN_ID' GPUS='${GPUS:-0,1,2,3}' SEED='${SEED:-42}' MAX_IMAGES='${MAX_IMAGES:-0}' VIS_COUNT='${VIS_COUNT:-200}' bash scripts/tsr_stage0_stage1_3ca_inner.sh > '$ROOT_CONTAINER/results/tsr_ohem/$RUN_ID/stage0_stage1_outer.log' 2>&1"
  echo "launched stage0_stage1"
  echo "log=$LOG_DIR/stage0_stage1_outer.log"
elif [[ "$MODE" == "stage2" ]]; then
  RUN_ROOT="$ROOT_HOST/results/aaai_p0_paired/${STAGE2_RUN_ID:-${RUN_ID}_stage2}"
  mkdir -p "$RUN_ROOT"
  docker exec -d "$CONTAINER" bash -lc \
    "cd '$ROOT_CONTAINER' && RUN_ID='${STAGE2_RUN_ID:-${RUN_ID}_stage2}' STAGE01_RUN_ID='$RUN_ID' GPUS='${GPUS:-0,1,2,3}' SEED='${SEED:-42}' EPOCHS='${EPOCHS:-400}' BATCH_SIZE='${BATCH_SIZE:-16}' LAMBDA_REGION='${LAMBDA_REGION:-0.05}' bash scripts/tsr_stage2_single_seed_3ca_inner.sh > '$ROOT_CONTAINER/results/aaai_p0_paired/${STAGE2_RUN_ID:-${RUN_ID}_stage2}/stage2_outer.log' 2>&1"
  echo "launched stage2"
  echo "log=$RUN_ROOT/stage2_outer.log"
else
  echo "Unknown MODE=$MODE; use MODE=stage0_stage1 or MODE=stage2" >&2
  exit 2
fi
