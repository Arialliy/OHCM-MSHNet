#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-3ca2917d9c0c}"
RUN_ID="${RUN_ID:-20260623_pcar_stage200}"
ROOT="${ROOT:-/home/AAAI/OHCM-MSHNet}"
BANK_PATH="${BANK_PATH:-$ROOT/results/pcar_ohem/20260623_pcar/bank/persistent_clutter_bank.json}"
DATASET="${DATASET:-NUDT-SIRST}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if ! docker ps --format '{{.ID}} {{.Names}}' | grep -qE "^${CONTAINER}\b| ${CONTAINER}$"; then
  echo "Container is not running: $CONTAINER" >&2
  exit 1
fi

if ! docker exec "$CONTAINER" test -f "$BANK_PATH"; then
  echo "Missing persistent clutter bank in container: $BANK_PATH" >&2
  exit 1
fi

COMMON_ARGS="--tsr_region_start_epoch 80 --tsr_region_end_epoch 120 --tsr_target_scales 5,9 --tsr_topk 3 --tsr_dilate_radius 5 --tsr_margin 0.3 --tsr_rank_temp 0.5 --tsr_target_temp 0.25 --tsr_hard_temp 0.5 --tsr_topq 0.25 --tsr_bank_max_regions 3"

launch_one() {
  local method="$1"
  local gpu="$2"
  local extra_args="$3"
  echo "Launching $method on GPU $gpu"
  docker exec -d \
    -e RUN_ID="$RUN_ID" \
    -e METHOD="$method" \
    -e MODEL_NAME="MSHNetOHEM" \
    -e GPU="$gpu" \
    -e DATASET="$DATASET" \
    -e SEED="$SEED" \
    -e EPOCHS="$EPOCHS" \
    -e BATCH_SIZE="$BATCH_SIZE" \
    -e TRAIN_EXTRA_ARGS="$COMMON_ARGS $extra_args" \
    "$CONTAINER" \
    bash -lc "cd '$ROOT' && bash scripts/pcar_stage200_run_one_inner.sh"
}

launch_one "R3-A" 0 "--tsr_lambda_region 0.02 --tsr_region_loss_mode asym_rank"
launch_one "R3-B" 1 "--tsr_lambda_region 0.02 --tsr_region_loss_mode rank --tsr_bank_path $BANK_PATH"
launch_one "PCAR" 2 "--tsr_lambda_region 0.02 --tsr_region_loss_mode asym_rank --tsr_bank_path $BANK_PATH"
launch_one "PCAR-low" 3 "--tsr_lambda_region 0.01 --tsr_region_loss_mode asym_rank --tsr_bank_path $BANK_PATH"

echo "Launched stage-200 PCAR factor experiment: $RUN_ID"
