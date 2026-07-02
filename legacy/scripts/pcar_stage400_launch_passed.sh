#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-3ca2917d9c0c}"
RUN_ID="${RUN_ID:-20260623_pcar_stage400}"
STAGE200_RUN_ID="${STAGE200_RUN_ID:-20260623_pcar_stage200}"
ROOT="${ROOT:-/home/AAAI/OHCM-MSHNet}"
BANK_PATH="${BANK_PATH:-$ROOT/results/pcar_ohem/20260623_pcar/bank/persistent_clutter_bank.json}"
DATASET="${DATASET:-NUDT-SIRST}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if ! docker ps --format '{{.ID}} {{.Names}}' | grep -qE "^${CONTAINER}\b| ${CONTAINER}$"; then
  echo "Container is not running: $CONTAINER" >&2
  exit 1
fi

if ! docker exec "$CONTAINER" test -f "$BANK_PATH"; then
  echo "Missing persistent clutter bank in container: $BANK_PATH" >&2
  exit 1
fi

docker exec "$CONTAINER" bash -lc "cd '$ROOT' && python tools/pcar_stage200_gate.py \
  --run_root results/pcar_ohem/$STAGE200_RUN_ID \
  --output_dir results/pcar_ohem/$STAGE200_RUN_ID/gate >/dev/null"

mapfile -t PASSED_METHODS < <(
  docker exec "$CONTAINER" bash -lc "cd '$ROOT' && awk -F, 'NR > 1 && \$NF == \"PASS\" {print \$1}' results/pcar_ohem/$STAGE200_RUN_ID/gate/stage200_gate_table.csv"
)

if [[ "${#PASSED_METHODS[@]}" -eq 0 ]]; then
  echo "No methods passed stage-200 gate; not launching stage-400."
  exit 0
fi

COMMON_ARGS="--tsr_region_start_epoch 80 --tsr_region_end_epoch 120 --tsr_target_scales 5,9 --tsr_topk 3 --tsr_dilate_radius 5 --tsr_margin 0.3 --tsr_rank_temp 0.5 --tsr_target_temp 0.25 --tsr_hard_temp 0.5 --tsr_topq 0.25 --tsr_bank_max_regions 3"

method_args() {
  case "$1" in
    R3-A)
      printf '%s\n' "--tsr_lambda_region 0.02 --tsr_region_loss_mode asym_rank"
      ;;
    R3-B)
      printf '%s\n' "--tsr_lambda_region 0.02 --tsr_region_loss_mode rank --tsr_bank_path $BANK_PATH"
      ;;
    PCAR)
      printf '%s\n' "--tsr_lambda_region 0.02 --tsr_region_loss_mode asym_rank --tsr_bank_path $BANK_PATH"
      ;;
    PCAR-low)
      printf '%s\n' "--tsr_lambda_region 0.01 --tsr_region_loss_mode asym_rank --tsr_bank_path $BANK_PATH"
      ;;
    *)
      echo "Unknown method: $1" >&2
      return 1
      ;;
  esac
}

gpu=0
for method in "${PASSED_METHODS[@]}"; do
  extra_args="$(method_args "$method")"
  echo "Launching stage-400 $method on GPU $gpu"
  docker exec -d \
    -e RUN_ID="$RUN_ID" \
    -e STAGE200_RUN_ID="$STAGE200_RUN_ID" \
    -e METHOD="$method" \
    -e MODEL_NAME="MSHNetOHEM" \
    -e GPU="$gpu" \
    -e DATASET="$DATASET" \
    -e SEED="$SEED" \
    -e EPOCHS="$EPOCHS" \
    -e BATCH_SIZE="$BATCH_SIZE" \
    -e TRAIN_EXTRA_ARGS="$COMMON_ARGS $extra_args" \
    "$CONTAINER" \
    bash -lc "cd '$ROOT' && bash scripts/pcar_stage400_run_one_inner.sh"
  gpu=$(( (gpu + 1) % 4 ))
done

echo "Launched stage-400 for passed methods: ${PASSED_METHODS[*]}"
