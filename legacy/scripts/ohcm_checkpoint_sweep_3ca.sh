#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260616_checkpoint_sweep}"
ROOT="/home/AAAI/OHCM-MSHNet"
SRC_RUN="$ROOT/results/step5_pre_ohcm_seed_repro/20260615_ohcm_three_seed/OHCM/NUDT-SIRST"
RUN_ROOT="$ROOT/results/step5_pre_ohcm_stability/$RUN_ID"
DATASET="${DATASET:-NUDT-SIRST}"
THRESHOLD="${THRESHOLD:-0.5}"
EPOCHS="${EPOCHS:-50 100 150 200 250 300 350 400}"
HC_LIST="$ROOT/results/step0_baseline/20260611_155232/step2_hcset/hcset_${DATASET}.txt"
mkdir -p "$RUN_ROOT"

COMMON_ARGS=(
  --model_name OHCMMSHNet
  --dataset_dir ./datasets
  --dataset_name "$DATASET"
  --train_dataset_name "$DATASET"
  --threshold "$THRESHOLD"
  --mshnet_warm_epoch 5
  --mshnet_in_channels 1
  --ohcm_warm_epoch 60
  --ohcm_gamma_max 0.3
  --ohcm_gamma_ramp_epochs 60
  --ohcm_tau 0.5
  --ohcm_dilate_radius 5
  --ohcm_topk 3
  --ohcm_margin_m 0.1
  --ohcm_margin_delta 0.5
  --ohcm_mining_mode cc_area_lc_ms
  --lambda_clu 0.2
  --lambda_sup 0.5
  --lambda_margin 0.1
  --lambda_proto 0.0
)

run_seed() {
  local seed="$1"
  local gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$ROOT"
    for epoch in $EPOCHS; do
      local ckpt="$SRC_RUN/seed_${seed}/checkpoints/$DATASET/OHCMMSHNet_${epoch}.pth.tar"
      local out="$RUN_ROOT/OHCM/$DATASET/seed_${seed}/epoch_${epoch}"
      if [[ ! -f "$ckpt" ]]; then
        echo "Missing checkpoint: $ckpt"
        continue
      fi
      mkdir -p "$out"
      echo "Evaluating seed=$seed epoch=$epoch full"
      python tools/evaluate_checkpoint_direct.py \
        "${COMMON_ARGS[@]}" \
        --checkpoint "$ckpt" \
        --output_dir "$out/eval_full" \
        --method OHCM \
        --seed "$seed" \
        2>&1 | tee "$out/eval_full_console.log"

      echo "Evaluating seed=$seed epoch=$epoch hcset"
      python tools/evaluate_checkpoint_direct.py \
        "${COMMON_ARGS[@]}" \
        --checkpoint "$ckpt" \
        --output_dir "$out/eval_hcset" \
        --image_list "$HC_LIST" \
        --method OHCM \
        --seed "$seed" \
        2>&1 | tee "$out/eval_hcset_console.log"
    done
  ) 2>&1 | tee "$RUN_ROOT/seed_${seed}_gpu${gpu}.log"
}

run_seed 0 0 &
pid0=$!
run_seed 1 1 &
pid1=$!
wait "$pid0" "$pid1"
run_seed 2 0

cd "$ROOT"
python tools/summarize_ohcm_checkpoint_sweep.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds 0,1,2 \
  --epochs "$(printf '%s' "$EPOCHS" | tr ' ' ',')" \
  2>&1 | tee "$RUN_ROOT/summarize_console.log"
chown -R 1004:1004 "$RUN_ROOT"
