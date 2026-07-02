#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260615_ohcm_three_seed}"
ROOT="/home/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/step5_pre_ohcm_seed_repro/$RUN_ID"
DATASET="${DATASET:-NUDT-SIRST}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-4}"
THRESHOLD="${THRESHOLD:-0.5}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1}"
HC_LIST="$ROOT/results/step0_baseline/20260611_155232/step2_hcset/hcset_${DATASET}.txt"
mkdir -p "$RUN_ROOT"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local out="$RUN_ROOT/OHCM/$DATASET/seed_$seed"
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$ROOT"
    python train.py \
      --model_names OHCMMSHNet \
      --dataset_names "$DATASET" \
      --batchSize "$BATCH_SIZE" \
      --patchSize 256 \
      --nEpochs "$EPOCHS" \
      --optimizer_name Adagrad \
      --threads 1 \
      --intervals 10 \
      --seed "$seed" \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      --save "$out/checkpoints" \
      --ohcm_warm_epoch 60 \
      --ohcm_tau 0.5 \
      --ohcm_dilate_radius 5 \
      --ohcm_topk 3 \
      --ohcm_gamma_max 0.3 \
      --ohcm_gamma_ramp_epochs 60 \
      --ohcm_margin_m 0.1 \
      --ohcm_margin_delta 0.5 \
      --ohcm_mining_mode cc_area_lc_ms \
      --lambda_clu 0.2 \
      --lambda_sup 0.5 \
      --lambda_margin 0.1 \
      --lambda_proto 0.0 \
      2>&1 | tee -a "$out/train_console.log"

    python tools/export_step0_predictions.py \
      --model_name OHCMMSHNet \
      --dataset_dir ./datasets \
      --dataset_name "$DATASET" \
      --train_dataset_name "$DATASET" \
      --checkpoint "$out/checkpoints/$DATASET/OHCMMSHNet_${EPOCHS}.pth.tar" \
      --output_dir "$out/exports" \
      --threshold "$THRESHOLD" \
      --seed "$seed" \
      --mshnet_warm_epoch 5 \
      --mshnet_in_channels 1 \
      --ohcm_warm_epoch 60 \
      --ohcm_gamma_max 0.3 \
      --ohcm_gamma_ramp_epochs 60 \
      --ohcm_tau 0.5 \
      --ohcm_dilate_radius 5 \
      --ohcm_topk 3 \
      --ohcm_margin_m 0.1 \
      --ohcm_margin_delta 0.5 \
      --ohcm_mining_mode cc_area_lc_ms \
      --lambda_clu 0.2 \
      --lambda_sup 0.5 \
      --lambda_margin 0.1 \
      --lambda_proto 0.0 \
      2>&1 | tee "$out/export_console.log"

    python tools/evaluate_prediction_exports.py \
      --dataset_dir ./datasets \
      --dataset_name "$DATASET" \
      --train_dataset_name "$DATASET" \
      --exports_dir "$out/exports" \
      --output_dir "$out/eval_full" \
      --method OHCM \
      --seed "$seed" \
      --threshold "$THRESHOLD" \
      2>&1 | tee "$out/eval_full_console.log"

    python tools/evaluate_prediction_exports.py \
      --dataset_dir ./datasets \
      --dataset_name "$DATASET" \
      --train_dataset_name "$DATASET" \
      --exports_dir "$out/exports" \
      --output_dir "$out/eval_hcset" \
      --image_list "$HC_LIST" \
      --method OHCM \
      --seed "$seed" \
      --threshold "$THRESHOLD" \
      2>&1 | tee "$out/eval_hcset_console.log"

    python tools/analyze_step1_hard_clutter.py \
      --dataset_dir ./datasets \
      --dataset_name "$DATASET" \
      --exports_dir "$out/exports" \
      --output_dir "$out/fp_analysis" \
      --seed "$seed" \
      --threshold "$THRESHOLD" \
      2>&1 | tee "$out/fp_analysis_console.log"
  ) 2>&1 | tee "$RUN_ROOT/seed_${seed}_gpu${gpu}.log"
}

pids=()
idx=0
read -r -a gpu_list <<< "$GPUS"
gpu_count="${#gpu_list[@]}"
for seed in $SEEDS; do
  gpu="${gpu_list[$idx]}"
  run_seed "$seed" "$gpu" &
  pids+=("$!")
  idx=$((idx + 1))
  if [[ "$idx" -ge "$gpu_count" ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    pids=()
    idx=0
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

cd "$ROOT"
python tools/summarize_ohcm_seed_repro.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds "$(printf '%s' "$SEEDS" | tr ' ' ',')" \
  2>&1 | tee "$RUN_ROOT/summarize_console.log"
chown -R 1004:1004 "$RUN_ROOT"
