#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260616_checkpoint_sweep_hc_first}"
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

eval_hc_seed() {
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
      echo "Evaluating HC seed=$seed epoch=$epoch"
      python tools/evaluate_checkpoint_direct.py \
        "${COMMON_ARGS[@]}" \
        --checkpoint "$ckpt" \
        --output_dir "$out/eval_hcset" \
        --image_list "$HC_LIST" \
        --method OHCM \
        --seed "$seed" \
        2>&1 | tee "$out/eval_hcset_console.log"
    done
  ) 2>&1 | tee "$RUN_ROOT/seed_${seed}_hc_gpu${gpu}.log"
}

eval_full_one() {
  local seed="$1"
  local epoch="$2"
  local gpu="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$ROOT"
    local ckpt="$SRC_RUN/seed_${seed}/checkpoints/$DATASET/OHCMMSHNet_${epoch}.pth.tar"
    local out="$RUN_ROOT/OHCM/$DATASET/seed_${seed}/epoch_${epoch}"
    echo "Evaluating Full seed=$seed epoch=$epoch"
    python tools/evaluate_checkpoint_direct.py \
      "${COMMON_ARGS[@]}" \
      --checkpoint "$ckpt" \
      --output_dir "$out/eval_full" \
      --method OHCM \
      --seed "$seed" \
      2>&1 | tee "$out/eval_full_console.log"
  ) 2>&1 | tee "$RUN_ROOT/seed_${seed}_full_epoch_${epoch}_gpu${gpu}.log"
}

eval_hc_seed 0 0 &
pid0=$!
eval_hc_seed 1 1 &
pid1=$!
wait "$pid0" "$pid1"
eval_hc_seed 2 0

cd "$ROOT"
python tools/summarize_ohcm_checkpoint_sweep.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds 0,1,2 \
  --epochs "$(printf '%s' "$EPOCHS" | tr ' ' ',')" \
  2>&1 | tee "$RUN_ROOT/summarize_hc_only_console.log"

python - "$RUN_ROOT/ohcm_checkpoint_sweep_best_by_hc.csv" "$RUN_ROOT/best_epochs.tsv" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = []
with src.open(newline="") as f:
    for row in csv.DictReader(f):
        if row["split"] == "hcset":
            rows.append((int(row["seed"]), int(row["epoch"])))
rows = sorted(set(rows))
dst.write_text("\n".join(f"{seed}\t{epoch}" for seed, epoch in rows) + "\n", encoding="utf-8")
PY

while IFS=$'\t' read -r seed epoch; do
  [[ -z "$seed" ]] && continue
  gpu=0
  if [[ "$seed" == "1" ]]; then
    gpu=1
  fi
  eval_full_one "$seed" "$epoch" "$gpu"
done < "$RUN_ROOT/best_epochs.tsv"

python tools/summarize_ohcm_checkpoint_sweep.py \
  --run_root "$RUN_ROOT" \
  --dataset "$DATASET" \
  --seeds 0,1,2 \
  --epochs "$(printf '%s' "$EPOCHS" | tr ' ' ',')" \
  2>&1 | tee "$RUN_ROOT/summarize_final_console.log"
chown -R 1004:1004 "$RUN_ROOT"
