#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/AAAI/OHCM-MSHNet}"
DATASET="${DATASET:-NUDT-SIRST}"
RUN_ID="${RUN_ID:-20260622_tsr_ohem}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
MAX_IMAGES="${MAX_IMAGES:-0}"
VIS_COUNT="${VIS_COUNT:-200}"
RUN_ROOT="$ROOT/results/tsr_ohem/$RUN_ID"
STATS_JSON="$RUN_ROOT/stage0_target_scales/${DATASET}_target_scale_stats.json"
PARITY_JSON="$RUN_ROOT/stage0_parity/${DATASET}_seed${SEED}_parity.json"
MINER_DIR="$RUN_ROOT/stage1_miner/${DATASET}_seed${SEED}_train"
CHECKPOINT="${CHECKPOINT:-$ROOT/results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/$DATASET/seed_${SEED}/checkpoints/$DATASET/MSHNetOHEM_400.pth.tar}"

cd "$ROOT"
mkdir -p "$RUN_ROOT"

{
  echo "RUN_ID=$RUN_ID"
  echo "dataset=$DATASET"
  echo "seed=$SEED"
  echo "gpus=$GPUS"
  echo "checkpoint=$CHECKPOINT"
  echo "max_images=$MAX_IMAGES"
  echo "vis_count=$VIS_COUNT"
} > "$RUN_ROOT/manifest_stage0_stage1.txt"

CUDA_VISIBLE_DEVICES="$GPUS" python tools/tsr_target_scale_stats.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --output_json "$STATS_JSON" \
  2>&1 | tee "$RUN_ROOT/stage0_target_scale_stats.log"

SCALES="$(python -c "import json; print(json.load(open('$STATS_JSON'))['target_scales_arg'])")"
DILATE="$(python -c "import json; print(json.load(open('$STATS_JSON'))['safe_background_dilation_radius'])")"
echo "target_scales=$SCALES" | tee -a "$RUN_ROOT/manifest_stage0_stage1.txt"
echo "safe_dilation_radius=$DILATE" | tee -a "$RUN_ROOT/manifest_stage0_stage1.txt"

CUDA_VISIBLE_DEVICES="$GPUS" python tools/tsr_parity_check.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --patchSize 256 \
  --batchSize 4 \
  --threads 0 \
  --seed "$SEED" \
  --epoch 120 \
  --target_scales "$SCALES" \
  --output_json "$PARITY_JSON" \
  2>&1 | tee "$RUN_ROOT/stage0_parity.log"

CUDA_VISIBLE_DEVICES="$GPUS" python tools/tsr_miner_diagnose.py \
  --dataset_dir ./datasets \
  --dataset_name "$DATASET" \
  --split train \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$MINER_DIR" \
  --target_scales "$SCALES" \
  --topk 3 \
  --beta 0.5 \
  --nms_iou 0.3 \
  --weight_temp 0.2 \
  --dilate_radius "$DILATE" \
  --max_images "$MAX_IMAGES" \
  --vis_count "$VIS_COUNT" \
  --seed "$SEED" \
  2>&1 | tee "$RUN_ROOT/stage1_miner.log"

chown -R 1004:1004 "$RUN_ROOT" || true
