#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
GPU_IRSTD="${GPU_IRSTD:-2}"
GPU_NUDT="${GPU_NUDT:-3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
OUT="$ROOT/repro_runs/mshnet_basicirstd/$RUN_ID"

mkdir -p "$OUT/IRSTD-1K" "$OUT/NUDT-SIRST"

docker run -d \
  --name "ohcm_mshnet_irstd1k_${RUN_ID}" \
  --gpus "\"device=${GPU_IRSTD}\"" \
  -v /home/ly:/home \
  -w /home/AAAI/OHCM-MSHNet \
  "$IMAGE" \
  bash -lc "set -o pipefail; python train.py \
    --model_names MSHNet \
    --dataset_names IRSTD-1K \
    --batchSize 4 \
    --nEpochs 400 \
    --optimizer_name Adagrad \
    --threads 1 \
    --intervals 10 \
    --save /home/AAAI/OHCM-MSHNet/repro_runs/mshnet_basicirstd/${RUN_ID}/IRSTD-1K/log \
    2>&1 | tee /home/AAAI/OHCM-MSHNet/repro_runs/mshnet_basicirstd/${RUN_ID}/IRSTD-1K/console.log"

docker run -d \
  --name "ohcm_mshnet_nudt_${RUN_ID}" \
  --gpus "\"device=${GPU_NUDT}\"" \
  -v /home/ly:/home \
  -w /home/AAAI/OHCM-MSHNet \
  "$IMAGE" \
  bash -lc "set -o pipefail; python train.py \
    --model_names MSHNet \
    --dataset_names NUDT-SIRST \
    --batchSize 4 \
    --nEpochs 400 \
    --optimizer_name Adagrad \
    --threads 1 \
    --intervals 10 \
    --save /home/AAAI/OHCM-MSHNet/repro_runs/mshnet_basicirstd/${RUN_ID}/NUDT-SIRST/log \
    2>&1 | tee /home/AAAI/OHCM-MSHNet/repro_runs/mshnet_basicirstd/${RUN_ID}/NUDT-SIRST/console.log"

printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'IRSTD-1K log: %s/IRSTD-1K/console.log\n' "$OUT"
printf 'NUDT-SIRST log: %s/NUDT-SIRST/console.log\n' "$OUT"
docker ps --filter "name=ohcm_mshnet_" --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}'
