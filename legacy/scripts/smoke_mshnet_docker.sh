#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
GPU="${GPU:-2}"
ROOT="${ROOT:-/home/ly/AAAI/OHCM-MSHNet}"

docker run --rm \
  --name ohcm_mshnet_smoke \
  --gpus "\"device=${GPU}\"" \
  -v /home/ly:/home \
  -w /home/AAAI/OHCM-MSHNet \
  "$IMAGE" \
  bash -lc "python train.py \
    --model_names MSHNet \
    --dataset_names NUAA-SIRST \
    --batchSize 4 \
    --nEpochs 1 \
    --optimizer_name Adagrad \
    --threads 1 \
    --intervals 1 \
    --save /home/AAAI/OHCM-MSHNet/repro_smoke/log"
