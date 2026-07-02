#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-3ca2917d9c0c}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_aaai_p0_paired)}"
ROOT_HOST="/home/ly/AAAI/OHCM-MSHNet"
ROOT_CONTAINER="/home/AAAI/OHCM-MSHNet"
RUN_ROOT_HOST="$ROOT_HOST/results/aaai_p0_paired/$RUN_ID"
RUN_ROOT_CONTAINER="$ROOT_CONTAINER/results/aaai_p0_paired/$RUN_ID"

mkdir -p "$RUN_ROOT_HOST"

docker exec "$CONTAINER" bash -lc "mkdir -p '$RUN_ROOT_CONTAINER'"
docker exec "$CONTAINER" bash -lc "cat > '$RUN_ROOT_CONTAINER/manifest.txt' <<'EOF'
RUN_ID=$RUN_ID
container=$CONTAINER
dataset=NUDT-SIRST
seeds=42,43,44
queue seed43: GPU2, methods MSHNetFocal/MSHNetOHEM/MSHNetTopKNeg/OHCM-light seed43 plus OHCM-late-inhibition seed42/44
queue seed44: GPU3, methods MSHNetFocal/MSHNetOHEM/MSHNetTopKNeg/OHCM-light seed44 plus OHCM-late-inhibition seed43
baseline MSHNet seed42/43/44 comes from results/step0_baseline/20260611_155232
seed42 Focal/OHEM/TopK/OHCM-light comes from results/step3_ohcm_light_gate/20260613_step3_gate
EOF"

docker exec -d "$CONTAINER" bash -lc "cd '$ROOT_CONTAINER' && RUN_ID='$RUN_ID' QUEUE=seed43 GPU=2 bash scripts/aaai_p0_queue_3ca_inner.sh > '$RUN_ROOT_CONTAINER/queue_seed43_gpu2.log' 2>&1"
docker exec -d "$CONTAINER" bash -lc "cd '$ROOT_CONTAINER' && RUN_ID='$RUN_ID' QUEUE=seed44 GPU=3 bash scripts/aaai_p0_queue_3ca_inner.sh > '$RUN_ROOT_CONTAINER/queue_seed44_gpu3.log' 2>&1"

printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT_HOST"
printf 'GPU2 queue log=%s\n' "$RUN_ROOT_HOST/queue_seed43_gpu2.log"
printf 'GPU3 queue log=%s\n' "$RUN_ROOT_HOST/queue_seed44_gpu3.log"
