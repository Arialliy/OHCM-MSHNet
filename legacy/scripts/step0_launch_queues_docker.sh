#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/step0_baseline/$RUN_ID"

mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/manifest.txt" <<EOF
RUN_ID=$RUN_ID
IMAGE=$IMAGE
GPU_A=$GPU_A
GPU_B=$GPU_B
plan:
  gpu_a:
    - IRSTD-1K seed 42
    - NUDT-SIRST seed 43
  gpu_b:
    - NUDT-SIRST seed 42
    - NUAA-SIRST seed 42
    - NUDT-SIRST seed 44
EOF

setsid bash -lc "
  set -euo pipefail
  IMAGE='$IMAGE' RUN_ID='$RUN_ID' GPU='$GPU_A' DATASET='IRSTD-1K' SEED=42 '$ROOT/scripts/step0_run_one_docker.sh'
  IMAGE='$IMAGE' RUN_ID='$RUN_ID' GPU='$GPU_A' DATASET='NUDT-SIRST' SEED=43 '$ROOT/scripts/step0_run_one_docker.sh'
  python3 '$ROOT/tools/summarize_step0.py' --run_root '$RUN_ROOT' || true
" > "$RUN_ROOT/queue_gpu${GPU_A}.log" 2>&1 < /dev/null &
PID_A=$!

setsid bash -lc "
  set -euo pipefail
  IMAGE='$IMAGE' RUN_ID='$RUN_ID' GPU='$GPU_B' DATASET='NUDT-SIRST' SEED=42 '$ROOT/scripts/step0_run_one_docker.sh'
  IMAGE='$IMAGE' RUN_ID='$RUN_ID' GPU='$GPU_B' DATASET='NUAA-SIRST' SEED=42 '$ROOT/scripts/step0_run_one_docker.sh'
  IMAGE='$IMAGE' RUN_ID='$RUN_ID' GPU='$GPU_B' DATASET='NUDT-SIRST' SEED=44 '$ROOT/scripts/step0_run_one_docker.sh'
  python3 '$ROOT/tools/summarize_step0.py' --run_root '$RUN_ROOT' || true
" > "$RUN_ROOT/queue_gpu${GPU_B}.log" 2>&1 < /dev/null &
PID_B=$!

printf '%s\n' "$PID_A" > "$RUN_ROOT/queue_gpu${GPU_A}.pid"
printf '%s\n' "$PID_B" > "$RUN_ROOT/queue_gpu${GPU_B}.pid"

printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'GPU%s queue PID=%s log=%s\n' "$GPU_A" "$PID_A" "$RUN_ROOT/queue_gpu${GPU_A}.log"
printf 'GPU%s queue PID=%s log=%s\n' "$GPU_B" "$PID_B" "$RUN_ROOT/queue_gpu${GPU_B}.log"
