#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
IMAGE="${IMAGE:-rccn-5090-pytorch270-cu128}"
THRESHOLD="${THRESHOLD:-0.5}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
ROOT="/home/ly/AAAI/OHCM-MSHNet"
RUN_ROOT="$ROOT/results/step0_baseline/$RUN_ID"

EXPECTED=(
  "IRSTD-1K:42"
  "NUDT-SIRST:42"
  "NUAA-SIRST:42"
  "NUDT-SIRST:43"
  "NUDT-SIRST:44"
)

mkdir -p "$RUN_ROOT"

printf '[%s] Step1 watcher started for RUN_ID=%s\n' "$(date '+%F %T')" "$RUN_ID"

while true; do
  missing=()
  for item in "${EXPECTED[@]}"; do
    dataset="${item%%:*}"
    seed="${item##*:}"
    summary="$RUN_ROOT/$dataset/seed_$seed/exports/summary_metrics.json"
    if [[ ! -f "$summary" ]]; then
      missing+=("$dataset seed_$seed")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    break
  fi

  printf '[%s] Waiting for Step0 exports: %s\n' "$(date '+%F %T')" "${missing[*]}"
  sleep "$CHECK_INTERVAL"
done

printf '[%s] Step0 exports complete. Launching Step1 diagnosis.\n' "$(date '+%F %T')"

for item in "${EXPECTED[@]}"; do
  dataset="${item%%:*}"
  seed="${item##*:}"
  printf '[%s] Step1 start: %s seed_%s\n' "$(date '+%F %T')" "$dataset" "$seed"
  IMAGE="$IMAGE" RUN_ID="$RUN_ID" DATASET="$dataset" SEED="$seed" THRESHOLD="$THRESHOLD" \
    "$ROOT/scripts/step1_run_one_docker.sh"
  printf '[%s] Step1 done: %s seed_%s\n' "$(date '+%F %T')" "$dataset" "$seed"
done

python3 "$ROOT/tools/summarize_step1.py" --run_root "$RUN_ROOT"
printf '[%s] Step1 all done.\n' "$(date '+%F %T')"
