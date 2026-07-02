#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
DATASET="${DATASET:-NUDT-SIRST}"

python tools/official/analyze_threshold_matched.py \
  --run_root "${RUN_ROOT}" \
  --dataset "${DATASET}" \
  --seeds "${SEEDS:-42,43,44}" \
  --threshold "${THRESHOLD:-0.5}" \
  --output_dir "${OUTPUT_DIR:-${RUN_ROOT}/threshold_matched}"
