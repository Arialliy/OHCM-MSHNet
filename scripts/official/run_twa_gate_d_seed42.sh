#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OHEM_SUMMARY=${OHEM_SUMMARY:-"${ROOT}/docs/internal/twa/seed42_nudt/eval_hcval_ohem/summary_metrics.json"}
if [[ ! -f "${OHEM_SUMMARY}" ]]; then
  echo "Missing paired OHEM HC-Val summary: ${OHEM_SUMMARY}" >&2
  exit 1
fi

bash scripts/official/eval_twa_seed42_hcval.sh

python tools/official/check_twa_gate_d_hcval.py \
  --ohem_summary "${OHEM_SUMMARY}" \
  --twa_summary "${ROOT}/docs/internal/twa/seed42_nudt/eval_hcval_twa_no_bn/summary_metrics.json" \
  --output "${ROOT}/docs/internal/twa/seed42_nudt/gate_twa_d_summary.json"
