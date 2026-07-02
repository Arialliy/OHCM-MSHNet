#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BASE="${ROOT}/docs/internal/twa/seed42_nudt"

python "${ROOT}/tools/official/check_late_snapshot_gate_a_seed42.py" \
  --gate_e_summary "${BASE}/gate_twa_e_summary.json" \
  --ohem_full "${BASE}/eval_full_ohem/summary_metrics.json" \
  --ohem_hcval "${BASE}/eval_hcval_ohem/summary_metrics.json" \
  --snapshot_full "${BASE}/eval_full_single_ep250/summary_metrics.json" \
  --snapshot_hcval "${BASE}/eval_hcval_single_ep250/summary_metrics.json" \
  --twa4_hcval "${BASE}/eval_hcval_twa4_no_bn/summary_metrics.json" \
  --snapshot_name "ep250" \
  --output "${BASE}/gate_late_snapshot_ep250_a_summary.json"
