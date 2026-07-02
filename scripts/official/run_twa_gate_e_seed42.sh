#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BASE="${ROOT}/docs/internal/twa/seed42_nudt"

python "${ROOT}/tools/official/check_twa_gate_e_mechanism.py" \
  --gate_d_summary "${BASE}/gate_twa_d_summary.json" \
  --ohem_full "${BASE}/eval_full_ohem/summary_metrics.json" \
  --ohem_hcval "${BASE}/eval_hcval_ohem/summary_metrics.json" \
  --twa4_full "${BASE}/eval_full_twa4_no_bn/summary_metrics.json" \
  --twa4_hcval "${BASE}/eval_hcval_twa4_no_bn/summary_metrics.json" \
  --tce4_hcval "${BASE}/eval_hcval_tce4/summary_metrics.json" \
  --single_late "ep250=${BASE}/eval_hcval_single_ep250/summary_metrics.json" \
  --single_late "ep300=${BASE}/eval_hcval_single_ep300/summary_metrics.json" \
  --single_late "ep350=${BASE}/eval_hcval_single_ep350/summary_metrics.json" \
  --single_late "ep400=${BASE}/eval_hcval_ohem/summary_metrics.json" \
  --twa_variant_hcval "TWA-2=${BASE}/eval_hcval_twa2_no_bn/summary_metrics.json" \
  --twa_variant_hcval "TWA-3=${BASE}/eval_hcval_twa3_no_bn/summary_metrics.json" \
  --twa_variant_hcval "TWA-4=${BASE}/eval_hcval_twa4_no_bn/summary_metrics.json" \
  --twa_variant_full "TWA-2=${BASE}/eval_full_twa2_no_bn/summary_metrics.json" \
  --twa_variant_full "TWA-3=${BASE}/eval_full_twa3_no_bn/summary_metrics.json" \
  --twa_variant_full "TWA-4=${BASE}/eval_full_twa4_no_bn/summary_metrics.json" \
  --output "${BASE}/gate_twa_e_summary.json"
