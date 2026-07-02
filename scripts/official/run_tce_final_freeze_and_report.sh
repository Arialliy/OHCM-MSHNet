#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

python "${ROOT}/tools/official/check_tce_final_freeze.py" \
  --tcsr_gate_a_summary "${ROOT}/docs/internal/tcsr/seed42_nudt/gate_tcsr_a_bank_summary.json" \
  --tce_frozen_plan "${ROOT}/docs/internal/tce_final/tce4_frozen_method_plan.json" \
  --output "${ROOT}/docs/internal/tce_final/gate_tce_f0_freeze_summary.json"

python "${ROOT}/tools/official/aggregate_tce_final_report.py" \
  --manifest "${ROOT}/docs/internal/tce_final/tce4_internal_manifest.json" \
  --output "${ROOT}/docs/internal/tce_final/gate_tce_f1_internal_report.json"
