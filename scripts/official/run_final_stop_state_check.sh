#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

"${PYTHON_BIN}" "${ROOT}/tools/official/check_final_stop_state.py" \
  --root "${ROOT}" \
  --output "${ROOT}/docs/internal/final_stop_state_summary.json"

echo "Gate-FINAL-STOP-CONSISTENCY PASS: repository is in read-only failure archive state."
