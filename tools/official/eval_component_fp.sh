#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CHECKPOINT MODEL_NAME OUTPUT_DIR [IMAGE_LIST]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/eval_fixed.sh" "$@"

summary="${3}/summary_metrics.json"
components="${3}/fp_components.csv"
if [[ ! -f "${summary}" || ! -f "${components}" ]]; then
  echo "Component FP outputs were not produced: ${summary}, ${components}" >&2
  exit 1
fi

python - <<PY
import json
from pathlib import Path
summary = json.loads(Path("${summary}").read_text())
census = summary.get("fp_census_at_threshold")
if not census:
    raise SystemExit("summary_metrics.json is missing fp_census_at_threshold")
print(json.dumps(census, indent=2))
PY
