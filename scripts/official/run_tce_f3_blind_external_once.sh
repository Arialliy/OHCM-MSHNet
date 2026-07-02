#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export ROOT
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
cd "${ROOT}"
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

F3_FAIL_SUMMARY="${ROOT}/docs/internal/tce_final/gate_tce_f3_fail_summary.json"
F3_ONCE_LOCK="${ROOT}/docs/internal/tce_final/gate_tce_f3_once_lock.json"

if [[ -f "${F3_FAIL_SUMMARY}" ]]; then
  "${PYTHON_BIN}" "${ROOT}/tools/official/check_final_stop_state.py" \
    --root "${ROOT}" \
    --output "${ROOT}/docs/internal/final_stop_state_summary.json"
  echo "F3 is already stopped by external Pd regression. Refusing to rerun blind/external once." >&2
  exit 2
fi

if [[ -f "${F3_ONCE_LOCK}" ]]; then
  LOCK_STATUS=$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${F3_ONCE_LOCK}")
try:
    print(json.loads(p.read_text(encoding="utf-8")).get("status", ""))
except Exception:
    print("INVALID_LOCK")
PY
)
  if [[ "${LOCK_STATUS}" == STOPPED* ]]; then
    echo "F3 once-lock is stopped (${LOCK_STATUS}). Refusing to rerun." >&2
    exit 2
  fi
fi

F0="${ROOT}/docs/internal/tce_final/gate_tce_f0_freeze_summary.json"
F1="${ROOT}/docs/internal/tce_final/gate_tce_f1_internal_report.json"
F2="${ROOT}/docs/internal/tce_final/gate_tce_f2_threshold_component_report.json"
PLAN="${ROOT}/docs/internal/tce_final/tce4_frozen_method_plan.json"
MANIFEST="${ROOT}/docs/internal/tce_final/tce4_f3_locked_eval_manifest.json"
LOCK="${ROOT}/docs/internal/tce_final/gate_tce_f3_once_lock.json"
PREFLIGHT="${ROOT}/docs/internal/tce_final/gate_tce_f3_preflight_summary.json"
FINAL="${ROOT}/docs/internal/tce_final/gate_tce_f3_blind_external_report.json"

manifest_get() {
  "${PYTHON_BIN}" - "$MANIFEST" "$@" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cmd = sys.argv[2]
if cmd == "splits":
    print("\n".join(manifest["splits"]))
elif cmd == "dataset":
    print(manifest["split_datasets"][sys.argv[3]])
elif cmd == "dataset_dir":
    print(manifest["dataset_dir"])
elif cmd == "train_dataset_name":
    print(manifest["train_dataset_name"])
elif cmd == "model_name":
    print(manifest["model_name"])
elif cmd == "summary":
    split, seed, method = sys.argv[3], sys.argv[4], sys.argv[5]
    print(manifest["summary_paths"][split][seed][method])
elif cmd == "checkpoint":
    seed, epoch = sys.argv[3], sys.argv[4]
    print(manifest["checkpoint_paths"][seed][epoch])
elif cmd == "tce_checkpoints":
    seed = sys.argv[3]
    epochs = [str(item) for item in manifest["tce_epochs"]]
    print(",".join(manifest["checkpoint_paths"][seed][epoch] for epoch in epochs))
else:
    raise SystemExit(f"unknown manifest command: {cmd}")
PY
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT" "$1" ;;
  esac
}

"${PYTHON_BIN}" tools/official/check_tce_f3_preflight.py \
  --f0_summary "${F0}" \
  --f1_summary "${F1}" \
  --f2_summary "${F2}" \
  --frozen_method_plan "${PLAN}" \
  --f3_manifest "${MANIFEST}" \
  --once_lock "${LOCK}" \
  --final_report "${FINAL}" \
  --output "${PREFLIGHT}"

DATASET_DIR=$(manifest_get dataset_dir)
TRAIN_DATASET_NAME=$(manifest_get train_dataset_name)
MODEL_NAME=$(manifest_get model_name)
mapfile -t SPLITS < <(manifest_get splits)

for SPLIT in "${SPLITS[@]}"; do
  DATASET_NAME=$(manifest_get dataset "$SPLIT")
  for SEED in 42 43 44; do
    OHEM_SUMMARY=$(abs_path "$(manifest_get summary "$SPLIT" "$SEED" ohem)")
    TCE4_SUMMARY=$(abs_path "$(manifest_get summary "$SPLIT" "$SEED" tce4)")
    OHEM_OUT=$(dirname "$OHEM_SUMMARY")
    TCE4_OUT=$(dirname "$TCE4_SUMMARY")
    CKPT400=$(manifest_get checkpoint "$SEED" 400)
    TCE_CHECKPOINTS=$(manifest_get tce_checkpoints "$SEED")

    if [[ -f "$OHEM_SUMMARY" ]]; then
      echo "[F3] Skip existing OHEM summary seed=${SEED} split=${SPLIT}: ${OHEM_SUMMARY}"
    else
      echo "[F3] Evaluating OHEM-400 seed=${SEED} split=${SPLIT} dataset=${DATASET_NAME}"
      "${PYTHON_BIN}" tools/official/evaluate_checkpoint_direct.py \
        --dataset_dir "${DATASET_DIR}" \
        --dataset_name "${DATASET_NAME}" \
        --train_dataset_name "${TRAIN_DATASET_NAME}" \
        --model_name "${MODEL_NAME}" \
        --mshnet_export_head final \
        --checkpoint "${CKPT400}" \
        --output_dir "${OHEM_OUT}" \
        --method "OHEM400_F3_${SPLIT}_seed${SEED}" \
        --seed "${SEED}" \
        --threshold 0.5 \
        --thresholds "0.5"
    fi

    if [[ -f "$TCE4_SUMMARY" ]]; then
      echo "[F3] Skip existing TCE-4 summary seed=${SEED} split=${SPLIT}: ${TCE4_SUMMARY}"
    else
      echo "[F3] Evaluating TCE-4 seed=${SEED} split=${SPLIT} dataset=${DATASET_NAME}"
      "${PYTHON_BIN}" tools/legacy/pcar/pcar_checkpoint_ensemble_eval.py \
        --dataset_dir "${DATASET_DIR}" \
        --dataset_name "${DATASET_NAME}" \
        --train_dataset_name "${TRAIN_DATASET_NAME}" \
        --model_name "${MODEL_NAME}" \
        --checkpoints "${TCE_CHECKPOINTS}" \
        --aggregation mean \
        --threshold 0.5 \
        --thresholds "0.5" \
        --output_dir "${TCE4_OUT}" \
        --method "TCE4_OHEM_F3_${SPLIT}_seed${SEED}" \
        --seed "${SEED}" \
        --device cuda
    fi
  done
done

"${PYTHON_BIN}" tools/official/check_tce_f3_blind_external_report.py \
  --manifest "${MANIFEST}" \
  --once_lock "${LOCK}" \
  --output "${FINAL}"
