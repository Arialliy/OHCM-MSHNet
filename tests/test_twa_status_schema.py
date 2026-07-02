import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.branch_status import STOPPED_TWA_VARIANTS


def test_gate_twa_abc_has_next_allowed_gate():
    summary_path = Path("docs/internal/twa/seed42_nudt/gate_twa_abc_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["next_allowed_gate"] == "Gate-TWA-D-HCVal-seed42"
    assert summary["current_candidate"] == "twa_without_bn"
    assert summary["twa_bn_status"] == "STOP_BN_RECALIBRATION"


def test_twa_bn_recalibration_is_stopped():
    assert "twa_bn_recalibrated" in STOPPED_TWA_VARIANTS
    assert "Gate-TWA-B" in STOPPED_TWA_VARIANTS["twa_bn_recalibrated"]
