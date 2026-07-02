import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def valid_plan():
    return {
        "decision": "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE",
        "method": {
            "name": "TCE-4-OHEM",
            "base": "MSHNetOHEM",
            "checkpoints": [250, 300, 350, 400],
            "aggregation": "existing_official_tce_aggregation",
            "threshold": 0.5,
            "training": "no_new_training",
            "inference_forward_count": 4,
        },
        "forbidden": [
            "seed_search",
            "checkpoint_search",
            "threshold_search",
            "BN_recalibration_tuning",
            "TCSR_training",
            "new_loss",
            "new_model_structure",
        ],
    }


def run_checker(tcsr, plan, out):
    tool = PROJECT_ROOT / "tools" / "official" / "check_tce_final_freeze.py"
    return subprocess.run(
        [
            sys.executable,
            str(tool),
            "--tcsr_gate_a_summary",
            str(tcsr),
            "--tce_frozen_plan",
            str(plan),
            "--output",
            str(out),
        ],
        cwd=str(PROJECT_ROOT),
    )


def test_tce_freeze_passes_when_tcsr_stopped_and_plan_fixed(tmp_path):
    tcsr = tmp_path / "tcsr.json"
    plan = tmp_path / "plan.json"
    out = tmp_path / "out.json"

    write_json(tcsr, {"gate_pass": False, "decision": "STOP_TCSR_AT_BANK_AUDIT"})
    write_json(plan, valid_plan())

    proc = run_checker(tcsr, plan, out)

    assert proc.returncode == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["gate_pass"] is True
    assert result["next_allowed_gate"] == "Gate-TCE-F1-internal-evidence-aggregation"


def test_tce_freeze_fails_if_checkpoint_list_changes(tmp_path):
    tcsr = tmp_path / "tcsr.json"
    plan = tmp_path / "plan.json"
    out = tmp_path / "out.json"
    broken = valid_plan()
    broken["method"]["checkpoints"] = [300, 350, 400]
    broken["method"]["inference_forward_count"] = 3

    write_json(tcsr, {"gate_pass": False, "decision": "STOP_TCSR_AT_BANK_AUDIT"})
    write_json(plan, broken)

    proc = run_checker(tcsr, plan, out)

    assert proc.returncode != 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["gate_pass"] is False
    assert "checkpoints must be [250,300,350,400]" in result["failures"]


def test_tce_freeze_fails_if_tcsr_not_stopped(tmp_path):
    tcsr = tmp_path / "tcsr.json"
    plan = tmp_path / "plan.json"
    out = tmp_path / "out.json"

    write_json(tcsr, {"gate_pass": True, "decision": "PASS_TCSR_BANK_AUDIT"})
    write_json(plan, valid_plan())

    proc = run_checker(tcsr, plan, out)

    assert proc.returncode != 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert "TCSR Gate-A must be failed/stopped" in result["failures"]
