import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_bcv_gate_b import add_split_decision


def test_bcv_gate_b_split_decision_allows_fp_residual_audit_after_candidate_failure():
    summary = {
        "target_residual_bg_ratio_mean": 5.7653,
        "candidate_to_budget_ratio_mean": 1.77e-5,
        "residual_auroc_target_vs_far_mean": 0.9355,
        "checks": {
            "background_reconstruction_error": True,
            "target_residual_bg_ratio": True,
            "candidate_to_budget_ratio": False,
            "target_leakage": True,
            "flat_candidate_ratio": True,
            "residual_auroc": True,
        },
    }
    args = SimpleNamespace(
        min_residual_auroc=0.65,
        min_target_residual_bg_ratio=1.5,
        min_candidate_to_budget_ratio=1.0,
    )

    add_split_decision(summary, args)

    assert summary["background_residual_gate_pass"] is True
    assert summary["candidate_mining_gate_pass"] is False
    assert summary["legacy_all_checks_pass"] is False
    assert summary["overall_decision"] == "PROCEED_TO_FP_RESIDUAL_AUDIT"
    assert summary["gate_pass"] is True


def test_bcv_gate_b_split_decision_stops_when_residual_signal_fails():
    summary = {
        "target_residual_bg_ratio_mean": 1.0,
        "candidate_to_budget_ratio_mean": 2.0,
        "residual_auroc_target_vs_far_mean": 0.55,
        "checks": {
            "background_reconstruction_error": True,
            "target_residual_bg_ratio": False,
            "candidate_to_budget_ratio": True,
            "target_leakage": True,
            "flat_candidate_ratio": True,
            "residual_auroc": False,
        },
    }
    args = SimpleNamespace(
        min_residual_auroc=0.65,
        min_target_residual_bg_ratio=1.5,
        min_candidate_to_budget_ratio=1.0,
    )

    add_split_decision(summary, args)

    assert summary["background_residual_gate_pass"] is False
    assert summary["candidate_mining_gate_pass"] is True
    assert summary["overall_decision"] == "STOP_BCV"
    assert summary["gate_pass"] is False
