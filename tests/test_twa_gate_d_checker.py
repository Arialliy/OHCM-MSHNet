import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_twa_gate_d_hcval import evaluate_gate


def summary(miou, fa_ppm, precision, pd):
    return {
        "metrics_at_threshold": {
            "threshold": 0.5,
            "mIoU": miou,
            "FA_ppm": fa_ppm,
            "Precision": precision,
            "Pd": pd,
        }
    }


def run_gate(ohem, twa):
    return evaluate_gate(
        ohem,
        twa,
        min_delta_miou=0.005,
        min_fa_reduction=10.0,
        min_delta_precision=0.0,
        min_delta_pd=0.0,
    )


def test_gate_d_pass_when_hcval_improves():
    result = run_gate(
        summary(0.60, 380.0, 0.66, 0.83),
        summary(0.606, 369.0, 0.67, 0.84),
    )

    assert result["gate_pass"] is True
    assert result["next_allowed_gate"] == "Gate-TWA-E"


def test_gate_d_fails_when_pd_drops():
    result = run_gate(
        summary(0.60, 380.0, 0.66, 0.83),
        summary(0.606, 369.0, 0.67, 0.82),
    )

    assert result["gate_pass"] is False
    assert result["checks"]["delta_Pd"] is False
    assert result["next_allowed_gate"] == "STOP_TWA_4_NO_BN"


def test_gate_d_fails_when_fa_not_reduced():
    result = run_gate(
        summary(0.60, 380.0, 0.66, 0.83),
        summary(0.606, 371.0, 0.67, 0.84),
    )

    assert result["gate_pass"] is False
    assert result["checks"]["delta_FA_ppm"] is False


def test_gate_d_requires_metric_presence():
    with pytest.raises(KeyError):
        run_gate(summary(0.60, 380.0, 0.66, 0.83), {"metrics_at_threshold": {}})
