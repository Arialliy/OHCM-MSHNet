import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_tcsr_bank_gate_a import evaluate


def make_summary(**overrides):
    items = [{"image_id": f"{idx:06d}", "neg_pixels": 1, "protect_pixels": 2} for idx in range(60)]
    summary = {
        "split": "train",
        "train_only": True,
        "num_images": 60,
        "train_images": 60,
        "num_images_with_neg": 60,
        "neg_pixels_total": 600,
        "protect_pixels_total": 120,
        "target_leakage_pixels_total": 0,
        "neg_protect_overlap_pixels_total": 0,
        "items": items,
    }
    summary.update(overrides)
    return summary


def test_tcsr_gate_a_passes_valid_bank_summary():
    result = evaluate(make_summary())

    assert result["gate_pass"] is True
    assert result["decision"] == "PASS_TCSR_BANK_AUDIT"
    assert result["next_allowed_gate"] == "Gate-TCSR-B-activation-sanity"


def test_tcsr_gate_a_fails_target_leakage():
    result = evaluate(make_summary(target_leakage_pixels_total=1))

    assert result["gate_pass"] is False
    assert result["decision"] == "STOP_TCSR_AT_BANK_AUDIT"
    assert "no_target_leakage" in result["fail_reasons"]


def test_tcsr_gate_a_fails_sparse_bank_too_small():
    result = evaluate(make_summary(num_images_with_neg=10, neg_pixels_total=100))

    assert result["gate_pass"] is False
    assert "enough_images_with_neg" in result["fail_reasons"]
    assert "enough_neg_pixels" in result["fail_reasons"]


def test_tcsr_gate_a_fails_neg_protect_overlap():
    result = evaluate(make_summary(neg_protect_overlap_pixels_total=1))

    assert result["gate_pass"] is False
    assert "no_neg_protect_overlap" in result["fail_reasons"]
