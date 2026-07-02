import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_bcv_gate_d2_mass_shape import (
    audit_image_mass_components,
    build_gate_d2_summary,
    weighted_quantile,
)
from utils.residual_shape_features import ResidualShapeWeights


def gate_d2_args(min_far_fp_components=1):
    return SimpleNamespace(
        target_near_radius=3,
        match_distance=0.0,
        shape_ring_radius=3,
        dog_sigma_small=1.0,
        dog_sigma_large=2.0,
        min_far_fp_components=min_far_fp_components,
        min_pixel_mass_rate_component_99=0.15,
        min_confidence_mass_rate_component_99=0.15,
        min_pixel_mass_rate_pixel_995=0.10,
        min_confidence_mass_rate_pixel_995=0.10,
        hard_stop_min_mass_rate=0.10,
    )


def test_weighted_quantile_uses_pixel_mass_weights():
    assert weighted_quantile([1.0, 5.0], [100.0, 1.0], 0.50) == 1.0
    assert weighted_quantile([1.0, 5.0], [1.0, 100.0], 0.50) == 5.0


def test_bcv_gate_d2_passes_when_low_shape_fp_carries_mass():
    rows = [
        {"component_type": "gt_target", "shape_score": 1.0, "area": 20, "confidence_mass": 18.0},
        {"component_type": "gt_target", "shape_score": 2.0, "area": 20, "confidence_mass": 18.0},
        {"component_type": "far_fp", "shape_score": 0.5, "area": 50, "confidence_mass": 45.0, "peak_prob": 0.95},
        {"component_type": "far_fp", "shape_score": 3.0, "area": 10, "confidence_mass": 2.0, "peak_prob": 0.20},
    ]
    image_rows = [{"target_leakage_pixels": 0}]

    summary = build_gate_d2_summary(rows, image_rows, gate_d2_args())

    assert summary["mass_gate_pass"] is True
    assert summary["overall_decision"] == "PROCEED_TO_DETERMINISTIC_FORMULA_CALIBRATION"
    assert summary["suppressible_far_fp_component_rate_at_target_component_recall_99"] == 0.5
    assert summary["suppressible_far_fp_pixel_mass_rate_at_target_component_recall_99"] > 0.8
    assert summary["suppressible_far_fp_confidence_mass_rate_at_target_component_recall_99"] > 0.9


def test_bcv_gate_d2_stops_when_suppressed_fp_has_no_mass():
    rows = [
        {"component_type": "gt_target", "shape_score": 1.0, "area": 20, "confidence_mass": 18.0},
        {"component_type": "gt_target", "shape_score": 2.0, "area": 20, "confidence_mass": 18.0},
        {"component_type": "far_fp", "shape_score": 0.5, "area": 1, "confidence_mass": 0.05, "peak_prob": 0.05},
        {"component_type": "far_fp", "shape_score": 3.0, "area": 99, "confidence_mass": 90.0, "peak_prob": 0.98},
    ]
    image_rows = [{"target_leakage_pixels": 0}]

    summary = build_gate_d2_summary(rows, image_rows, gate_d2_args())

    assert summary["mass_gate_pass"] is False
    assert summary["overall_decision"] == "STOP_BCV"
    assert summary["stop_conditions"]["pixel_mass_rate_component_99_below_hard_stop"] is True
    assert summary["stop_conditions"]["confidence_mass_rate_component_99_below_hard_stop"] is True


def test_bcv_gate_d2_audit_collects_far_fp_area_and_confidence_mass():
    gt = np.zeros((24, 24), dtype=bool)
    gt[3:6, 3:6] = True
    pred = np.zeros_like(gt)
    pred[3:6, 3:6] = True
    pred[18, 4:11] = True
    residual = np.full((24, 24), 0.1, dtype=np.float32)
    residual[gt] = 3.0
    residual[18, 4:11] = 3.0
    prob = np.full((24, 24), 0.05, dtype=np.float32)
    prob[pred] = 0.8

    rows, counts = audit_image_mass_components(
        "mass_collect",
        gt,
        pred,
        prob,
        residual,
        gate_d2_args(),
        ResidualShapeWeights(),
    )
    far = [row for row in rows if row["component_type"] == "far_fp"]

    assert counts["far_fp_component_count"] == 1
    assert far[0]["area"] == 7
    assert np.isclose(far[0]["confidence_mass"], 5.6, atol=1e-6)
    assert np.isclose(far[0]["peak_prob"], 0.8, atol=1e-6)
