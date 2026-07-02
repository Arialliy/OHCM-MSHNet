import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_bcv_gate_d_residual_shape import audit_image_shape_components, build_gate_d_summary
from utils.residual_shape_features import ResidualShapeWeights


def gate_d_args(min_far_fp_components=1):
    return SimpleNamespace(
        target_near_radius=3,
        match_distance=0.0,
        shape_ring_radius=3,
        dog_sigma_small=1.0,
        dog_sigma_large=2.0,
        min_far_fp_components=min_far_fp_components,
        min_shape_auc=0.70,
        hard_stop_min_shape_auc=0.65,
        min_single_feature_auc=0.70,
        min_suppressible_far_fp_rate_99=0.20,
        min_suppressible_far_fp_rate_995=0.10,
        hard_stop_min_suppressible_far_fp_rate_99=0.10,
    )


def test_bcv_gate_d_passes_when_shape_separates_target_from_far_fp():
    gt = np.zeros((24, 24), dtype=bool)
    gt[3:6, 3:6] = True
    pred = np.zeros_like(gt)
    pred[3:6, 3:6] = True
    pred[18, 4:11] = True
    residual = np.full((24, 24), 0.1, dtype=np.float32)
    residual[gt] = 3.0
    residual[18, 4:11] = 3.0

    args = gate_d_args()
    rows, counts = audit_image_shape_components("shape_pass", gt, pred, residual, args, ResidualShapeWeights())
    summary = build_gate_d_summary(rows, [counts], args)

    assert summary["target_component_count"] == 1
    assert summary["far_fp_component_count"] == 1
    assert summary["shape_auc_target_vs_far_fp"] == 1.0
    assert summary["suppressible_far_fp_rate_at_target_recall_99"] == 1.0
    assert summary["gate_pass"] is True
    assert summary["overall_decision"] == "PROCEED_TO_DETERMINISTIC_SHAPE_CALIBRATION"


def test_bcv_gate_d_stops_when_far_fp_shape_is_more_target_like():
    gt = np.zeros((24, 24), dtype=bool)
    gt[4, 3:10] = True
    pred = np.zeros_like(gt)
    pred[4, 3:10] = True
    pred[18:21, 18:21] = True
    residual = np.full((24, 24), 0.1, dtype=np.float32)
    residual[gt] = 3.0
    residual[18:21, 18:21] = 3.0

    args = gate_d_args()
    rows, counts = audit_image_shape_components("shape_fail", gt, pred, residual, args, ResidualShapeWeights())
    summary = build_gate_d_summary(rows, [counts], args)

    assert summary["gate_pass"] is False
    assert summary["overall_decision"] == "STOP_BCV"
    assert summary["stop_conditions"]["shape_auc_below_hard_stop"] is True
