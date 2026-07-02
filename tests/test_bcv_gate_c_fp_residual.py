import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_bcv_gate_c_fp_residual import audit_image_components, build_summary


def gate_c_args(min_far_fp_components=1):
    return SimpleNamespace(
        min_far_fp_components=min_far_fp_components,
        min_target_vs_far_fp_auroc=0.65,
        hard_stop_min_auroc=0.60,
        min_suppressible_far_fp_rate_99=0.20,
        min_suppressible_far_fp_rate_995=0.10,
        hard_stop_min_suppressible_far_fp_rate_99=0.10,
    )


def test_bcv_gate_c_separates_target_near_fp_and_far_fp_components():
    gt = np.zeros((16, 16), dtype=bool)
    gt[2:4, 2:4] = True
    pred = np.zeros_like(gt)
    pred[2:4, 2:4] = True
    pred[3:5, 6:8] = True
    pred[12:14, 12:14] = True
    residual = np.full((16, 16), 0.2, dtype=np.float32)
    residual[gt] = 3.0
    residual[3:5, 6:8] = 0.4
    residual[12:14, 12:14] = 0.1

    component_rows, image_counts = audit_image_components(
        image_name="synthetic",
        gt_mask=gt,
        pred_mask=pred,
        residual=residual,
        target_near_radius=3,
        match_distance=0.0,
    )
    summary = build_summary(component_rows, [image_counts], gate_c_args())

    assert summary["target_component_count"] == 1
    assert summary["matched_target_prediction_component_count"] == 1
    assert summary["near_fp_component_count"] == 1
    assert summary["far_fp_component_count"] == 1
    assert summary["target_vs_far_fp_residual_auroc"] == 1.0
    assert summary["suppressible_far_fp_rate_at_target_recall_99"] == 1.0
    assert summary["gate_pass"] is True
    assert summary["overall_decision"] == "PROCEED_TO_DETERMINISTIC_CALIBRATION"


def test_bcv_gate_c_stops_when_far_fp_residual_overlaps_target():
    gt = np.zeros((16, 16), dtype=bool)
    gt[2:4, 2:4] = True
    pred = np.zeros_like(gt)
    pred[2:4, 2:4] = True
    pred[12:14, 12:14] = True
    residual = np.full((16, 16), 0.2, dtype=np.float32)
    residual[gt] = 0.2
    residual[12:14, 12:14] = 0.4

    component_rows, image_counts = audit_image_components(
        image_name="overlap",
        gt_mask=gt,
        pred_mask=pred,
        residual=residual,
        target_near_radius=3,
        match_distance=0.0,
    )
    summary = build_summary(component_rows, [image_counts], gate_c_args())

    assert summary["gate_pass"] is False
    assert summary["overall_decision"] == "STOP_BCV"
    assert summary["stop_conditions"]["auroc_below_hard_stop"] is True
