import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.audit_ohem_error_components import build_summary
from tools.official.build_ohem_error_components import classify_components


def test_component_touching_gt_is_not_detached_far_fp():
    gt = np.zeros((32, 32), dtype=np.uint8)
    pred = np.zeros((32, 32), dtype=np.uint8)
    gt[10:12, 10:12] = 1
    pred[10:12, 10:12] = 1

    comps = classify_components(pred, gt, target_dilate_radius=3, near_radius=8, far_radius=12)

    assert comps[0].component_type == "target_hit_or_overlap"
    assert not comps[0].is_detached_far_fp


def test_boundary_dilation_component_is_boundary_excess():
    gt = np.zeros((32, 32), dtype=np.uint8)
    pred = np.zeros((32, 32), dtype=np.uint8)
    gt[10:12, 10:12] = 1
    pred[13:15, 10:12] = 1

    comps = classify_components(pred, gt, target_dilate_radius=3, near_radius=8, far_radius=12)

    assert comps[0].component_type == "boundary_excess"
    assert comps[0].is_boundary_excess == 1


def test_far_component_is_detached_far_fp():
    gt = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    gt[5:7, 5:7] = 1
    pred[50:53, 50:53] = 1

    comps = classify_components(pred, gt, target_dilate_radius=3, near_radius=8, far_radius=16)

    assert comps[0].component_type == "detached_far_fp"
    assert comps[0].is_detached_far_fp == 1


def test_flat_component_contrast_classification_correct():
    gt = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    prob = np.zeros((64, 64), dtype=np.float32) + 0.1
    image = np.zeros((64, 64), dtype=np.float32)
    pred[40:43, 40:43] = 1
    prob[40:43, 40:43] = 0.51

    comps = classify_components(
        pred,
        gt,
        prob=prob,
        image=image,
        target_dilate_radius=3,
        near_radius=8,
        far_radius=16,
        prob_contrast_min=0.05,
        image_contrast_min=0.50,
    )

    assert comps[0].is_nonflat == 1
    assert comps[0].prob_contrast > 0.05


def test_target_like_area_uses_train_gt_area_statistics():
    gt = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[40:43, 40:43] = 1
    stats = {
        "target_area_p01": 4,
        "target_area_p05": 6,
        "target_area_p25": 8,
        "target_area_median": 9,
        "target_area_p75": 12,
        "target_area_p95": 16,
        "target_area_p99": 20,
    }

    comps = classify_components(pred, gt, target_area_stats=stats, target_dilate_radius=3, near_radius=8, far_radius=16)

    assert comps[0].area == 9
    assert comps[0].is_target_like_area == 1


def write_component_csv(path: Path, rows: list[dict]):
    fields = [
        "image_id",
        "component_id",
        "component_type",
        "area",
        "max_prob",
        "is_nonflat",
        "is_target_like_area",
        "target_leakage_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_gate_false_blocks_ready_check(tmp_path):
    summary = {
        "gate_pass": False,
        "total_detached_far_fp_components": 0,
        "nonflat_detached_far_fp_ratio": 0.0,
        "flat_bg_ratio_mean": 1.0,
        "target_leakage_components": 0,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_error_component_ready.py"),
            "--audit_summary",
            str(summary_path),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["error_component_ready"] is False
    assert "error_component_audit_gate_failed" in payload["errors"]


def test_non_target_component_leakage_forces_no_go(tmp_path):
    component_csv = tmp_path / "error_components.csv"
    summary_json = tmp_path / "summary.json"
    image_csv = tmp_path / "image_level_counts.csv"
    summary_json.write_text(json.dumps({"num_images": 1}), encoding="utf-8")
    image_csv.write_text(
        "image_id,component_count,detached_far_fp_components,boundary_excess_components,ohem_budget_pixels\n"
        "a,1,0,0,1\n",
        encoding="utf-8",
    )
    write_component_csv(
        component_csv,
        [
            {
                "image_id": "a",
                "component_id": 1,
                "component_type": "detached_far_fp",
                "area": 4,
                "max_prob": 0.9,
                "is_nonflat": 1,
                "is_target_like_area": 1,
                "target_leakage_pixels": 4,
            }
        ],
    )

    summary, _type_rows, _nonflat_rows, _image_rows = build_summary(
        component_csv,
        summary_json,
        {
            "min_detached_far_fp_components": 0,
            "min_images_with_detached_far_fp_ratio": 0.0,
            "min_nonflat_detached_far_fp_ratio": 0.0,
            "min_target_like_area_detached_far_fp_ratio": 0.0,
            "min_mean_detached_far_fp_peak_prob": 0.0,
            "max_boundary_excess_dominance_ratio": 1.0,
            "min_train_candidate_to_budget_ratio_mean": 0.0,
            "max_flat_bg_ratio_mean": 1.0,
        },
    )

    assert summary["gate_pass"] is False
    assert "target_leakage_components_nonzero" in summary["fail_reasons"]
