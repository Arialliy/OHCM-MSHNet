import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.audit_apf_candidates import select_candidate_mask


def test_apf_candidate_excludes_target_dilation():
    prob = np.zeros((32, 32), dtype=np.float32)
    prob[8:24, 8:24] = 0.5
    gt = np.zeros((32, 32), dtype=bool)
    gt[15:17, 15:17] = True

    candidate, _stats = select_candidate_mask(
        prob=prob,
        gt_mask=gt,
        target_dilation_radius=5,
        tau_low=0.25,
        tau_high=0.60,
        hard_top_q=0.01,
    )

    assert not bool((candidate & gt).any())
    assert candidate.sum() > 0


def test_check_failed_routes_blocks_pfr_by_default():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py"),
            "--model_name",
            "PFRMSHNet",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "[BLOCKED] PFRMSHNet" in result.stdout


def test_check_failed_routes_allows_explicit_failure_analysis():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py"),
            "--model_name",
            "PFRMSHNet",
            "--allow_failed_route",
            "--reason",
            "failure-analysis rerun only",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "failure-analysis rerun only" in result.stdout


def test_check_apf_ready_blocks_failed_summary(tmp_path):
    summary = {
        "gate_pass": False,
        "num_images": 1,
        "target_leakage_pixels_total": 0,
        "candidate_to_budget_ratio_mean": 0.1,
        "flat_bg_ratio_mean": 0.0,
        "ohem_fp_component_coverage_mean": 0.0,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    anchor_dir = tmp_path / "anchors"
    candidate_dir = tmp_path / "candidates"
    anchor_dir.mkdir()
    candidate_dir.mkdir()
    checkpoint = tmp_path / "ckpt.pth.tar"
    checkpoint.write_bytes(b"fake")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_apf_ready.py"),
            "--summary",
            str(summary_path),
            "--anchor_dir",
            str(anchor_dir),
            "--candidate_dir",
            str(candidate_dir),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["apf_ready"] is False
    assert "candidate_audit_gate_failed" in payload["errors"]


def test_apf_model_not_registered_before_candidate_audit():
    from net import SUPPORTED_MODEL_NAMES

    assert "MSHNetAPFOHEM" not in SUPPORTED_MODEL_NAMES
