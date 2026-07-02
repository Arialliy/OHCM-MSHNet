import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ecdv_gate_b_passes_valid_summary(tmp_path):
    summary = {
        "target_dilate_overlap_pixels": 0,
        "decoys_per_image_mean": 1.0,
        "evidence_response_success_ratio": 0.50,
        "mean_prob_gain": 0.20,
        "area_in_target_range_ratio": 0.80,
        "flat_artifact_ratio": 0.30,
        "preview_audit_pass": True,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_ecdv_gate_b.py"),
            "--bank_dir",
            str(tmp_path),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    checked = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert checked["gate_pass"] is True


def test_ecdv_gate_b_blocks_target_overlap(tmp_path):
    summary = {
        "target_dilate_overlap_pixels": 1,
        "decoys_per_image_mean": 1.0,
        "evidence_response_success_ratio": 0.80,
        "mean_prob_gain": 0.30,
        "area_in_target_range_ratio": 1.0,
        "flat_artifact_ratio": 0.0,
        "preview_audit_pass": True,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_ecdv_gate_b.py"),
            "--bank_dir",
            str(tmp_path),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    checked = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert checked["checks"]["target_dilate_overlap_pixels"] is False
    assert checked["gate_pass"] is False
