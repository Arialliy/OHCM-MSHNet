from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_final_stop_state import (
    FinalStopStateError,
    check_final_stop_state,
)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_minimal_stopped_repo(root: Path) -> None:
    tce_dir = root / "docs" / "internal" / "tce_final"
    write_json(
        tce_dir / "gate_tce_f3_fail_summary.json",
        {
            "gate": "Gate-TCE-F3-blind-external-once",
            "decision": "F3_FAIL_NO_REDESIGN",
            "failed_splits": {
                "external_nuaa_sirst": {"min_delta_Pd": -0.018348624},
                "external_irstd_1k": {"min_delta_Pd": -0.013468013},
            },
            "not_completed_splits": {
                "external_sirst3": {
                    "reason": "manifest integrity failure: missing masks/images",
                    "total_entries": 1079,
                    "missing_masks": 365,
                    "missing_images": 1,
                }
            },
            "forbidden_next_actions": [
                "threshold search",
                "seed search",
                "checkpoint search",
                "split redefinition",
                "new model training",
            ],
        },
    )
    write_json(
        tce_dir / "gate_tce_f3_once_lock.json",
        {"status": "STOPPED_BY_F3_PD_REGRESSION"},
    )
    write_text(
        root / "README.md",
        """
# OHCM-MSHNet

## Current Official Status

STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
STOP_TCSR_AT_BANK_AUDIT
No active AAAI main-method branch remains.

Forbidden: no new training, no new evaluation, no threshold search.
""".strip(),
    )
    write_text(
        root / "STOPPED_BRANCHES_SUMMARY.md",
        """
# Stopped Branches Summary

STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
F3_FAIL_NO_REDESIGN because of external Pd regression.
STOP_TCSR_AT_BANK_AUDIT.
STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD.
""".strip(),
    )


def test_final_stop_state_passes(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    out = tmp_path / "docs" / "internal" / "final_stop_state_summary.json"

    result = check_final_stop_state(root=tmp_path, output_path=out)

    assert result["gate_pass"] is True
    assert result["decision"] == "READ_ONLY_FAILURE_ARCHIVE_STATE"
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["gate"] == "Gate-FINAL-STOP-CONSISTENCY"


def test_final_stop_state_fails_if_lock_is_started(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_json(
        tmp_path / "docs" / "internal" / "tce_final" / "gate_tce_f3_once_lock.json",
        {"status": "STARTED"},
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)


def test_final_stop_state_fails_if_no_negative_pd(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_json(
        tmp_path / "docs" / "internal" / "tce_final" / "gate_tce_f3_fail_summary.json",
        {
            "gate": "Gate-TCE-F3-blind-external-once",
            "decision": "F3_FAIL_NO_REDESIGN",
            "failed_splits": {
                "external_nuaa_sirst": {"min_delta_Pd": 0.0},
            },
        },
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)


def test_final_stop_state_fails_if_final_report_exists(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_json(
        tmp_path / "docs" / "internal" / "tce_final" / "gate_tce_f3_blind_external_report.json",
        {"should_not_exist": True},
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)


def test_final_stop_state_fails_if_readme_top_still_active(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_text(
        tmp_path / "README.md",
        """
# OHCM-MSHNet

## Current Official Status

Current active candidate:
- TWA-OHEM without BN recalibration
- next allowed gate: Gate-TWA-D HC-Val on seed42 only
""".strip(),
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)
