import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tcd_ready_guard_exits_nonzero_on_failed_summary(tmp_path):
    summary = {
        "gate_pass": False,
        "fail_reasons": ["topk_far_teacher_student_diff_too_small"],
        "topk_far_absdiff_mean": 0.0,
        "teacher_lower_on_student_high_far_rate": 0.0,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_tcd_ready.py"),
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
    assert "TCD_NOT_READY" in result.stdout
