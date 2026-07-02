import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.audit_tce_teacher_information import audit_records


def make_record(teacher, student, gt):
    return {
        "image_name": "synthetic",
        "teacher_prob": teacher.astype(np.float32),
        "student_prob": student.astype(np.float32),
        "gt_mask": gt.astype(bool),
        "p_250": teacher.astype(np.float32),
        "p_300": student.astype(np.float32),
    }


def base_arrays():
    gt = np.zeros((100, 100), dtype=bool)
    gt[48:52, 48:52] = True
    student = np.zeros((100, 100), dtype=np.float32)
    student[gt] = 0.90
    student[0, 0:10] = 0.80
    teacher = student.copy()
    return teacher, student, gt


def test_global_absdiff_low_can_pass_if_topk_far_signal_is_informative():
    teacher, student, gt = base_arrays()
    teacher[0, 0:10] = 0.20

    summary, _rows = audit_records(
        [make_record(teacher, student, gt)],
        topk_far_ratio=0.001,
        expected_count=1,
    )

    assert summary["global_absdiff_mean"] < 0.001
    assert summary["topk_far_absdiff_mean"] >= 0.003
    assert summary["teacher_lower_on_student_high_far_rate"] >= 0.20
    assert summary["teacher_preserves_target_rate"] >= 0.95
    assert summary["gate_pass"] is True


def test_t2r_fails_when_teacher_equals_student_everywhere():
    teacher, student, gt = base_arrays()

    summary, _rows = audit_records(
        [make_record(teacher, student, gt)],
        topk_far_ratio=0.001,
        expected_count=1,
    )

    assert summary["gate_pass"] is False
    assert "topk_far_teacher_student_diff_too_small" in summary["fail_reasons"]


def test_t2r_fails_when_teacher_does_not_preserve_target():
    teacher, student, gt = base_arrays()
    teacher[0, 0:10] = 0.20
    teacher[gt] = 0.10

    summary, _rows = audit_records(
        [make_record(teacher, student, gt)],
        topk_far_ratio=0.001,
        expected_count=1,
    )

    assert summary["gate_pass"] is False
    assert "teacher_does_not_preserve_targets" in summary["fail_reasons"]
