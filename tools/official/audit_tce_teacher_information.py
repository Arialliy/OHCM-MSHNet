#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label


def parse_args():
    parser = argparse.ArgumentParser(description="Gate-T2R audit for TCE teacher information in hard regions.")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--topk_far_ratio", type=float, default=0.005)
    parser.add_argument("--target_dilate", type=int, default=5)
    parser.add_argument("--far_dilate", type=int, default=9)
    parser.add_argument("--expected_count", type=int, default=697)
    parser.add_argument("--student_high_threshold", type=float, default=0.5)
    return parser.parse_args()


def safe_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def safe_rate(mask) -> float:
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


def topk_mask(scores: np.ndarray, valid_mask: np.ndarray, ratio: float) -> np.ndarray:
    valid_idx = np.flatnonzero(valid_mask.ravel())
    out = np.zeros(scores.shape, dtype=bool)
    if valid_idx.size == 0:
        return out
    k = max(1, int(valid_idx.size * float(ratio)))
    valid_scores = scores.ravel()[valid_idx]
    if k >= valid_idx.size:
        chosen = valid_idx
    else:
        chosen = valid_idx[np.argpartition(valid_scores, -k)[-k:]]
    out.ravel()[chosen] = True
    return out & valid_mask


def make_masks(gt_mask: np.ndarray, student_prob: np.ndarray, topk_far_ratio: float, target_dilate: int, far_dilate: int):
    gt = gt_mask.astype(bool)
    target_near = binary_dilation(gt, iterations=int(target_dilate))
    target_boundary = target_near & (~gt)
    far_bg = ~binary_dilation(gt, iterations=int(far_dilate))
    topk_far = topk_mask(student_prob, far_bg, topk_far_ratio)
    return gt, target_near, target_boundary, far_bg, topk_far


def detached_fp_mask(student_prob: np.ndarray, gt_mask: np.ndarray, target_near: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    pred = student_prob > threshold
    labeled, num = label(pred)
    detached = np.zeros_like(pred, dtype=bool)
    for component_id in range(1, num + 1):
        component = labeled == component_id
        if not np.logical_and(component, target_near).any():
            detached |= component
    return detached & (~gt_mask.astype(bool))


def member_stack(record: dict) -> np.ndarray | None:
    member_keys = sorted(
        [key for key in record if re.fullmatch(r"p_\d+", key)],
        key=lambda key: int(key.split("_")[1]),
    )
    if len(member_keys) < 2:
        return None
    return np.stack([record[key].astype(np.float32) for key in member_keys], axis=0)


def load_record(path: Path) -> dict:
    data = np.load(path)
    teacher_prob = data["teacher_prob"].astype(np.float32) if "teacher_prob" in data else data["p_tce"].astype(np.float32)
    if "student_prob" in data:
        student_prob = data["student_prob"].astype(np.float32)
    elif "p_ohem_400" in data:
        student_prob = data["p_ohem_400"].astype(np.float32)
    else:
        student_prob = data["p_400"].astype(np.float32)
    if "gt_mask" not in data:
        raise KeyError("Missing gt_mask in %s. Rebuild TCE soft labels with updated build_tce_soft_labels.py." % path)
    record = {
        "image_name": path.stem,
        "teacher_prob": teacher_prob,
        "student_prob": student_prob,
        "gt_mask": data["gt_mask"].astype(bool),
    }
    for key in data.files:
        if re.fullmatch(r"p_\d+", key):
            record[key] = data[key].astype(np.float32)
    return record


def audit_records(
    records: list[dict],
    topk_far_ratio: float = 0.005,
    target_dilate: int = 5,
    far_dilate: int = 9,
    expected_count: int | None = 697,
    student_high_threshold: float = 0.5,
):
    rows = []
    all_diff = []
    target_diff = []
    boundary_diff = []
    far_diff = []
    ohem_neg_diff = []
    topk_far_diff = []
    detached_diff = []
    topk_rank_disagreement = []
    teacher_lower_rates = []
    teacher_preserve_rates = []
    variance_topk_far = []
    variance_target = []
    informative_images = 0

    for record in records:
        teacher = record["teacher_prob"].astype(np.float32)
        student = record["student_prob"].astype(np.float32)
        gt = record["gt_mask"].astype(bool)
        diff = np.abs(teacher - student)
        gt, target_near, target_boundary, far_bg, topk_far = make_masks(
            gt, student, topk_far_ratio, target_dilate, far_dilate
        )
        ohem_neg = far_bg & (student > student_high_threshold)
        if not ohem_neg.any():
            ohem_neg = topk_far
        detached_fp = detached_fp_mask(student, gt, target_near, threshold=student_high_threshold)
        teacher_topk_far = topk_mask(teacher, far_bg, topk_far_ratio)
        union_topk = topk_far | teacher_topk_far
        overlap_topk = topk_far & teacher_topk_far
        rank_disagreement = 1.0 - (overlap_topk.sum() / max(1, union_topk.sum()))
        teacher_lower_rate = safe_rate(teacher[topk_far] < student[topk_far])
        teacher_preserve_rate = safe_rate((teacher[gt] + 0.02) >= student[gt])
        stack = member_stack(record)
        var_topk = float("nan")
        var_target = float("nan")
        if stack is not None:
            variance = stack.var(axis=0)
            var_topk = safe_mean(variance[topk_far])
            var_target = safe_mean(variance[gt])

        row = {
            "image_name": record.get("image_name", ""),
            "teacher_student_absdiff_all_pixels": safe_mean(diff),
            "teacher_student_absdiff_target": safe_mean(diff[gt]),
            "teacher_student_absdiff_target_boundary": safe_mean(diff[target_boundary]),
            "teacher_student_absdiff_far_background": safe_mean(diff[far_bg]),
            "teacher_student_absdiff_ohem_negatives": safe_mean(diff[ohem_neg]),
            "teacher_student_absdiff_topk_far_evidence": safe_mean(diff[topk_far]),
            "teacher_student_absdiff_detached_fp_components": safe_mean(diff[detached_fp]),
            "teacher_rank_disagreement_topk_far": rank_disagreement,
            "teacher_lower_than_student_on_high_far_rate": teacher_lower_rate,
            "teacher_preserves_target_rate": teacher_preserve_rate,
            "per_checkpoint_variance_topk_far": var_topk,
            "per_checkpoint_variance_target": var_target,
            "topk_far_pixels": int(topk_far.sum()),
            "ohem_negative_pixels": int(ohem_neg.sum()),
            "detached_fp_pixels": int(detached_fp.sum()),
            "target_pixels": int(gt.sum()),
        }
        informative = (
            row["teacher_student_absdiff_topk_far_evidence"] >= 0.003
            and row["teacher_lower_than_student_on_high_far_rate"] >= 0.20
        )
        informative_images += int(informative)
        row["informative_topk_far"] = int(informative)
        rows.append(row)

        all_diff.append(row["teacher_student_absdiff_all_pixels"])
        target_diff.append(row["teacher_student_absdiff_target"])
        boundary_diff.append(row["teacher_student_absdiff_target_boundary"])
        far_diff.append(row["teacher_student_absdiff_far_background"])
        ohem_neg_diff.append(row["teacher_student_absdiff_ohem_negatives"])
        topk_far_diff.append(row["teacher_student_absdiff_topk_far_evidence"])
        detached_diff.append(row["teacher_student_absdiff_detached_fp_components"])
        topk_rank_disagreement.append(rank_disagreement)
        teacher_lower_rates.append(teacher_lower_rate)
        teacher_preserve_rates.append(teacher_preserve_rate)
        variance_topk_far.append(var_topk)
        variance_target.append(var_target)

    num_labels = len(records)
    informative_image_ratio = informative_images / max(1, num_labels)
    fail_reasons = []
    if expected_count is not None and num_labels != int(expected_count):
        fail_reasons.append("missing_soft_labels")
    if safe_mean(topk_far_diff) < 0.003:
        fail_reasons.append("topk_far_teacher_student_diff_too_small")
    if safe_mean(teacher_lower_rates) < 0.20:
        fail_reasons.append("teacher_does_not_suppress_student_high_far")
    if safe_mean(teacher_preserve_rates) < 0.95:
        fail_reasons.append("teacher_does_not_preserve_targets")
    if informative_image_ratio < 0.30:
        fail_reasons.append("too_few_informative_images")

    summary = {
        "num_images": int(num_labels),
        "num_labels": int(num_labels),
        "expected_count": int(expected_count) if expected_count is not None else None,
        "global_absdiff_mean": safe_mean(all_diff),
        "target_absdiff_mean": safe_mean(target_diff),
        "boundary_absdiff_mean": safe_mean(boundary_diff),
        "far_bg_absdiff_mean": safe_mean(far_diff),
        "ohem_neg_absdiff_mean": safe_mean(ohem_neg_diff),
        "topk_far_absdiff_mean": safe_mean(topk_far_diff),
        "detached_fp_absdiff_mean": safe_mean(detached_diff),
        "topk_far_rank_disagreement": safe_mean(topk_rank_disagreement),
        "teacher_lower_on_student_high_far_rate": safe_mean(teacher_lower_rates),
        "teacher_preserves_target_rate": safe_mean(teacher_preserve_rates),
        "per_checkpoint_variance_topk_far_mean": safe_mean(variance_topk_far),
        "per_checkpoint_variance_target_mean": safe_mean(variance_target),
        "num_images_with_informative_topk_far": int(informative_images),
        "informative_image_ratio": float(informative_image_ratio),
        "gate_pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "image_name",
        "teacher_student_absdiff_all_pixels",
        "teacher_student_absdiff_target",
        "teacher_student_absdiff_target_boundary",
        "teacher_student_absdiff_far_background",
        "teacher_student_absdiff_ohem_negatives",
        "teacher_student_absdiff_topk_far_evidence",
        "teacher_student_absdiff_detached_fp_components",
        "teacher_rank_disagreement_topk_far",
        "teacher_lower_than_student_on_high_far_rate",
        "teacher_preserves_target_rate",
        "per_checkpoint_variance_topk_far",
        "per_checkpoint_variance_target",
        "topk_far_pixels",
        "ohem_negative_pixels",
        "detached_fp_pixels",
        "target_pixels",
        "informative_topk_far",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_paths = sorted(path for path in label_dir.glob("*.npz") if path.name != "summary.npz")
    records = [load_record(path) for path in label_paths]
    summary, rows = audit_records(
        records,
        topk_far_ratio=args.topk_far_ratio,
        target_dilate=args.target_dilate,
        far_dilate=args.far_dilate,
        expected_count=args.expected_count,
        student_high_threshold=args.student_high_threshold,
    )
    summary["label_dir"] = str(label_dir)
    summary["topk_far_ratio"] = float(args.topk_far_ratio)
    summary["target_dilate"] = int(args.target_dilate)
    summary["far_dilate"] = int(args.far_dilate)
    summary["outputs"] = {"per_image": str(out_dir / "per_image.csv")}
    write_csv(out_dir / "per_image.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
