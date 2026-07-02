#!/usr/bin/env python3
"""Audit whether OHEM error components can support component-level mining."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Audit OHEM error components for Gate-ECA.")
    parser.add_argument("--component_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_detached_far_fp_components", type=int, default=50)
    parser.add_argument("--min_images_with_detached_far_fp_ratio", type=float, default=0.08)
    parser.add_argument("--min_nonflat_detached_far_fp_ratio", type=float, default=0.30)
    parser.add_argument("--min_target_like_area_detached_far_fp_ratio", type=float, default=0.30)
    parser.add_argument("--min_mean_detached_far_fp_peak_prob", type=float, default=0.50)
    parser.add_argument("--max_boundary_excess_dominance_ratio", type=float, default=0.70)
    parser.add_argument("--min_train_candidate_to_budget_ratio_mean", type=float, default=1.5)
    parser.add_argument("--max_flat_bg_ratio_mean", type=float, default=0.50)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def as_int(row: dict, key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def build_summary(component_csv: Path, summary_json: Path, thresholds: dict) -> tuple[dict, dict, list[dict], list[dict]]:
    build_summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
    components = read_csv(component_csv)
    image_csv = component_csv.parent / "image_level_counts.csv"
    image_rows = read_csv(image_csv)
    num_images = int(build_summary.get("num_images", len(image_rows)) or 0)

    type_counts = Counter(row.get("component_type", "") for row in components)
    detached = [row for row in components if row.get("component_type") == "detached_far_fp"]
    nonflat_detached = [row for row in detached if as_int(row, "is_nonflat") > 0]
    target_like_detached = [row for row in detached if as_int(row, "is_target_like_area") > 0]
    target_leakage = [
        row
        for row in components
        if row.get("component_type") != "target_hit_or_overlap" and as_int(row, "target_leakage_pixels") > 0
    ]
    boundary = [row for row in components if row.get("component_type") == "boundary_excess"]
    fp_like = [row for row in components if row.get("component_type") != "target_hit_or_overlap"]

    detached_images = {row.get("image_id") for row in detached}
    detached_peak_probs = [as_float(row, "max_prob") for row in detached]
    total_detached_area = sum(as_int(row, "area") for row in detached)
    flat_detached_area = sum(as_int(row, "area") for row in detached if as_int(row, "is_nonflat") == 0)
    flat_bg_ratio_mean = safe_div(flat_detached_area, total_detached_area) if detached else 1.0

    detached_area_by_image = defaultdict(float)
    for row in detached:
        detached_area_by_image[row.get("image_id")] += as_float(row, "area")
    ratio_values = []
    image_level_counts = []
    for row in image_rows:
        image_id = row.get("image_id", "")
        budget = max(1.0, as_float(row, "ohem_budget_pixels", 1.0))
        detached_area = detached_area_by_image.get(image_id, 0.0)
        ratio = safe_div(detached_area, budget)
        ratio_values.append(ratio)
        image_level_counts.append(
            {
                "image_id": image_id,
                "component_count": as_int(row, "component_count"),
                "detached_far_fp_components": as_int(row, "detached_far_fp_components"),
                "boundary_excess_components": as_int(row, "boundary_excess_components"),
                "detached_far_fp_area": detached_area,
                "ohem_budget_pixels": budget,
                "candidate_to_budget_ratio": ratio,
            }
        )

    summary = {
        "gate_pass": True,
        "fail_reasons": [],
        "num_images": num_images,
        "component_count_total": len(components),
        "total_detached_far_fp_components": len(detached),
        "images_with_detached_far_fp": len(detached_images),
        "images_with_detached_far_fp_ratio": safe_div(len(detached_images), num_images),
        "nonflat_detached_far_fp_ratio": safe_div(len(nonflat_detached), len(detached)),
        "target_like_area_detached_far_fp_ratio": safe_div(len(target_like_detached), len(detached)),
        "mean_detached_far_fp_peak_prob": float(np.mean(detached_peak_probs)) if detached_peak_probs else 0.0,
        "target_leakage_components": len(target_leakage),
        "target_leakage_pixels_total": sum(as_int(row, "target_leakage_pixels") for row in target_leakage),
        "boundary_excess_dominance_ratio": safe_div(len(boundary), len(fp_like)),
        "train_candidate_to_budget_ratio_mean": float(np.mean(ratio_values)) if ratio_values else 0.0,
        "flat_bg_ratio_mean": flat_bg_ratio_mean,
        "component_csv": str(component_csv),
        "summary_json": str(summary_json),
        "thresholds": thresholds,
    }

    checks = [
        (
            summary["total_detached_far_fp_components"] < thresholds["min_detached_far_fp_components"],
            "total_detached_far_fp_components_too_low",
        ),
        (
            summary["images_with_detached_far_fp_ratio"] < thresholds["min_images_with_detached_far_fp_ratio"],
            "images_with_detached_far_fp_ratio_too_low",
        ),
        (
            summary["nonflat_detached_far_fp_ratio"] < thresholds["min_nonflat_detached_far_fp_ratio"],
            "nonflat_detached_far_fp_ratio_too_low",
        ),
        (
            summary["target_like_area_detached_far_fp_ratio"]
            < thresholds["min_target_like_area_detached_far_fp_ratio"],
            "target_like_area_detached_far_fp_ratio_too_low",
        ),
        (
            summary["mean_detached_far_fp_peak_prob"] < thresholds["min_mean_detached_far_fp_peak_prob"],
            "mean_detached_far_fp_peak_prob_too_low",
        ),
        (summary["target_leakage_components"] > 0, "target_leakage_components_nonzero"),
        (
            summary["boundary_excess_dominance_ratio"] > thresholds["max_boundary_excess_dominance_ratio"],
            "boundary_excess_dominance_ratio_too_high",
        ),
        (
            summary["train_candidate_to_budget_ratio_mean"]
            < thresholds["min_train_candidate_to_budget_ratio_mean"],
            "train_candidate_to_budget_ratio_mean_too_low",
        ),
        (
            summary["flat_bg_ratio_mean"] > thresholds["max_flat_bg_ratio_mean"],
            "flat_bg_ratio_mean_too_high",
        ),
    ]
    summary["fail_reasons"] = [reason for failed, reason in checks if failed]
    summary["gate_pass"] = len(summary["fail_reasons"]) == 0
    type_count_rows = [{"component_type": key, "count": value} for key, value in sorted(type_counts.items())]
    nonflat_rows = [row for row in detached if as_int(row, "is_nonflat") > 0]
    return summary, type_count_rows, nonflat_rows, image_level_counts


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "min_detached_far_fp_components": args.min_detached_far_fp_components,
        "min_images_with_detached_far_fp_ratio": args.min_images_with_detached_far_fp_ratio,
        "min_nonflat_detached_far_fp_ratio": args.min_nonflat_detached_far_fp_ratio,
        "min_target_like_area_detached_far_fp_ratio": args.min_target_like_area_detached_far_fp_ratio,
        "min_mean_detached_far_fp_peak_prob": args.min_mean_detached_far_fp_peak_prob,
        "max_boundary_excess_dominance_ratio": args.max_boundary_excess_dominance_ratio,
        "min_train_candidate_to_budget_ratio_mean": args.min_train_candidate_to_budget_ratio_mean,
        "max_flat_bg_ratio_mean": args.max_flat_bg_ratio_mean,
    }
    summary, type_rows, nonflat_rows, image_rows = build_summary(
        Path(args.component_csv), Path(args.summary_json), thresholds
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "component_type_counts.csv", type_rows, ["component_type", "count"])
    if nonflat_rows:
        write_csv(output_dir / "nonflat_detached_far_fp.csv", nonflat_rows, list(nonflat_rows[0].keys()))
    else:
        write_csv(output_dir / "nonflat_detached_far_fp.csv", [], [])
    write_csv(
        output_dir / "image_level_counts.csv",
        image_rows,
        [
            "image_id",
            "component_count",
            "detached_far_fp_components",
            "boundary_excess_components",
            "detached_far_fp_area",
            "ohem_budget_pixels",
            "candidate_to_budget_ratio",
        ],
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
