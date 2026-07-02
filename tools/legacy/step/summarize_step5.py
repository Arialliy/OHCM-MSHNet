#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(path: Path, method: str, split: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics_at_threshold"]
    row = {
        "method": method,
        "dataset": data.get("dataset", ""),
        "train_dataset": data.get("train_dataset", data.get("dataset", "")),
        "seed": data.get("seed", ""),
        "split": split,
        "num_images": data.get("num_images", ""),
        "summary": str(path),
    }
    row.update({metric: metrics.get(metric, "") for metric in METRICS})
    return row


def add_baseline_rows(baseline_root: Path, rows_full, rows_hc):
    if not baseline_root.exists():
        return
    for path in sorted(baseline_root.glob("*/seed_*/exports/summary_metrics.json")):
        rows_full.append(metric_row(path, "MSHNet", "full"))
    for path in sorted((baseline_root / "step2_eval" / "MSHNet").glob("*/seed_*/hcset/summary_metrics.json")):
        rows_hc.append(metric_row(path, "MSHNet", "hcset"))


def add_step5_rows(run_root: Path, rows_full, rows_hc):
    for path in sorted(run_root.glob("*/*/seed_*/eval_full/summary_metrics.json")):
        method = path.relative_to(run_root).parts[0]
        rows_full.append(metric_row(path, method, "full"))
    for path in sorted(run_root.glob("*/*/seed_*/eval_hcset/summary_metrics.json")):
        method = path.relative_to(run_root).parts[0]
        rows_hc.append(metric_row(path, method, "hcset"))


def add_fp_rows(baseline_root: Path, run_root: Path):
    rows = []
    baseline_step1 = baseline_root / "step1_summary_table.csv"
    if baseline_step1.exists():
        with baseline_step1.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["method"] = "MSHNet"
                rows.append(row)
    for path in sorted(run_root.glob("*/*/seed_*/step1_fp_analysis/step1_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = path.relative_to(run_root).parts[0]
        rows.append(
            {
                "method": method,
                "dataset": data.get("dataset", ""),
                "seed": data.get("seed", ""),
                "threshold": data.get("threshold", ""),
                "num_images": data.get("num_images", ""),
                "total_gt_components": data.get("total_gt_components", ""),
                "total_pred_components": data.get("total_pred_components", ""),
                "false_positive_components": data.get("false_positive_components", ""),
                "target_near_components_excluded": data.get("target_near_components_excluded", ""),
                "high_response_fp_components": data.get("high_response_fp_components", ""),
                "target_like_hard_clutter_components": data.get("target_like_hard_clutter_components", ""),
                "hard_clutter_fraction_of_fp": data.get("hard_clutter_fraction_of_fp", ""),
                "high_response_fraction_of_fp": data.get("high_response_fraction_of_fp", ""),
                "gt_area_median": data.get("gt_area_median", ""),
                "summary": str(path),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize Step5 full-test, HC-Set, and FP decomposition results.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--baseline_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step0_baseline/20260611_155232")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    baseline_root = Path(args.baseline_root)
    rows_full = []
    rows_hc = []
    add_baseline_rows(baseline_root, rows_full, rows_hc)
    add_step5_rows(run_root, rows_full, rows_hc)

    fields = ["method", "dataset", "train_dataset", "seed", "split", "num_images", *METRICS, "summary"]
    write_csv(run_root / "step5_same_domain_table.csv", rows_full, fields)
    write_csv(run_root / "step5_hcset_table.csv", rows_hc, fields)

    fp_rows = add_fp_rows(baseline_root, run_root)
    fp_fields = [
        "method",
        "dataset",
        "seed",
        "threshold",
        "num_images",
        "total_gt_components",
        "total_pred_components",
        "false_positive_components",
        "target_near_components_excluded",
        "high_response_fp_components",
        "target_like_hard_clutter_components",
        "hard_clutter_fraction_of_fp",
        "high_response_fraction_of_fp",
        "gt_area_median",
        "summary",
    ]
    write_csv(run_root / "step5_fp_decomposition_table.csv", fp_rows, fp_fields)
    print(f"Wrote {run_root / 'step5_same_domain_table.csv'}")
    print(f"Wrote {run_root / 'step5_hcset_table.csv'}")
    print(f"Wrote {run_root / 'step5_fp_decomposition_table.csv'}")


if __name__ == "__main__":
    main()
