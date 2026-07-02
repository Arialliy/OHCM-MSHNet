#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]


def read_summary(path: Path, method: str, split: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics_at_threshold"]
    row = {
        "method": method,
        "dataset": data.get("dataset", ""),
        "seed": data.get("seed", ""),
        "split": split,
        "num_images": data.get("num_images", ""),
        "summary": str(path),
    }
    row.update({metric: metrics.get(metric, "") for metric in METRICS})
    return row


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize Step3 gate results.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--baseline_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step0_baseline/20260611_155232")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    baseline_root = Path(args.baseline_root)
    rows = []

    baseline_full = baseline_root / "NUDT-SIRST" / "seed_42" / "exports" / "summary_metrics.json"
    baseline_hc = baseline_root / "step2_eval" / "MSHNet" / "NUDT-SIRST" / "seed_42" / "hcset" / "summary_metrics.json"
    if baseline_full.exists():
        rows.append(read_summary(baseline_full, "MSHNet", "full"))
    if baseline_hc.exists():
        rows.append(read_summary(baseline_hc, "MSHNet", "hcset"))

    for method_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        method = method_dir.name
        for path in sorted(method_dir.glob("*/seed_*/eval_full/summary_metrics.json")):
            rows.append(read_summary(path, method, "full"))
        for path in sorted(method_dir.glob("*/seed_*/eval_hcset/summary_metrics.json")):
            rows.append(read_summary(path, method, "hcset"))

    fields = ["method", "dataset", "seed", "split", "num_images", *METRICS, "summary"]
    write_csv(run_root / "step3_gate_table.csv", rows, fields)

    fp_rows = []
    baseline_step1 = baseline_root / "NUDT-SIRST" / "seed_42" / "step1" / "step1_summary.json"
    if baseline_step1.exists():
        data = json.loads(baseline_step1.read_text(encoding="utf-8"))
        fp_rows.append(
            {
                "method": "MSHNet",
                "dataset": data.get("dataset", ""),
                "seed": data.get("seed", ""),
                "false_positive_components": data.get("false_positive_components", ""),
                "target_like_hard_clutter_components": data.get("target_like_hard_clutter_components", ""),
                "hard_clutter_fraction_of_fp": data.get("hard_clutter_fraction_of_fp", ""),
                "summary": str(baseline_step1),
            }
        )

    for path in sorted(run_root.glob("*/*/seed_*/fp_analysis/step1_summary.json")):
        method = path.relative_to(run_root).parts[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        fp_rows.append(
            {
                "method": method,
                "dataset": data.get("dataset", ""),
                "seed": data.get("seed", ""),
                "false_positive_components": data.get("false_positive_components", ""),
                "target_like_hard_clutter_components": data.get("target_like_hard_clutter_components", ""),
                "hard_clutter_fraction_of_fp": data.get("hard_clutter_fraction_of_fp", ""),
                "summary": str(path),
            }
        )
    write_csv(
        run_root / "step3_fp_table.csv",
        fp_rows,
        ["method", "dataset", "seed", "false_positive_components", "target_like_hard_clutter_components", "hard_clutter_fraction_of_fp", "summary"],
    )

    print(f"Wrote {run_root / 'step3_gate_table.csv'}")
    print(f"Wrote {run_root / 'step3_fp_table.csv'}")


if __name__ == "__main__":
    main()
