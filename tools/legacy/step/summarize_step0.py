#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]


def main():
    parser = argparse.ArgumentParser(description="Summarize Step0 seed results.")
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summaries = sorted(run_root.glob("*/seed_*/exports/summary_metrics.json"))
    rows = []
    grouped = defaultdict(list)

    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = data["dataset"]
        seed = data["seed"]
        metrics = data["metrics_at_threshold"]
        row = {"dataset": dataset, "seed": seed, "summary": str(path)}
        row.update({metric: metrics[metric] for metric in METRICS})
        rows.append(row)
        grouped[dataset].append(row)

    detail_path = run_root / "step0_seed_details.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", *METRICS, "summary"])
        writer.writeheader()
        writer.writerows(rows)

    mean_rows = []
    for dataset, items in sorted(grouped.items()):
        row = {"dataset": dataset, "num_seeds": len(items)}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        mean_rows.append(row)

    summary_path = run_root / "step0_mean_std.csv"
    fieldnames = ["dataset", "num_seeds"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mean_rows)

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
