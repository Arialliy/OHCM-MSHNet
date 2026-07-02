#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
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


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize Step1 hard-clutter diagnosis outputs.")
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary_paths = sorted(run_root.glob("*/seed_*/step1/step1_summary.json"))
    rows = []
    type_counts = defaultdict(int)

    for path in summary_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {field: data.get(field, "") for field in SUMMARY_FIELDS if field != "summary"}
        row["summary"] = str(path)
        rows.append(row)
        for clutter_type, count in data.get("clutter_type_counts", {}).items():
            key = (data.get("dataset", ""), data.get("seed", ""), clutter_type)
            type_counts[key] += int(count)

    write_csv(run_root / "step1_summary_table.csv", rows, SUMMARY_FIELDS)

    type_rows = []
    for (dataset, seed, clutter_type), count in sorted(type_counts.items()):
        type_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "clutter_type": clutter_type,
                "count": count,
            }
        )
    write_csv(run_root / "step1_type_count_table.csv", type_rows, ["dataset", "seed", "clutter_type", "count"])

    print(f"Wrote {run_root / 'step1_summary_table.csv'}")
    print(f"Wrote {run_root / 'step1_type_count_table.csv'}")


if __name__ == "__main__":
    main()
