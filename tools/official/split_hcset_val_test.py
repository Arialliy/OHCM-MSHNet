#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def split_dataset(rows: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda r: (-float(r.get("hc_score", 0.0)), r.get("image_name", "")))
    if len(ordered) <= 1:
        return ordered, []
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    n_val = min(n_val, len(ordered) - 1)

    val = []
    test = []
    for idx, row in enumerate(ordered):
        target = val if idx % max(2, int(math.floor(1.0 / max(1e-6, val_fraction)))) == 0 and len(val) < n_val else test
        target.append(row)
    for row in ordered:
        if len(val) >= n_val:
            break
        if row not in val:
            val.append(row)
            if row in test:
                test.remove(row)
    return val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a fixed MSHNet-defined HC-Set into HC-Val and HC-Test lists.")
    parser.add_argument("--hcset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.35)
    args = parser.parse_args()

    hcset_dir = Path(args.hcset_dir)
    output_dir = Path(args.output_dir)
    image_rows = read_csv(hcset_dir / "hcset_images.csv")
    component_rows = read_csv(hcset_dir / "hcset_components.csv")

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in image_rows:
        by_dataset[row["dataset"]].append(row)

    subset_images = []
    summary = {
        "source_hcset_dir": str(hcset_dir),
        "protocol": "deterministic split of the existing MSHNet seed42-defined HC-Set",
        "limitation": "The dataset has train/test indices but no independent validation index; this is a locked HC-Set split, not a newly mined validation set.",
        "val_fraction": args.val_fraction,
        "datasets": {},
    }
    image_to_subset = {}

    for dataset, rows in sorted(by_dataset.items()):
        val_rows, test_rows = split_dataset(rows, args.val_fraction)
        val_names = [row["image_name"] for row in val_rows]
        test_names = [row["image_name"] for row in test_rows]
        write_list(output_dir / f"hcval_{dataset}.txt", val_names)
        write_list(output_dir / f"hctest_{dataset}.txt", test_names)

        for subset, subset_rows in [("hcval", val_rows), ("hctest", test_rows)]:
            for row in subset_rows:
                out = dict(row)
                out["hc_subset"] = subset
                subset_images.append(out)
                image_to_subset[(dataset, row["image_name"])] = subset

        summary["datasets"][dataset] = {
            "hcval_images": len(val_rows),
            "hctest_images": len(test_rows),
            "hcval_list": str(output_dir / f"hcval_{dataset}.txt"),
            "hctest_list": str(output_dir / f"hctest_{dataset}.txt"),
        }

    subset_components = []
    for row in component_rows:
        dataset = row.get("hcset_dataset") or row.get("dataset")
        subset = image_to_subset.get((dataset, row.get("image_name", "")), "")
        if not subset:
            continue
        out = dict(row)
        out["hc_subset"] = subset
        subset_components.append(out)

    image_fields = ["hc_subset", *image_rows[0].keys()] if image_rows else ["hc_subset"]
    component_fields = ["hc_subset", *component_rows[0].keys()] if component_rows else ["hc_subset"]
    write_csv(output_dir / "hc_val_test_images.csv", subset_images, image_fields)
    write_csv(output_dir / "hc_val_test_components.csv", subset_components, component_fields)
    (output_dir / "hc_val_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
