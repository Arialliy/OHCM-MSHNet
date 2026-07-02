#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(row, key):
    try:
        return float(row.get(key, 0.0))
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Build fixed Hard-Clutter Evaluation Subsets from Step1 baseline diagnostics.")
    parser.add_argument("--step1_root", required=True, help="Root containing DATASET/seed_SEED/step1/step1_fp_components.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_mean_prob", type=float, default=0.5)
    parser.add_argument("--min_scale_similarity", type=float, default=0.35)
    parser.add_argument("--min_local_contrast_z", type=float, default=0.5)
    parser.add_argument("--allow_weak_if_empty", action="store_true", default=True)
    args = parser.parse_args()

    step1_root = Path(args.step1_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_component_rows = []
    image_rows = []
    summary = {
        "source_step1_root": str(step1_root),
        "source_seed": args.seed,
        "rules": {
            "clutter_type": "target_like_hard_clutter",
            "min_mean_prob": args.min_mean_prob,
            "min_scale_similarity": args.min_scale_similarity,
            "min_local_contrast_z": args.min_local_contrast_z,
        },
        "datasets": {},
    }

    for dataset in args.datasets:
        fp_path = step1_root / dataset / f"seed_{args.seed}" / "step1" / "step1_fp_components.csv"
        rows = read_csv(fp_path)
        selected = []
        for row in rows:
            if row.get("clutter_type") != "target_like_hard_clutter":
                continue
            if safe_float(row, "mean_prob") < args.min_mean_prob:
                continue
            if safe_float(row, "scale_similarity") < args.min_scale_similarity:
                continue
            if safe_float(row, "local_contrast_z") < args.min_local_contrast_z:
                continue
            selected.append(row)

        if not selected and args.allow_weak_if_empty:
            selected = [row for row in rows if row.get("clutter_type") == "target_like_hard_clutter"]

        grouped = defaultdict(list)
        for row in selected:
            grouped[row["image_name"]].append(row)
            out_row = dict(row)
            out_row["hcset_dataset"] = dataset
            all_component_rows.append(out_row)

        dataset_image_rows = []
        for image_name, items in grouped.items():
            target_like_count = len(items)
            max_prob = max(safe_float(item, "mean_prob") for item in items)
            max_score = max(
                safe_float(item, "mean_prob")
                * safe_float(item, "scale_similarity")
                * max(0.0, safe_float(item, "local_contrast_z"))
                for item in items
            )
            dataset_image_rows.append(
                {
                    "dataset": dataset,
                    "seed": args.seed,
                    "image_name": image_name,
                    "target_like_fp_components": target_like_count,
                    "max_mean_prob": max_prob,
                    "hc_score": max_score,
                }
            )

        dataset_image_rows.sort(key=lambda row: (float(row["hc_score"]), row["image_name"]), reverse=True)
        image_rows.extend(dataset_image_rows)
        image_list_path = output_dir / f"hcset_{dataset}.txt"
        image_list_path.write_text("\n".join(row["image_name"] for row in dataset_image_rows) + ("\n" if dataset_image_rows else ""))

        summary["datasets"][dataset] = {
            "num_images": len(dataset_image_rows),
            "num_components": len(selected),
            "image_list": str(image_list_path),
        }

    component_fields = [
        "hcset_dataset",
        "dataset",
        "seed",
        "image_index",
        "image_name",
        "component_id",
        "area",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "centroid_x",
        "centroid_y",
        "mean_prob",
        "max_prob",
        "mean_logit",
        "max_logit",
        "local_inside_mean",
        "local_ring_mean",
        "local_contrast",
        "local_contrast_z",
        "distance_to_target",
        "nearest_gt_area",
        "gt_area_median",
        "scale_similarity",
        "multi_scale_response",
        "max_iou_to_gt",
        "clutter_type",
        "manual_label",
        "is_high_response",
    ]
    image_fields = ["dataset", "seed", "image_name", "target_like_fp_components", "max_mean_prob", "hc_score"]
    write_csv(output_dir / "hcset_components.csv", all_component_rows, component_fields)
    write_csv(output_dir / "hcset_images.csv", image_rows, image_fields)
    (output_dir / "hcset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
