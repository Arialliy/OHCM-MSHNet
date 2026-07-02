#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from utils import get_img_norm_cfg
from utils.component_geometry import (
    build_center_heatmap,
    build_core_boundary_maps,
    component_area_bins,
    dilate_binary,
)
from utils.local_peak import select_background_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CGA component geometry targets before training.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--center_positive_thr", type=float, default=0.1)
    parser.add_argument("--scale_bins", type=str, default="4,9,16,36")
    parser.add_argument("--boundary_radius", type=int, default=2)
    parser.add_argument("--peak_min_k", type=int, default=8)
    parser.add_argument("--peak_max_k", type=int, default=256)
    parser.add_argument("--peak_topk_ratio", type=float, default=0.001)
    parser.add_argument("--peak_dilate_radius", type=int, default=3)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def split_ids(args, dataset):
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = tuple(int(x) for x in args.scale_bins.split(",") if x.strip())

    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, split_source = split_ids(args, dataset)
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    rows = []
    totals = {
        "images_with_targets": 0,
        "images_with_target_but_no_center": 0,
        "center_peak_count_total": 0,
        "target_leakage_pixels": 0,
    }
    center_positive_ratios = []
    boundary_pixels = []
    local_peak_counts = []
    scale_bin_counts = np.zeros((len(bins),), dtype=np.int64)

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            img = img[:, :, :h, :w].float()
            gt = gt_mask[:, :, :h, :w].float()
            has_target = bool((gt > 0.5).any().item())
            if has_target:
                totals["images_with_targets"] += 1

            center, _scale_map, _scale_valid = build_center_heatmap(gt)
            peak_count = int((center >= 0.999).sum().item())
            center_positive_ratio = float((center > args.center_positive_thr).float().mean().item())
            _core, boundary, _ignore = build_core_boundary_maps(gt, boundary_radius=args.boundary_radius)
            scale_target, scale_valid, counts = component_area_bins(gt, bins=bins)
            peaks = select_background_peaks(
                img,
                gt,
                topk_ratio=args.peak_topk_ratio,
                min_k=args.peak_min_k,
                max_k=args.peak_max_k,
                dilate_radius=args.peak_dilate_radius,
            )
            target_dilated = dilate_binary(gt, args.peak_dilate_radius) > 0.5
            leakage = int((peaks & target_dilated).sum().item())

            if has_target and peak_count <= 0:
                totals["images_with_target_but_no_center"] += 1
            totals["center_peak_count_total"] += peak_count
            totals["target_leakage_pixels"] += leakage
            center_positive_ratios.append(center_positive_ratio)
            boundary_count = int(boundary.sum().item())
            boundary_pixels.append(boundary_count)
            local_peak_count = int(peaks.sum().item())
            local_peak_counts.append(local_peak_count)
            scale_bin_counts += counts.detach().cpu().numpy()

            rows.append(
                {
                    "image_id": name,
                    "has_target": int(has_target),
                    "center_peak_count": peak_count,
                    "center_heatmap_positive_ratio": center_positive_ratio,
                    "boundary_pixels": boundary_count,
                    "local_bg_peak_count": local_peak_count,
                    "target_leakage_pixels": leakage,
                    "scale_valid_pixels": int(scale_valid.sum().item()),
                    "scale_bin": int(scale_target[scale_valid[:, 0]].float().mean().item()) if scale_valid.any() else -1,
                }
            )
            if (idx + 1) % 100 == 0:
                print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    total_components = int(scale_bin_counts.sum())
    max_scale_bin_ratio = safe_div(int(scale_bin_counts.max()) if len(scale_bin_counts) else 0, total_components)
    summary = {
        "gate_pass": False,
        "fail_reasons": [],
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "num_images": len(rows),
        "images_with_targets": totals["images_with_targets"],
        "images_with_target_but_no_center": totals["images_with_target_but_no_center"],
        "center_peak_count_total": totals["center_peak_count_total"],
        "center_heatmap_positive_ratio_mean": float(np.mean(center_positive_ratios)) if center_positive_ratios else 0.0,
        "scale_bin_counts": [int(v) for v in scale_bin_counts.tolist()],
        "scale_bin_max_ratio": max_scale_bin_ratio,
        "boundary_pixels_mean": float(np.mean(boundary_pixels)) if boundary_pixels else 0.0,
        "local_bg_peak_count_mean": float(np.mean(local_peak_counts)) if local_peak_counts else 0.0,
        "target_leakage_pixels": totals["target_leakage_pixels"],
        "thresholds": {
            "peak_min_k": args.peak_min_k,
            "scale_bin_max_ratio": 0.85,
        },
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
        },
    }

    if summary["center_peak_count_total"] <= 0:
        summary["fail_reasons"].append("center_targets_empty")
    if summary["images_with_target_but_no_center"] != 0:
        summary["fail_reasons"].append("images_with_target_but_no_center")
    if summary["local_bg_peak_count_mean"] < args.peak_min_k:
        summary["fail_reasons"].append("local_bg_peak_count_too_low")
    if summary["scale_bin_max_ratio"] >= 0.85:
        summary["fail_reasons"].append("scale_bin_distribution_too_single")
    if summary["boundary_pixels_mean"] <= 0:
        summary["fail_reasons"].append("boundary_targets_empty")
    if summary["target_leakage_pixels"] != 0:
        summary["fail_reasons"].append("target_leakage_pixels_nonzero")
    summary["gate_pass"] = len(summary["fail_reasons"]) == 0

    write_csv(
        output_dir / "per_image.csv",
        rows,
        [
            "image_id",
            "has_target",
            "center_peak_count",
            "center_heatmap_positive_ratio",
            "boundary_pixels",
            "local_bg_peak_count",
            "target_leakage_pixels",
            "scale_valid_pixels",
            "scale_bin",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
