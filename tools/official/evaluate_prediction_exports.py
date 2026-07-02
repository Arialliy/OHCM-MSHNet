#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure


IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")


def load_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def connected_regions(mask):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def region_iou(region, target_mask):
    region_mask = np.zeros(target_mask.shape, dtype=bool)
    region_mask[region.coords[:, 0], region.coords[:, 1]] = True
    intersection = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(intersection, union)


def match_components(pred_mask, gt_mask, distance_threshold=3.0):
    pred_regions = connected_regions(pred_mask)
    gt_regions = connected_regions(gt_mask)
    used_pred = set()
    matched_targets = 0

    for gt_region in gt_regions:
        gt_centroid = np.asarray(gt_region.centroid)
        for pred_idx, pred_region in enumerate(pred_regions):
            if pred_idx in used_pred:
                continue
            pred_centroid = np.asarray(pred_region.centroid)
            if np.linalg.norm(pred_centroid - gt_centroid) < distance_threshold:
                used_pred.add(pred_idx)
                matched_targets += 1
                break

    fp_components = 0
    for pred_idx, pred_region in enumerate(pred_regions):
        if pred_idx in used_pred:
            continue
        if region_iou(pred_region, gt_mask) <= 0:
            fp_components += 1

    return matched_targets, len(gt_regions), fp_components


def init_fp_census():
    return {
        "matched_target_components": 0.0,
        "boundary_excess_instances": 0.0,
        "boundary_excess_pixel_mass": 0.0,
        "detached_near_fp_components": 0.0,
        "detached_near_fp_pixel_mass": 0.0,
        "far_fp_components": 0.0,
        "far_fp_pixel_mass": 0.0,
        "unmatched_fp_components": 0.0,
        "fp_pixel_mass": 0.0,
        "confidence_mass": 0.0,
        "boundary_excess_confidence_mass": 0.0,
        "detached_near_fp_confidence_mass": 0.0,
        "far_fp_confidence_mass": 0.0,
    }


def update_fp_census(census, component_rows, image_name, prob, pred_mask, gt_mask, threshold):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)
    for idx, region in enumerate(connected_regions(pred), start=1):
        component = np.zeros_like(pred, dtype=bool)
        component[region.coords[:, 0], region.coords[:, 1]] = True
        overlaps_gt = bool(np.logical_and(component, gt).any())
        fp_mask = component & (~gt)
        if overlaps_gt:
            census["matched_target_components"] += 1.0
        if not fp_mask.any():
            continue

        if overlaps_gt:
            category = "boundary_excess"
            min_distance = 0.0
            unmatched = 0
            census["boundary_excess_instances"] += 1.0
        else:
            min_distance = float(dist_to_gt[fp_mask].min())
            if min_distance <= 10.0:
                category = "detached_near_fp"
                census["detached_near_fp_components"] += 1.0
            else:
                category = "far_fp"
                census["far_fp_components"] += 1.0
            unmatched = 1
            census["unmatched_fp_components"] += 1.0

        fp_pixels = float(fp_mask.sum())
        confidence = float(prob[fp_mask].sum())
        census["fp_pixel_mass"] += fp_pixels
        census["confidence_mass"] += confidence
        census[f"{category}_pixel_mass"] += fp_pixels
        census[f"{category}_confidence_mass"] += confidence
        cy, cx = (float(v) for v in region.centroid)
        component_rows.append({
            "image_name": image_name,
            "threshold": threshold,
            "component_id": idx,
            "category": category,
            "matched_target_component": int(overlaps_gt),
            "unmatched_fp_component": unmatched,
            "component_area": int(region.area),
            "fp_pixel_mass": int(fp_pixels),
            "mean_probability": float(prob[fp_mask].mean()),
            "max_probability": float(prob[fp_mask].max()),
            "confidence_mass": confidence,
            "minimum_distance_to_gt": min_distance,
            "component_center_y": cy,
            "component_center_x": cx,
            "bbox_y0": int(region.bbox[0]),
            "bbox_y1": int(region.bbox[2]),
            "bbox_x0": int(region.bbox[1]),
            "bbox_x1": int(region.bbox[3]),
        })


def finalize_fp_census(census):
    total_fp_pixels = census["fp_pixel_mass"]
    total_confidence = census["confidence_mass"]
    near_pixels = census["boundary_excess_pixel_mass"] + census["detached_near_fp_pixel_mass"]
    near_confidence = census["boundary_excess_confidence_mass"] + census["detached_near_fp_confidence_mass"]
    unmatched = census["unmatched_fp_components"]
    census["target_near_pixel_mass"] = near_pixels
    census["target_near_confidence_mass"] = near_confidence
    census["R_near_pixel"] = near_pixels / max(1.0, total_fp_pixels)
    census["R_near_component"] = census["detached_near_fp_components"] / max(1.0, unmatched)
    census["R_near_confidence"] = near_confidence / max(1e-12, total_confidence)
    return census


def update_stats(stats, prob, gt_mask, threshold):
    pred = prob > threshold
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)

    item = stats[threshold]
    item["inter"] += float(inter)
    item["union"] += float(union)
    item["tp"] += float(tp)
    item["fp"] += float(fp)
    item["fn"] += float(fn)
    item["pixels"] += float(pred.size)
    item["niou_sum"] += safe_div(inter, union)
    item["count"] += 1
    item["matched_targets"] += float(matched_targets)
    item["target_components"] += float(target_components)
    item["fp_components"] += float(fp_components)


def stats_rows(stats):
    rows = []
    for threshold in sorted(stats):
        item = stats[threshold]
        precision = safe_div(item["tp"], item["tp"] + item["fp"])
        recall = safe_div(item["tp"], item["tp"] + item["fn"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        fa = safe_div(item["fp"], item["pixels"])
        rows.append(
            {
                "threshold": threshold,
                "mIoU": safe_div(item["inter"], item["union"]),
                "nIoU": safe_div(item["niou_sum"], item["count"]),
                "Pd": safe_div(item["matched_targets"], item["target_components"]),
                "FA": fa,
                "FA_ppm": fa * 1_000_000.0,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "FP_components": item["fp_components"],
            }
        )
    return rows


def image_metrics(image_name, prob, pred, gt, threshold):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "image_name": image_name,
        "threshold": threshold,
        "IoU": safe_div(inter, union),
        "nIoU": safe_div(inter, union),
        "Pd": safe_div(matched_targets, target_components),
        "FA": safe_div(fp, pred.size),
        "FA_ppm": safe_div(fp, pred.size) * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": safe_div(2.0 * precision * recall, precision + recall),
        "FP_components": fp_components,
        "target_components": target_components,
        "target_area": int(gt.sum()),
        "mean_prob_target": float(prob[gt].mean()) if gt.any() else 0.0,
        "mean_prob_bg": float(prob[~gt].mean()) if (~gt).any() else 0.0,
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def require_direct_export_parity(exports_dir: Path, image_list: str | None) -> dict:
    summary_path = exports_dir / "direct_export_parity" / "direct_export_parity_summary.json"
    if not summary_path.exists():
        raise SystemExit(
            "Direct/export parity gate is missing; refusing to evaluate exports. "
            f"Expected {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("pass") is not True:
        errors.append("parity pass is not true")
    payload_exports = payload.get("exports_dir")
    if not payload_exports:
        errors.append("parity exports_dir is missing")
    elif resolved(payload_exports) != resolved(exports_dir):
        errors.append(f"parity exports_dir {payload_exports!r} does not match {str(exports_dir)!r}")

    payload_image_list = payload.get("image_list")
    if image_list:
        if payload_image_list and resolved(payload_image_list) != resolved(image_list):
            errors.append(f"parity image_list {payload_image_list!r} does not match {image_list!r}")
    elif payload_image_list:
        errors.append("parity gate used an image_list but evaluation is running on the full split")

    checks = payload.get("checks", {})
    for name in (
        "max_prob_diff",
        "mask_diff_pixels",
        "mIoU_diff",
        "Pd_diff",
        "FA_ppm_diff",
        "direct_target_gt_background",
        "export_target_gt_background",
    ):
        if checks.get(name) is not True:
            errors.append(f"parity check {name} is not PASS")

    if errors:
        raise SystemExit("Direct/export parity gate failed; evaluation is blocked:\n- " + "\n- ".join(errors))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate exported probability maps on full test set or a fixed subset.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--method", default="")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--skip_direct_export_parity_gate", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    exports_dir = Path(args.exports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_payload = None
    if not args.skip_direct_export_parity_gate:
        parity_payload = require_direct_export_parity(exports_dir, args.image_list)

    if args.image_list:
        image_names = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]
        subset_name = Path(args.image_list).stem
    else:
        test_list = dataset_dir / "img_idx" / f"test_{args.dataset_name}.txt"
        image_names = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
        subset_name = "full_test"

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    threshold_stats = {
        threshold: {
            "inter": 0.0,
            "union": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pixels": 0.0,
            "niou_sum": 0.0,
            "count": 0,
            "matched_targets": 0.0,
            "target_components": 0.0,
            "fp_components": 0.0,
        }
        for threshold in thresholds
    }

    per_image_rows = []
    fp_census = init_fp_census()
    fp_component_rows = []
    for name in image_names:
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)
        gt = load_mask(find_file(dataset_dir / "masks", name))
        h = min(prob.shape[0], gt.shape[0])
        w = min(prob.shape[1], gt.shape[1])
        prob = prob[:h, :w]
        gt = gt[:h, :w]
        pred = prob > args.threshold
        per_image_rows.append(image_metrics(name, prob, pred, gt, args.threshold))
        update_fp_census(fp_census, fp_component_rows, name, prob, pred, gt, args.threshold)
        for threshold in thresholds:
            update_stats(threshold_stats, prob, gt, threshold)

    per_image_fields = [
        "image_name",
        "threshold",
        "IoU",
        "nIoU",
        "Pd",
        "FA",
        "FA_ppm",
        "Precision",
        "Recall",
        "F1",
        "FP_components",
        "target_components",
        "target_area",
        "mean_prob_target",
        "mean_prob_bg",
    ]
    write_csv(output_dir / "metrics_per_image.csv", per_image_rows, per_image_fields)
    fp_component_fields = [
        "image_name",
        "threshold",
        "component_id",
        "category",
        "matched_target_component",
        "unmatched_fp_component",
        "component_area",
        "fp_pixel_mass",
        "mean_probability",
        "max_probability",
        "confidence_mass",
        "minimum_distance_to_gt",
        "component_center_y",
        "component_center_x",
        "bbox_y0",
        "bbox_y1",
        "bbox_x0",
        "bbox_x1",
    ]
    write_csv(output_dir / "fp_components.csv", fp_component_rows, fp_component_fields)
    rows = stats_rows(threshold_stats)
    threshold_fields = ["threshold", "mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]
    write_csv(output_dir / "threshold_curve.csv", rows, threshold_fields)

    summary = {
        "dataset": args.dataset_name,
        "train_dataset": args.train_dataset_name or args.dataset_name,
        "method": args.method,
        "seed": args.seed,
        "subset": subset_name,
        "image_list": os.path.abspath(args.image_list) if args.image_list else None,
        "num_images": len(image_names),
        "threshold": args.threshold,
        "metrics_at_threshold": next(row for row in rows if abs(row["threshold"] - args.threshold) < 1e-9),
        "outputs": {
            "per_image_metrics": str(output_dir / "metrics_per_image.csv"),
            "threshold_curve": str(output_dir / "threshold_curve.csv"),
            "fp_components": str(output_dir / "fp_components.csv"),
        },
        "fp_census_at_threshold": finalize_fp_census(fp_census),
        "direct_export_parity": parity_payload,
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics_at_threshold"], indent=2), flush=True)


if __name__ == "__main__":
    main()
