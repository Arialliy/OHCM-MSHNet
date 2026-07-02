#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage import measure

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def read_names(dataset_dir: Path, dataset_name: str, image_list: str | None) -> list[str]:
    if image_list:
        return [line.strip() for line in Path(image_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        line.strip()
        for line in (dataset_dir / "img_idx" / f"test_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def region_iou(region, target_mask: np.ndarray) -> float:
    region_mask = np.zeros(target_mask.shape, dtype=bool)
    region_mask[region.coords[:, 0], region.coords[:, 1]] = True
    intersection = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(intersection, union)


def match_components(pred_mask: np.ndarray, gt_mask: np.ndarray, distance_threshold: float = 3.0) -> tuple[int, int, int]:
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


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def local_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    dilated = binary_dilation(mask, structure=disk(radius))
    ring = dilated & (~mask)
    return ring if ring.any() else ~mask


def values_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std()), float(values.max()), float(np.percentile(values, 95))


def component_features(image: np.ndarray, prob: np.ndarray, mask: np.ndarray, region, ring_radius: int) -> dict:
    comp = component_mask(mask.shape, region.coords)
    ring = local_ring(comp, ring_radius)
    comp_img = image[comp]
    ring_img = image[ring]
    comp_prob = prob[comp]
    ring_prob = prob[ring]
    img_mean, img_std, img_max, img_p95 = values_stats(comp_img)
    ring_mean, ring_std, ring_max, ring_p95 = values_stats(ring_img)
    prob_mean, prob_std, prob_max, prob_p95 = values_stats(comp_prob)
    ring_prob_mean, ring_prob_std, ring_prob_max, ring_prob_p95 = values_stats(ring_prob)
    ring_std = max(ring_std, 1e-6)
    ring_prob_std = max(ring_prob_std, 1e-6)
    return {
        "area": float(region.area),
        "extent": float(region.extent),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "mean_prob": prob_mean,
        "max_prob": prob_max,
        "prob_contrast": prob_mean - ring_prob_mean,
        "prob_peak_contrast": prob_max - ring_prob_p95,
        "prob_z": (prob_mean - ring_prob_mean) / ring_prob_std,
        "prob_peak_z": (prob_max - ring_prob_mean) / ring_prob_std,
        "mean_img": img_mean,
        "max_img": img_max,
        "img_contrast": img_mean - ring_mean,
        "img_peak_contrast": img_max - ring_p95,
        "img_z": (img_mean - ring_mean) / ring_std,
        "img_peak_z": (img_max - ring_mean) / ring_std,
    }


def apply_rule(feature_value: float, op: str, threshold: float) -> bool:
    if op == "ge":
        return feature_value >= threshold
    if op == "le":
        return feature_value <= threshold
    raise ValueError(op)


def update(stats: dict, pred: np.ndarray, gt: np.ndarray) -> None:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)
    stats["inter"] += float(inter)
    stats["union"] += float(union)
    stats["tp"] += float(tp)
    stats["fp"] += float(fp)
    stats["fn"] += float(fn)
    stats["pixels"] += float(pred.size)
    stats["niou_sum"] += safe_div(inter, union)
    stats["count"] += 1
    stats["matched_targets"] += float(matched_targets)
    stats["target_components"] += float(target_components)
    stats["fp_components"] += float(fp_components)


def empty_stats() -> dict:
    return {
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


def stats_row(stats: dict) -> dict:
    precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fa = safe_div(stats["fp"], stats["pixels"])
    return {
        "mIoU": safe_div(stats["inter"], stats["union"]),
        "nIoU": safe_div(stats["niou_sum"], stats["count"]),
        "Pd": safe_div(stats["matched_targets"], stats["target_components"]),
        "FA": fa,
        "FA_ppm": fa * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP_components": stats["fp_components"],
    }


def precompute_cases(args) -> list[dict]:
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    names = read_names(dataset_dir, args.dataset_name, args.image_list)
    exports_dir = Path(args.exports_dir)
    cases = []
    for name in names:
        image = load_gray(find_file(dataset_dir / "images", name))
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        image = image[: gt.shape[0], : gt.shape[1]]
        pred = prob > args.threshold
        comps = []
        for region in connected_regions(pred):
            feats = component_features(image, prob, pred, region, args.ring_radius)
            comps.append({"coords": region.coords.copy(), "features": feats})
        cases.append({"gt": gt, "components": comps})
    return cases


def evaluate_case(args, cases: list[dict], feature: str, op: str, threshold: float) -> dict:
    stats = empty_stats()
    removed_components = 0
    kept_components = 0
    for case in cases:
        gt = case["gt"]
        filtered = np.zeros_like(gt, dtype=bool)
        for component in case["components"]:
            feats = component["features"]
            keep = apply_rule(feats[feature], op, threshold)
            if keep:
                coords = component["coords"]
                filtered[coords[:, 0], coords[:, 1]] = True
                kept_components += 1
            else:
                removed_components += 1
        update(stats, filtered, gt)
    row = stats_row(stats)
    row.update(
        {
            "label": args.label,
            "threshold": args.threshold,
            "feature": feature,
            "op": op,
            "rule_threshold": threshold,
            "removed_components": removed_components,
            "kept_components": kept_components,
            "images": len(cases),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep single-feature component filtering rules.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label", required=True)
    parser.add_argument("--features", default="img_peak_z,img_z,img_contrast,img_peak_contrast,prob_z,prob_peak_z,prob_contrast,prob_peak_contrast,extent,eccentricity,solidity,mean_img,max_img,area")
    parser.add_argument("--ops", default="ge,le")
    parser.add_argument("--rule_thresholds", default="-1,0,0.5,1,1.5,2,2.5,3,4,5,7.5,10,15,20,30,50,75,100")
    parser.add_argument("--ring_radius", type=int, default=9)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    features = [item.strip() for item in args.features.split(",") if item.strip()]
    ops = [item.strip() for item in args.ops.split(",") if item.strip()]
    thresholds = [float(item) for item in args.rule_thresholds.split(",") if item.strip()]
    cases = precompute_cases(args)
    rows = []
    for feature in features:
        for op in ops:
            for threshold in thresholds:
                rows.append(evaluate_case(args, cases, feature, op, threshold))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "images", "threshold", "feature", "op", "rule_threshold",
        "mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1",
        "FP_components", "removed_components", "kept_components",
    ]
    with (output_dir / "feature_filter_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "feature_filter_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output_dir / "feature_filter_sweep.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
