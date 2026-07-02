#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
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


def filter_components(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    pred = prob > threshold
    if min_area <= 1:
        return pred
    keep = np.zeros_like(pred, dtype=bool)
    for region in connected_regions(pred):
        if int(region.area) >= min_area:
            keep[region.coords[:, 0], region.coords[:, 1]] = True
    return keep


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


def stats_row(stats: dict, threshold: float, min_area: int) -> dict:
    precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fa = safe_div(stats["fp"], stats["pixels"])
    return {
        "threshold": threshold,
        "min_area": min_area,
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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep connected-component area filtering for exported probability maps.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_areas", default="1,2,3,5,8,10,15,20,30,50")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    names = read_names(dataset_dir, args.dataset_name, args.image_list)
    exports_dir = Path(args.exports_dir)
    min_areas = [int(item) for item in args.min_areas.split(",") if item.strip()]
    rows = []

    for min_area in min_areas:
        stats = empty_stats()
        for name in names:
            gt = load_mask(find_file(dataset_dir / "masks", name))
            prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
            pred = filter_components(prob, args.threshold, min_area)
            update(stats, pred, gt)
        row = stats_row(stats, args.threshold, min_area)
        row["label"] = args.label
        row["images"] = len(names)
        rows.append(row)

    output_dir = Path(args.output_dir)
    fields = [
        "label", "images", "threshold", "min_area", "mIoU", "nIoU", "Pd", "FA", "FA_ppm",
        "Precision", "Recall", "F1", "FP_components",
    ]
    write_csv(output_dir / "component_filter_sweep.csv", rows, fields)
    (output_dir / "component_filter_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
