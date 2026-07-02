#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
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


def matched_region_indices(pred_regions, gt_mask: np.ndarray, distance_threshold: float = 3.0) -> set[int]:
    gt_regions = connected_regions(gt_mask)
    used_pred: set[int] = set()
    for gt_region in gt_regions:
        gt_centroid = np.asarray(gt_region.centroid)
        for pred_idx, pred_region in enumerate(pred_regions):
            if pred_idx in used_pred:
                continue
            pred_centroid = np.asarray(pred_region.centroid)
            if np.linalg.norm(pred_centroid - gt_centroid) < distance_threshold:
                used_pred.add(pred_idx)
                break
    return used_pred


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def local_ring(mask: np.ndarray, radius: int, gt: np.ndarray | None = None) -> np.ndarray:
    dilated = binary_dilation(mask, structure=disk(radius))
    ring = dilated & (~mask)
    if gt is not None:
        ring = ring & (~gt)
    if not ring.any():
        ring = ~mask
    return ring


def stats_for_values(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std()), float(values.max()), float(np.percentile(values, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract component-level image/probability/shape features.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ring_radius", type=int, default=9)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    exports_dir = Path(args.exports_dir)
    names = read_names(dataset_dir, args.dataset_name, args.image_list)
    rows = []

    for name in names:
        image = load_gray(find_file(dataset_dir / "images", name))
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        image = image[: gt.shape[0], : gt.shape[1]]
        pred = prob > args.threshold
        pred_regions = connected_regions(pred)
        matched = matched_region_indices(pred_regions, gt)
        dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)

        for idx, region in enumerate(pred_regions):
            comp = component_mask(gt.shape, region.coords)
            fp_mask = comp & (~gt)
            overlaps_gt = bool(np.logical_and(comp, gt).any())
            if idx in matched:
                category = "matched_target"
            elif overlaps_gt:
                category = "boundary_touching_unmatched"
            else:
                min_dist = float(dist_to_gt[comp].min()) if comp.any() else math.inf
                category = "detached_near_fp" if min_dist <= 10.0 else "far_fp"

            ring = local_ring(comp, args.ring_radius, gt)
            comp_img = image[comp]
            ring_img = image[ring]
            comp_prob = prob[comp]
            ring_prob = prob[ring]
            img_mean, img_std, img_max, img_p95 = stats_for_values(comp_img)
            ring_mean, ring_std, ring_max, ring_p95 = stats_for_values(ring_img)
            prob_mean, prob_std, prob_max, prob_p95 = stats_for_values(comp_prob)
            ring_prob_mean, ring_prob_std, ring_prob_max, ring_prob_p95 = stats_for_values(ring_prob)
            ring_std_eps = max(ring_std, 1e-6)
            prob_ring_std_eps = max(ring_prob_std, 1e-6)
            y0, x0, y1, x1 = region.bbox
            height = y1 - y0
            width = x1 - x0
            min_dist_fp = float(dist_to_gt[fp_mask].min()) if fp_mask.any() else 0.0

            rows.append(
                {
                    "label": args.label,
                    "image_id": name,
                    "component_id": idx + 1,
                    "category": category,
                    "matched_target": int(idx in matched),
                    "overlaps_gt": int(overlaps_gt),
                    "component_area": int(region.area),
                    "fp_pixel_mass": int(fp_mask.sum()),
                    "bbox_h": int(height),
                    "bbox_w": int(width),
                    "bbox_area": int(height * width),
                    "extent": float(region.extent),
                    "eccentricity": float(region.eccentricity),
                    "solidity": float(region.solidity),
                    "mean_prob": prob_mean,
                    "std_prob": prob_std,
                    "max_prob": prob_max,
                    "p95_prob": prob_p95,
                    "ring_mean_prob": ring_prob_mean,
                    "ring_std_prob": ring_prob_std,
                    "prob_contrast": prob_mean - ring_prob_mean,
                    "prob_peak_contrast": prob_max - ring_prob_p95,
                    "prob_z": (prob_mean - ring_prob_mean) / prob_ring_std_eps,
                    "prob_peak_z": (prob_max - ring_prob_mean) / prob_ring_std_eps,
                    "mean_img": img_mean,
                    "std_img": img_std,
                    "max_img": img_max,
                    "p95_img": img_p95,
                    "ring_mean_img": ring_mean,
                    "ring_std_img": ring_std,
                    "ring_max_img": ring_max,
                    "ring_p95_img": ring_p95,
                    "img_contrast": img_mean - ring_mean,
                    "img_peak_contrast": img_max - ring_p95,
                    "img_z": (img_mean - ring_mean) / ring_std_eps,
                    "img_peak_z": (img_max - ring_mean) / ring_std_eps,
                    "minimum_distance_to_gt": min_dist_fp,
                    "bbox_y0": int(y0),
                    "bbox_y1": int(y1),
                    "bbox_x0": int(x0),
                    "bbox_x1": int(x1),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (output_dir / "component_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    summary = {"label": args.label, "images": len(names), "components": len(rows), "category_counts": counts}
    (output_dir / "component_features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
