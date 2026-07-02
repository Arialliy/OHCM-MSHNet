#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
BOUNDARY_EXCESS = "Boundary excess"
DETACHED_NEAR_FP = "Detached near-FP"
FAR_FP = "Far-FP"
MATCHED_TARGET_COMPONENT = "Matched target component"
REQUIRED_EXPORT_CHECKS = (
    "export_vs_summary",
    "curve_0p5_vs_summary",
    "gt_mean_gt_bg",
    "direct_vs_export_metrics",
    "direct_export_max_diff",
)


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


def read_curve(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
        return rows


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_export_gate(args) -> dict:
    validation_path = Path(args.validation_json)
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("status") != "PASS":
        errors.append(f"validation status is {payload.get('status')!r}, expected PASS")
    checks = payload.get("checks", {})
    for name in REQUIRED_EXPORT_CHECKS:
        if checks.get(name) is not True:
            errors.append(f"validation check {name} is not PASS")

    validation_exports = payload.get("exports_dir")
    if not validation_exports:
        errors.append("validation exports_dir is missing")
    elif resolved(validation_exports) != resolved(args.exports_dir):
        errors.append(
            f"validation exports_dir {validation_exports!r} does not match census exports_dir {args.exports_dir!r}"
        )

    validation_image_list = payload.get("image_list")
    if args.image_list:
        if not validation_image_list:
            errors.append("census uses an image_list but validation image_list is missing")
        elif resolved(validation_image_list) != resolved(args.image_list):
            errors.append(
                f"validation image_list {validation_image_list!r} does not match census image_list {args.image_list!r}"
            )
    elif validation_image_list:
        errors.append("validation used an image_list but census is running on the full split")

    if errors:
        raise SystemExit("Export validation gate failed; census is blocked:\n- " + "\n- ".join(errors))

    return {
        "validation_json": str(validation_path),
        "status": payload.get("status"),
        "method": payload.get("method"),
        "exports_dir": validation_exports,
        "image_list": validation_image_list,
        "checks": {name: checks.get(name) for name in REQUIRED_EXPORT_CHECKS},
        "tolerances": payload.get("tolerances", {}),
        "direct_export_max_abs_diff": payload.get("direct_export_max_abs_diff"),
        "mean_prob_target": payload.get("export_metrics", {}).get("mean_prob_target"),
        "mean_prob_bg": payload.get("export_metrics", {}).get("mean_prob_bg"),
    }


def select_pd_matched_threshold(curve_path: Path, target_pd: float) -> tuple[float, dict, bool]:
    rows = read_curve(curve_path)
    feasible = [row for row in rows if row.get("Pd", -1.0) >= target_pd]
    if feasible:
        best = min(feasible, key=lambda row: (row.get("FA_ppm", math.inf), -row.get("threshold", 0.0)))
        return float(best["threshold"]), best, True
    best = max(rows, key=lambda row: (row.get("Pd", -1.0), -row.get("FA_ppm", math.inf)))
    return float(best["threshold"]), best, False


def components(mask: np.ndarray):
    label = measure.label(mask.astype(np.uint8), connectivity=2)
    return measure.regionprops(label)


def target_centers(gt: np.ndarray) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in region.centroid) for region in components(gt)]


def dilate_numpy(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    from scipy.ndimage import binary_dilation

    yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    structure = (yy * yy + xx * xx) <= radius * radius
    return binary_dilation(mask.astype(bool), structure=structure)


def distance_bin(min_distance: float) -> str:
    if min_distance <= 2.0:
        return "T1"
    if min_distance <= 5.0:
        return "T2"
    if min_distance <= 10.0:
        return "T3"
    return "Far-FP"


def nearest_center(component_center: tuple[float, float], centers: list[tuple[float, float]]) -> tuple[float | None, float | None, float | None]:
    if not centers:
        return None, None, None
    cy, cx = component_center
    distances = [(math.hypot(cy - ty, cx - tx), ty, tx) for ty, tx in centers]
    dist, ty, tx = min(distances, key=lambda item: item[0])
    return ty, tx, dist


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def census(args) -> dict:
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    names = read_names(dataset_dir, args.dataset_name, args.image_list)
    exports_dir = Path(args.exports_dir)
    rows = []
    class_stats = {}
    unmatched_fp_components = 0
    detached_near_fp_components = 0
    far_fp_components = 0
    matched_target_components = 0
    boundary_excess_instances = 0
    total_fp_pixels = 0.0
    total_confidence = 0.0
    boundary_excess_pixels = 0.0
    boundary_excess_confidence = 0.0
    detached_near_fp_pixels = 0.0
    detached_near_fp_confidence = 0.0
    far_fp_pixels = 0.0
    far_fp_confidence = 0.0

    for name in names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        pred = prob > args.threshold
        gt_dilated = dilate_numpy(gt, args.dilate_radius)
        dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)
        centers = target_centers(gt)
        for idx, region in enumerate(components(pred), start=1):
            component = np.zeros_like(pred, dtype=bool)
            component[region.coords[:, 0], region.coords[:, 1]] = True
            fp_mask = component & (~gt)
            overlaps_gt = bool(np.logical_and(component, gt).any())
            if overlaps_gt:
                matched_target_components += 1
            if not fp_mask.any():
                continue
            if overlaps_gt:
                min_distance = 0.0
                dist_bin = "Boundary"
                category = BOUNDARY_EXCESS
                target_near_pixel = True
                unmatched_fp_component = False
                target_near_component = False
            else:
                min_distance = float(dist_to_gt[fp_mask].min())
                dist_bin = distance_bin(min_distance)
                target_near_component = min_distance <= 10.0
                target_near_pixel = target_near_component
                unmatched_fp_component = True
                category = DETACHED_NEAR_FP if target_near_component else FAR_FP
            cy, cx = (float(v) for v in region.centroid)
            ty, tx, center_distance = nearest_center((cy, cx), centers)
            fp_probs = prob[fp_mask]
            fp_pixels = int(fp_mask.sum())
            confidence_mass = float(fp_probs.sum())
            row = {
                "method": args.method,
                "split": args.split_name,
                "threshold_mode": args.threshold_mode,
                "threshold": args.threshold,
                "image_id": name,
                "component_id": idx,
                "category": category,
                "distance_bin": dist_bin,
                "target_near": int(target_near_pixel),
                "target_near_pixel": int(target_near_pixel),
                "target_near_component": int(target_near_component),
                "unmatched_fp_component": int(unmatched_fp_component),
                "matched_target_component": int(overlaps_gt),
                "manual_subtype": "",
                "component_area": int(region.area),
                "pixel_mass": fp_pixels,
                "fp_pixel_mass": fp_pixels,
                "mean_probability": float(fp_probs.mean()),
                "max_probability": float(fp_probs.max()),
                "confidence_mass": confidence_mass,
                "minimum_distance_to_gt": min_distance,
                "overlaps_gt": int(overlaps_gt),
                "overlap_with_gt_dilation": int(np.logical_and(fp_mask, gt_dilated).any()),
                "component_center_y": cy,
                "component_center_x": cx,
                "nearest_target_center_y": ty if ty is not None else "",
                "nearest_target_center_x": tx if tx is not None else "",
                "center_distance_to_nearest_target": center_distance if center_distance is not None else "",
                "bbox_y0": int(region.bbox[0]),
                "bbox_y1": int(region.bbox[2]),
                "bbox_x0": int(region.bbox[1]),
                "bbox_x1": int(region.bbox[3]),
            }
            rows.append(row)
            stat = class_stats.setdefault(category, {"events": 0, "components": 0, "fp_pixels": 0.0, "confidence_mass": 0.0})
            stat["events"] += 1
            stat["components"] += int(unmatched_fp_component)
            stat["fp_pixels"] += fp_pixels
            stat["confidence_mass"] += confidence_mass
            total_fp_pixels += fp_pixels
            total_confidence += confidence_mass
            if category == BOUNDARY_EXCESS:
                boundary_excess_instances += 1
                boundary_excess_pixels += fp_pixels
                boundary_excess_confidence += confidence_mass
            elif category == DETACHED_NEAR_FP:
                unmatched_fp_components += 1
                detached_near_fp_components += 1
                detached_near_fp_pixels += fp_pixels
                detached_near_fp_confidence += confidence_mass
            else:
                unmatched_fp_components += 1
                far_fp_components += 1
                far_fp_pixels += fp_pixels
                far_fp_confidence += confidence_mass

    output_dir = Path(args.output_dir)
    fields = [
        "method", "split", "threshold_mode", "threshold", "image_id", "component_id", "category",
        "distance_bin", "target_near", "target_near_pixel", "target_near_component",
        "unmatched_fp_component", "matched_target_component", "manual_subtype",
        "component_area", "pixel_mass", "fp_pixel_mass",
        "mean_probability", "max_probability", "confidence_mass",
        "minimum_distance_to_gt", "overlaps_gt", "overlap_with_gt_dilation", "component_center_y",
        "component_center_x", "nearest_target_center_y", "nearest_target_center_x",
        "center_distance_to_nearest_target", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1",
    ]
    write_csv(output_dir / "fp_components.csv", rows, fields)
    target_near_pixels = boundary_excess_pixels + detached_near_fp_pixels
    target_near_confidence = boundary_excess_confidence + detached_near_fp_confidence
    summary = {
        "method": args.method,
        "split": args.split_name,
        "threshold_mode": args.threshold_mode,
        "threshold": args.threshold,
        "pd_target": args.pd_target,
        "pd_target_reached": args.pd_target_reached,
        "images": len(names),
        "fp_component_definition": "unmatched predicted connected components only; matched target components are excluded from the component denominator",
        "pixel_near_definition": "boundary-excess pixels plus detached near-FP pixels",
        "component_near_definition": "detached near-FP components divided by all unmatched FP components",
        "matched_target_components": matched_target_components,
        "boundary_excess_instances": boundary_excess_instances,
        "boundary_excess_pixel_mass": boundary_excess_pixels,
        "boundary_excess_confidence_mass": boundary_excess_confidence,
        "unmatched_fp_components": unmatched_fp_components,
        "detached_near_fp_components": detached_near_fp_components,
        "far_fp_components": far_fp_components,
        "detached_near_fp_pixel_mass": detached_near_fp_pixels,
        "detached_near_fp_confidence_mass": detached_near_fp_confidence,
        "far_fp_pixel_mass": far_fp_pixels,
        "far_fp_confidence_mass": far_fp_confidence,
        "fp_components": unmatched_fp_components,
        "fp_pixel_mass": total_fp_pixels,
        "confidence_mass": total_confidence,
        "target_near_components": detached_near_fp_components,
        "target_near_pixel_mass": target_near_pixels,
        "target_near_confidence_mass": target_near_confidence,
        "R_near_component": detached_near_fp_components / max(1, unmatched_fp_components),
        "R_near_pixel": target_near_pixels / max(1.0, total_fp_pixels),
        "R_near_confidence": target_near_confidence / max(1e-12, total_confidence),
        "R_component_target_near": detached_near_fp_components / max(1, unmatched_fp_components),
        "R_pixel_target_near": target_near_pixels / max(1.0, total_fp_pixels),
        "R_confidence_target_near": target_near_confidence / max(1e-12, total_confidence),
        "class_stats": class_stats,
        "export_validation": args.export_validation,
        "component_csv": str(output_dir / "fp_components.csv"),
    }
    (output_dir / "fp_distance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Distance-stratified census of real FP components.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--threshold_mode", default="fixed", choices=["fixed", "pd_matched"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_curve", default=None)
    parser.add_argument("--pd_target", type=float, default=None)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--dilate_radius", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    args.export_validation = validate_export_gate(args)
    args.pd_target_reached = None
    if args.threshold_mode == "pd_matched":
        if args.threshold_curve is None or args.pd_target is None:
            raise ValueError("--threshold_curve and --pd_target are required for pd_matched mode")
        args.threshold, _, args.pd_target_reached = select_pd_matched_threshold(Path(args.threshold_curve), args.pd_target)
    census(args)


if __name__ == "__main__":
    main()
