#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def load_raw(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("I"), dtype=np.float32)


def to_u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1, 99])
    return (np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def region_mask(region, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[region.coords[:, 0], region.coords[:, 1]] = True
    return mask


def fp_components(prob: np.ndarray, gt: np.ndarray, threshold: float):
    pred = prob > threshold
    label = measure.label(pred.astype(np.uint8), connectivity=2)
    out = []
    for idx, region in enumerate(measure.regionprops(label), start=1):
        mask = region_mask(region, gt.shape)
        if np.logical_and(mask, gt).any():
            continue
        coords = region.coords
        out.append(
            {
                "id": idx,
                "region": region,
                "mask": mask,
                "area": int(region.area),
                "mean_conf": float(prob[coords[:, 0], coords[:, 1]].mean()),
                "max_conf": float(prob[coords[:, 0], coords[:, 1]].max()),
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "bbox": tuple(int(v) for v in region.bbox),
            }
        )
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum() + 1e-6)


def is_match(a, b, center_dist: float) -> bool:
    if iou(a["mask"], b["mask"]) > 0:
        return True
    dy = a["centroid_y"] - b["centroid_y"]
    dx = a["centroid_x"] - b["centroid_x"]
    return math.sqrt(dy * dy + dx * dx) <= center_dist


def summarize(rows, category: str) -> dict:
    items = [row for row in rows if row["category"] == category]
    return {
        "category": category,
        "count": len(items),
        "avg_area": float(np.mean([row["area"] for row in items])) if items else 0.0,
        "avg_confidence": float(np.mean([row["mean_conf"] for row in items])) if items else 0.0,
        "max_confidence": float(np.max([row["max_conf"] for row in items])) if items else 0.0,
        "total_fa_pixels": int(sum(row["area"] for row in items)),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def visualize_new_fp(rows, dataset_dir: Path, output_dir: Path, limit: int) -> None:
    vis_dir = output_dir / "new_fp_top100"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(sorted(rows, key=lambda r: r["max_conf"], reverse=True)[:limit]):
        name = row["image"]
        raw = load_raw(find_file(dataset_dir / "images", name))
        y0, x0, y1, x1 = row["bbox_y0"], row["bbox_x0"], row["bbox_y1"], row["bbox_x1"]
        pad = 24
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        yy0 = max(0, cy - pad)
        yy1 = min(raw.shape[0], cy + pad)
        xx0 = max(0, cx - pad)
        xx1 = min(raw.shape[1], cx + pad)
        crop = Image.fromarray(to_u8(raw[yy0:yy1, xx0:xx1])).convert("RGB")
        draw = ImageDraw.Draw(crop)
        draw.rectangle([x0 - xx0, y0 - yy0, x1 - xx0 - 1, y1 - yy0 - 1], outline=(255, 0, 0), width=2)
        draw.text((2, 2), f"{name} conf={row['max_conf']:.3f}", fill=(255, 255, 0))
        crop.save(vis_dir / f"{idx:03d}_{name}_{row['component_id']}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="FP transfer analysis between OHEM and TSR/PCAR outputs.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--ohem_exports", required=True)
    parser.add_argument("--method_exports", required=True)
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--center_dist", type=float, default=3.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    if args.image_list:
        names = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]
    else:
        names = [line.strip() for line in (dataset_dir / "img_idx" / f"test_{args.dataset_name}.txt").read_text().splitlines() if line.strip()]

    ohem_exports = Path(args.ohem_exports)
    method_exports = Path(args.method_exports)
    output_dir = Path(args.output_dir)
    rows = []
    for name in names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        ohem_prob = np.load(ohem_exports / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        method_prob = np.load(method_exports / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        ohem_fp = fp_components(ohem_prob, gt, args.threshold)
        method_fp = fp_components(method_prob, gt, args.threshold)
        ohem_shared = set()
        method_shared = set()
        for i, old in enumerate(ohem_fp):
            for j, new in enumerate(method_fp):
                if is_match(old, new, args.center_dist):
                    ohem_shared.add(i)
                    method_shared.add(j)
        for i, comp in enumerate(ohem_fp):
            category = "shared" if i in ohem_shared else "removed"
            y0, x0, y1, x1 = comp["bbox"]
            rows.append({
                "image": name,
                "category": category,
                "source": "OHEM",
                "component_id": comp["id"],
                "area": comp["area"],
                "mean_conf": comp["mean_conf"],
                "max_conf": comp["max_conf"],
                "centroid_y": comp["centroid_y"],
                "centroid_x": comp["centroid_x"],
                "bbox_y0": y0,
                "bbox_x0": x0,
                "bbox_y1": y1,
                "bbox_x1": x1,
            })
        for j, comp in enumerate(method_fp):
            if j in method_shared:
                continue
            y0, x0, y1, x1 = comp["bbox"]
            rows.append({
                "image": name,
                "category": "new",
                "source": args.method_name,
                "component_id": comp["id"],
                "area": comp["area"],
                "mean_conf": comp["mean_conf"],
                "max_conf": comp["max_conf"],
                "centroid_y": comp["centroid_y"],
                "centroid_x": comp["centroid_x"],
                "bbox_y0": y0,
                "bbox_x0": x0,
                "bbox_y1": y1,
                "bbox_x1": x1,
            })

    fields = ["image", "category", "source", "component_id", "area", "mean_conf", "max_conf", "centroid_y", "centroid_x", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"]
    write_csv(output_dir / "fp_transfer_components.csv", rows, fields)
    summary_rows = [summarize(rows, category) for category in ["removed", "shared", "new"]]
    write_csv(output_dir / "fp_transfer_summary.csv", summary_rows, ["category", "count", "avg_area", "avg_confidence", "max_confidence", "total_fa_pixels"])
    summary = {row["category"]: row for row in summary_rows}
    (output_dir / "fp_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    visualize_new_fp([row for row in rows if row["category"] == "new"], dataset_dir, output_dir, limit=100)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
