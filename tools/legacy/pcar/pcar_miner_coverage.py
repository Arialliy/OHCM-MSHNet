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


def fp_components(prob: np.ndarray, gt: np.ndarray, threshold: float):
    pred = prob > threshold
    label = measure.label(pred.astype(np.uint8), connectivity=2)
    out = []
    for region in measure.regionprops(label):
        mask = np.zeros_like(pred, dtype=bool)
        mask[region.coords[:, 0], region.coords[:, 1]] = True
        if np.logical_and(mask, gt).any():
            continue
        out.append({"mask": mask, "area": int(region.area), "centroid": region.centroid, "bbox": tuple(int(v) for v in region.bbox)})
    return out


def box_mask(box, shape):
    y0, y1, x0, x1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y0):min(shape[0], y1), max(0, x0):min(shape[1], x1)] = True
    return mask


def read_candidates(path: Path):
    by_image = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row.get("image") or row.get("image_id")
            if not image:
                continue
            y0 = int(float(row["y0"]))
            y1 = int(float(row["y1"]))
            x0 = int(float(row["x0"]))
            x1 = int(float(row["x1"]))
            by_image.setdefault(image, []).append({**row, "box": (y0, y1, x0, x1)})
    return by_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TSR/persistent miner coverage of OHEM FP components.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--ohem_exports", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--flat_prob_threshold", type=float, default=0.05)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    if args.image_list:
        names = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]
    else:
        names = [line.strip() for line in (dataset_dir / "img_idx" / f"train_{args.dataset_name}.txt").read_text().splitlines() if line.strip()]
    candidates = read_candidates(Path(args.candidate_csv))
    ohem_exports = Path(args.ohem_exports)
    rows = []
    total_fp = covered_fp = 0
    total_cand = fp_cand = gt_leak = flat_cand = low_conf_cand = 0
    active_images = 0
    for name in names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(ohem_exports / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        fps = fp_components(prob, gt, args.threshold)
        cands = candidates.get(name, [])
        if cands:
            active_images += 1
        image_covered = 0
        image_fp_cand = 0
        image_gt_leak = 0
        image_flat = 0
        for fp in fps:
            hit = any(np.logical_and(fp["mask"], box_mask(c["box"], gt.shape)).any() for c in cands)
            image_covered += int(hit)
        for cand in cands:
            cmask = box_mask(cand["box"], gt.shape)
            overlaps_fp = any(np.logical_and(fp["mask"], cmask).any() for fp in fps)
            leaks_gt = np.logical_and(cmask, gt).any()
            mean_prob = float(prob[cmask].mean()) if cmask.any() else 0.0
            image_fp_cand += int(overlaps_fp)
            image_gt_leak += int(leaks_gt)
            image_flat += int(mean_prob < args.flat_prob_threshold)
            low_conf_cand += int(mean_prob < args.flat_prob_threshold)
        total_fp += len(fps)
        covered_fp += image_covered
        total_cand += len(cands)
        fp_cand += image_fp_cand
        gt_leak += image_gt_leak
        flat_cand += image_flat
        rows.append({
            "image": name,
            "fp_components": len(fps),
            "covered_fp_components": image_covered,
            "candidate_count": len(cands),
            "fp_covering_candidates": image_fp_cand,
            "gt_leak_candidates": image_gt_leak,
            "flat_candidates": image_flat,
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "miner_coverage_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": len(names),
        "active_images": active_images,
        "active_image_ratio": active_images / max(1, len(names)),
        "fp_components": total_fp,
        "covered_fp_components": covered_fp,
        "fp_recall_at_k": covered_fp / max(1, total_fp),
        "candidates": total_cand,
        "fp_covering_candidates": fp_cand,
        "candidate_precision_at_k": fp_cand / max(1, total_cand),
        "gt_leak_candidates": gt_leak,
        "gt_leak_ratio": gt_leak / max(1, total_cand),
        "flat_candidates": flat_cand,
        "flat_candidate_ratio": flat_cand / max(1, total_cand),
        "low_conf_candidate_ratio": low_conf_cand / max(1, total_cand),
    }
    (output_dir / "miner_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
