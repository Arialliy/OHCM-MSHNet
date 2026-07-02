#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
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


def load_raw(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("I"), dtype=np.float32)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def prob_rgb(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    red = (prob * 255).astype(np.uint8)
    blue = ((1.0 - prob) * 255).astype(np.uint8)
    green = (np.minimum(prob, 1.0 - prob) * 2.0 * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0].numpy() > 0


def components(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    label = measure.label(mask.astype(np.uint8), connectivity=2)
    for region in measure.regionprops(label):
        comp = np.zeros_like(mask, dtype=bool)
        comp[region.coords[:, 0], region.coords[:, 1]] = True
        out.append(comp)
    return out


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    return inter / float(area_a + area_b - inter + 1e-6)


def local_contrast(raw: np.ndarray, box: tuple[int, int, int, int], ring: int = 6) -> float:
    y0, y1, x0, x1 = box
    patch = raw[y0:y1, x0:x1]
    ry0, ry1 = max(0, y0 - ring), min(raw.shape[0], y1 + ring)
    rx0, rx1 = max(0, x0 - ring), min(raw.shape[1], x1 + ring)
    surround = raw[ry0:ry1, rx0:rx1].copy()
    surround[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = np.nan
    outside = surround[~np.isnan(surround)]
    if patch.size == 0 or outside.size == 0:
        return 0.0
    return float((patch.mean() - outside.mean()) / (outside.std() + 1e-6))


def topq_mean(values: np.ndarray, q: float = 0.25) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(math.floor(q * flat.size)))
    return float(np.partition(flat, -k)[-k:].mean())


def target_area_range(dataset_dir: Path, names: list[str], min_scale: float, max_scale: float) -> tuple[int, int, dict]:
    areas = []
    for name in names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        for comp in components(gt):
            areas.append(int(comp.sum()))
    if not areas:
        return 1, 999999, {"target_areas": 0}
    arr = np.asarray(areas, dtype=np.float32)
    q05, q50, q95 = np.percentile(arr, [5, 50, 95])
    min_area = max(1, int(math.floor(q05 * min_scale)))
    max_area = max(min_area, int(math.ceil(q95 * max_scale)))
    return min_area, max_area, {
        "target_areas": len(areas),
        "target_area_q05": float(q05),
        "target_area_median": float(q50),
        "target_area_q95": float(q95),
        "candidate_min_area": int(min_area),
        "candidate_max_area": int(max_area),
    }


def crop_panel(raw: np.ndarray, prob: np.ndarray, gt: np.ndarray, cand: np.ndarray, box, margin: int, title: str) -> Image.Image:
    y0, y1, x0, x1 = box
    cy0, cy1 = max(0, y0 - margin), min(raw.shape[0], y1 + margin)
    cx0, cx1 = max(0, x0 - margin), min(raw.shape[1], x1 + margin)
    raw_u8 = to_uint8(raw[cy0:cy1, cx0:cx1])
    raw_rgb = np.stack([raw_u8, raw_u8, raw_u8], axis=-1)
    heat = prob_rgb(prob[cy0:cy1, cx0:cx1])
    overlay = raw_rgb.copy()
    cand_crop = cand[cy0:cy1, cx0:cx1]
    gt_crop = gt[cy0:cy1, cx0:cx1]
    overlay[gt_crop] = np.array([0, 220, 0], dtype=np.uint8)
    overlay[cand_crop] = np.array([255, 0, 0], dtype=np.uint8)
    panel = np.concatenate([raw_rgb, heat, overlay], axis=1)
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)
    rel_box = (x0 - cx0, y0 - cy0, x1 - cx0 - 1, y1 - cy0 - 1)
    for offset in (0, raw_rgb.shape[1], raw_rgb.shape[1] * 2):
        draw.rectangle((rel_box[0] + offset, rel_box[1], rel_box[2] + offset, rel_box[3]), outline=(255, 255, 0), width=1)
    draw.text((3, 3), title, fill=(255, 255, 255))
    return img


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(crop_paths: list[Path], output_dir: Path, per_sheet: int = 50, thumb_w: int = 240) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sheet_idx in range(0, len(crop_paths), per_sheet):
        paths = crop_paths[sheet_idx:sheet_idx + per_sheet]
        thumbs = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            scale = thumb_w / float(img.width)
            img = img.resize((thumb_w, max(1, int(img.height * scale))))
            thumbs.append(img)
        if not thumbs:
            continue
        cols = 5
        rows = int(math.ceil(len(thumbs) / cols))
        cell_h = max(img.height for img in thumbs)
        sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (20, 20, 20))
        for idx, img in enumerate(thumbs):
            x = (idx % cols) * thumb_w
            y = (idx // cols) * cell_h
            sheet.paste(img, (x, y))
        sheet.save(output_dir / f"contact_sheet_{sheet_idx // per_sheet + 1:03d}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an oracle review pool from true teacher FP components only.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--teacher_exports", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6")
    parser.add_argument("--dilate_radius", type=int, default=5)
    parser.add_argument("--min_threshold_persistence", type=int, default=2)
    parser.add_argument("--area_min_scale", type=float, default=0.5)
    parser.add_argument("--area_max_scale", type=float, default=2.0)
    parser.add_argument("--max_area", type=int, default=0)
    parser.add_argument("--min_area", type=int, default=0)
    parser.add_argument("--max_review", type=int, default=500)
    parser.add_argument("--crop_margin", type=int, default=24)
    parser.add_argument("--contact_sheet_count", type=int, default=500)
    args = parser.parse_args()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    thresholds = sorted(thresholds)
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    names = [line.strip() for line in (dataset_dir / "img_idx" / f"train_{args.dataset_name}.txt").read_text().splitlines() if line.strip()]
    auto_min, auto_max, area_stats = target_area_range(dataset_dir, names, args.area_min_scale, args.area_max_scale)
    min_area = args.min_area if args.min_area > 0 else auto_min
    max_area = args.max_area if args.max_area > 0 else auto_max

    output_dir = Path(args.output_dir)
    crop_dir = output_dir / "review_crops"
    mask_dir = output_dir / "candidate_masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    crop_paths = []
    raw_counts = {str(theta): 0 for theta in thresholds}
    candidate_id = 0
    for name in names:
        raw = load_raw(find_file(dataset_dir / "images", name))
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(Path(args.teacher_exports) / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        gt_dilated = dilate(gt, args.dilate_radius)
        comps_by_threshold = {}
        boxes_by_threshold = {}
        for theta in thresholds:
            safe_pred = (prob > theta) & (~gt_dilated)
            comps = components(safe_pred)
            comps_by_threshold[theta] = comps
            boxes_by_threshold[theta] = [bbox_from_mask(comp) for comp in comps if comp.any()]
            raw_counts[str(theta)] += len(comps)

        base_theta = thresholds[0]
        for comp in comps_by_threshold[base_theta]:
            if not comp.any():
                continue
            box = bbox_from_mask(comp)
            present = [base_theta]
            matched_masks = [comp]
            for theta in thresholds[1:]:
                best_idx = -1
                best_iou = 0.0
                for idx, other_box in enumerate(boxes_by_threshold[theta]):
                    iou = box_iou(box, other_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou > 0:
                    present.append(theta)
                    matched_masks.append(comps_by_threshold[theta][best_idx])
            if len(present) < args.min_threshold_persistence:
                continue
            cand_mask = matched_masks[-1]
            cand_area = int(cand_mask.sum())
            if cand_area < min_area or cand_area > max_area:
                continue
            y0, y1, x0, x1 = bbox_from_mask(cand_mask)
            if np.logical_and(cand_mask, gt_dilated).any():
                continue
            candidate_id += 1
            cid = f"oracle_{candidate_id:05d}"
            mask_path = mask_dir / f"{cid}.png"
            crop_path = crop_dir / f"{cid}_{name}.jpg"
            Image.fromarray((cand_mask.astype(np.uint8) * 255)).save(mask_path)
            title = f"{cid} {name} p={len(present)} area={cand_area}"
            crop = crop_panel(raw, prob, gt_dilated, cand_mask, (y0, y1, x0, x1), args.crop_margin, title)
            crop.save(crop_path, quality=92)
            crop_paths.append(crop_path)
            values = prob[cand_mask]
            rows.append({
                "candidate_id": cid,
                "image_id": name,
                "mask_path": str(mask_path),
                "crop_path": str(crop_path),
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "area": cand_area,
                "thresholds_present": ";".join(f"{theta:.2f}" for theta in present),
                "threshold_persistence": len(present),
                "teacher_score": topq_mean(values, 0.25),
                "mean_prob": float(values.mean()) if values.size else 0.0,
                "max_prob": float(values.max()) if values.size else 0.0,
                "local_contrast": local_contrast(raw, (y0, y1, x0, x1)),
                "gt_leakage": 0,
                "flat_candidate": int(float(values.mean()) < 0.05) if values.size else 1,
                "large_region": int(cand_area > max_area),
                "clutter_type": "",
                "verified": "false",
                "keep_for_oracle": "false",
            })

    rows.sort(
        key=lambda row: (
            int(row["threshold_persistence"]),
            float(row["max_prob"]),
            float(row["teacher_score"]),
            abs(float(row["local_contrast"])),
        ),
        reverse=True,
    )
    review_rows = rows[: args.max_review]
    fields = [
        "candidate_id", "image_id", "mask_path", "crop_path", "y0", "y1", "x0", "x1", "area",
        "thresholds_present", "threshold_persistence", "teacher_score", "mean_prob", "max_prob",
        "local_contrast", "gt_leakage", "flat_candidate", "large_region", "clutter_type",
        "verified", "keep_for_oracle",
    ]
    write_csv(output_dir / "oracle_candidate_pool.csv", rows, fields)
    write_csv(output_dir / "oracle_review_template.csv", review_rows, fields)
    make_contact_sheets([Path(row["crop_path"]) for row in review_rows[: args.contact_sheet_count]], output_dir / "contact_sheets")

    summary = {
        "dataset": args.dataset_name,
        "images": len(names),
        "thresholds": thresholds,
        "dilate_radius": args.dilate_radius,
        "min_threshold_persistence": args.min_threshold_persistence,
        "raw_component_counts_by_threshold": raw_counts,
        "area_stats": area_stats,
        "min_area": min_area,
        "max_area": max_area,
        "candidate_pool": len(rows),
        "review_candidates": len(review_rows),
        "flat_candidate_ratio": sum(int(row["flat_candidate"]) for row in rows) / max(1, len(rows)),
        "gt_leakage": 0,
        "candidate_csv": str(output_dir / "oracle_candidate_pool.csv"),
        "review_template_csv": str(output_dir / "oracle_review_template.csv"),
        "review_crop_dir": str(crop_dir),
        "contact_sheet_dir": str(output_dir / "contact_sheets"),
    }
    (output_dir / "oracle_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
