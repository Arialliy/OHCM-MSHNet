#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import measure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from net import Net  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg, seed_pytorch  # noqa: E402

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


def load_image(path: Path, img_norm_cfg) -> tuple[torch.Tensor, tuple[int, int]]:
    raw = np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    h, w = raw.shape
    img = Normalized(raw, img_norm_cfg)
    img = PadImg(img)
    tensor = torch.from_numpy(np.ascontiguousarray(img[None, None])).float()
    return tensor, (h, w)


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
        "target_prob_sum": 0.0,
        "target_pixel_count": 0,
        "bg_prob_sum": 0.0,
        "bg_pixel_count": 0,
    }


def update_stats(stats: dict, prob: np.ndarray, gt: np.ndarray, threshold: float) -> None:
    pred = prob > threshold
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
    if gt.any():
        stats["target_prob_sum"] += float(prob[gt].sum())
        stats["target_pixel_count"] += int(gt.sum())
    bg = ~gt
    if bg.any():
        stats["bg_prob_sum"] += float(prob[bg].sum())
        stats["bg_pixel_count"] += int(bg.sum())


def stats_row(stats: dict, threshold: float) -> dict:
    precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fa = safe_div(stats["fp"], stats["pixels"])
    return {
        "threshold": threshold,
        "mIoU": safe_div(stats["inter"], stats["union"]),
        "nIoU": safe_div(stats["niou_sum"], stats["count"]),
        "Pd": safe_div(stats["matched_targets"], stats["target_components"]),
        "FA": fa,
        "FA_ppm": fa * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP_components": stats["fp_components"],
        "mean_prob_target": safe_div(stats["target_prob_sum"], stats["target_pixel_count"]),
        "mean_prob_bg": safe_div(stats["bg_prob_sum"], stats["bg_pixel_count"]),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_view(img: torch.Tensor, view: str) -> torch.Tensor:
    if view == "orig":
        return img
    if view == "hflip":
        return torch.flip(img, dims=[3])
    if view == "vflip":
        return torch.flip(img, dims=[2])
    if view == "hvflip":
        return torch.flip(img, dims=[2, 3])
    raise ValueError(view)


def unapply_view(prob: torch.Tensor, view: str) -> torch.Tensor:
    return apply_view(prob, view)


def aggregate(probs: list[np.ndarray], mode: str, k: float) -> np.ndarray:
    stack = np.stack(probs, axis=0)
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "min":
        return stack.min(axis=0)
    if mode == "geom":
        return np.exp(np.log(np.clip(stack, 1e-7, 1.0)).mean(axis=0))
    if mode == "mean_minus_std":
        return np.clip(stack.mean(axis=0) - k * stack.std(axis=0), 0.0, 1.0)
    raise ValueError(mode)


def infer_prob(net: Net, img: torch.Tensor, views: list[str], mode: str, k: float, h: int, w: int) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for view in views:
            viewed = apply_view(img, view)
            logit = net.export_logits_features(viewed)["logit"]
            prob = torch.sigmoid(logit)
            prob = unapply_view(prob, view)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            probs.append(prob)
    return aggregate(probs, mode, k).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate flip-consensus TTA probability aggregation.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--views", default="orig,hflip,vflip,hvflip")
    parser.add_argument("--aggregation", default="mean_minus_std", choices=["mean", "min", "geom", "mean_minus_std"])
    parser.add_argument("--std_k", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    names = read_names(dataset_dir, args.dataset_name, args.image_list)
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    net = Net(args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    threshold_stats = {threshold: empty_stats() for threshold in thresholds}
    per_image_rows = []
    for idx, name in enumerate(names):
        img, (h, w) = load_image(find_file(dataset_dir / "images", name), img_norm_cfg)
        img = img.to(device)
        gt = load_mask(find_file(dataset_dir / "masks", name))[:h, :w]
        prob = infer_prob(net, img, views, args.aggregation, args.std_k, h, w)
        for threshold in thresholds:
            update_stats(threshold_stats[threshold], prob, gt, threshold)
        pred = prob > args.threshold
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        matched_targets, target_components, fp_components = match_components(pred, gt)
        per_image_rows.append(
            {
                "image_id": name,
                "threshold": args.threshold,
                "IoU": safe_div(inter, union),
                "Pd": safe_div(matched_targets, target_components),
                "FA_ppm": safe_div(np.logical_and(pred, ~gt).sum(), pred.size) * 1_000_000.0,
                "Precision": safe_div(inter, inter + np.logical_and(pred, ~gt).sum()),
                "FP_components": fp_components,
                "mean_prob_target": float(prob[gt].mean()) if gt.any() else 0.0,
                "mean_prob_bg": float(prob[~gt].mean()) if (~gt).any() else 0.0,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"TTA eval [{idx + 1}/{len(names)}]", flush=True)

    rows = [stats_row(threshold_stats[threshold], threshold) for threshold in sorted(threshold_stats)]
    fields = [
        "threshold", "mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1",
        "FP_components", "mean_prob_target", "mean_prob_bg",
    ]
    write_csv(output_dir / "threshold_curve.csv", rows, fields)
    write_csv(
        output_dir / "metrics_per_image.csv",
        per_image_rows,
        [
            "image_id", "threshold", "IoU", "Pd", "FA_ppm", "Precision", "FP_components",
            "mean_prob_target", "mean_prob_bg",
        ],
    )
    summary = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "model": args.model_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "image_list": os.path.abspath(args.image_list) if args.image_list else None,
        "num_images": len(names),
        "views": views,
        "aggregation": args.aggregation,
        "std_k": args.std_k,
        "threshold": args.threshold,
        "metrics_at_threshold": next(row for row in rows if abs(row["threshold"] - args.threshold) < 1e-9),
        "outputs": {
            "threshold_curve": str(output_dir / "threshold_curve.csv"),
            "per_image_metrics": str(output_dir / "metrics_per_image.csv"),
        },
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics_at_threshold"], indent=2), flush=True)


if __name__ == "__main__":
    main()
