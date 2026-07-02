#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def read_names(dataset_dir: Path, dataset_name: str, split: str, image_list: str | None) -> list[str]:
    if image_list:
        return [line.strip() for line in Path(image_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    list_path = dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt"
    return [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        "removed_components": 0.0,
        "kept_components": 0.0,
    }


def update_stats(stats: dict, pred: np.ndarray, gt: np.ndarray, removed_components: int, kept_components: int) -> None:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)
    stats["inter"] += float(inter)
    stats["union"] += float(union)
    stats["tp"] += float(inter)
    stats["fp"] += float(fp)
    stats["fn"] += float(fn)
    stats["pixels"] += float(pred.size)
    stats["niou_sum"] += safe_div(inter, union)
    stats["count"] += 1
    stats["matched_targets"] += float(matched_targets)
    stats["target_components"] += float(target_components)
    stats["fp_components"] += float(fp_components)
    stats["removed_components"] += float(removed_components)
    stats["kept_components"] += float(kept_components)


def stats_row(stats: dict, threshold: float, gate_threshold: float) -> dict:
    precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fa = safe_div(stats["fp"], stats["pixels"])
    return {
        "threshold": threshold,
        "gate_threshold": gate_threshold,
        "mIoU": safe_div(stats["inter"], stats["union"]),
        "nIoU": safe_div(stats["niou_sum"], stats["count"]),
        "Pd": safe_div(stats["matched_targets"], stats["target_components"]),
        "FA": fa,
        "FA_ppm": fa * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP_components": stats["fp_components"],
        "removed_components": stats["removed_components"],
        "kept_components": stats["kept_components"],
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


def tta_consensus(net: Net, img: torch.Tensor, views: list[str], h: int, w: int) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for view in views:
            viewed = apply_view(img, view)
            logit = net.export_logits_features(viewed)["logit"]
            prob = torch.sigmoid(logit)
            prob = apply_view(prob, view)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            probs.append(prob)
    return np.stack(probs, axis=0).min(axis=0).astype(np.float32)


def component_score(values: np.ndarray, metric: str) -> float:
    if values.size == 0:
        return 0.0
    if metric == "max":
        return float(values.max())
    if metric == "mean":
        return float(values.mean())
    if metric == "p95":
        return float(np.percentile(values, 95))
    raise ValueError(metric)


def evaluate_case(original_prob: np.ndarray, consensus_prob: np.ndarray, threshold: float, gate_threshold: float, metric: str) -> tuple[np.ndarray, int, int]:
    pred = original_prob > threshold
    filtered = np.zeros_like(pred, dtype=bool)
    removed = 0
    kept = 0
    for region in connected_regions(pred):
        score = component_score(consensus_prob[region.coords[:, 0], region.coords[:, 1]], metric)
        if score >= gate_threshold:
            filtered[region.coords[:, 0], region.coords[:, 1]] = True
            kept += 1
        else:
            removed += 1
    return filtered, removed, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter original OHEM components using flip-consensus component scores.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--views", default="orig,hflip,vflip,hvflip")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gate_metric", default="max", choices=["max", "mean", "p95"])
    parser.add_argument("--gate_thresholds", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
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
    names = read_names(dataset_dir, args.dataset_name, args.split, args.image_list)
    views = [item.strip() for item in args.views.split(",") if item.strip()]
    gate_thresholds = [float(item) for item in args.gate_thresholds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    net = Net(args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    cases = []
    exports_dir = Path(args.exports_dir)
    for idx, name in enumerate(names):
        img, (h, w) = load_image(find_file(dataset_dir / "images", name), img_norm_cfg)
        img = img.to(device)
        gt = load_mask(find_file(dataset_dir / "masks", name))[:h, :w]
        original_prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[:h, :w]
        consensus_prob = tta_consensus(net, img, views, h, w)
        cases.append({"name": name, "gt": gt, "original_prob": original_prob, "consensus_prob": consensus_prob})
        if (idx + 1) % 100 == 0:
            print(f"Component gate precompute [{idx + 1}/{len(names)}]", flush=True)

    rows = []
    for gate_threshold in gate_thresholds:
        stats = empty_stats()
        for case in cases:
            filtered, removed, kept = evaluate_case(
                case["original_prob"], case["consensus_prob"], args.threshold, gate_threshold, args.gate_metric
            )
            update_stats(stats, filtered, case["gt"], removed, kept)
        rows.append(stats_row(stats, args.threshold, gate_threshold))

    fields = [
        "threshold", "gate_threshold", "mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision",
        "Recall", "F1", "FP_components", "removed_components", "kept_components",
    ]
    write_csv(output_dir / "component_gate_sweep.csv", rows, fields)
    summary = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "split": args.split,
        "image_list": os.path.abspath(args.image_list) if args.image_list else None,
        "exports_dir": str(exports_dir),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "views": views,
        "threshold": args.threshold,
        "gate_metric": args.gate_metric,
        "num_images": len(names),
        "outputs": {"component_gate_sweep": str(output_dir / "component_gate_sweep.csv")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir / "component_gate_sweep.csv"), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
