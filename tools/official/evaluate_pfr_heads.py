#!/usr/bin/env python3
"""Evaluate PFR evidence/final/residual behavior.

This is an audit tool only. It must not trigger training.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit PFR evidence/final/residual heads.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="test", choices=["test", "full", "train"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--model_name", default="PFRMSHNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_dilate_radius", type=int, default=3)
    parser.add_argument("--far_dilate_radius", type=int, default=10)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--pfr_beta", type=float, default=0.5)
    parser.add_argument("--pfr_feature_channels", type=int, default=16)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))[0, 0].numpy() > 0


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def count_components(mask: np.ndarray) -> int:
    return int(measure.label(mask.astype(np.uint8), connectivity=2).max())


def region_to_mask(region, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[region.coords[:, 0], region.coords[:, 1]] = True
    return mask


def region_iou(region, target_mask: np.ndarray) -> float:
    region_mask = region_to_mask(region, target_mask.shape)
    inter = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(inter, union)


def match_components(pred_mask: np.ndarray, gt_mask: np.ndarray, distance_threshold: float = 3.0):
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


def init_stats():
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


def update_stats(stats: dict, prob: np.ndarray, gt_mask: np.ndarray, threshold: float) -> dict:
    pred = prob > threshold
    gt = gt_mask.astype(bool)
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
    return {
        "mIoU": safe_div(inter, union),
        "Pd": safe_div(matched_targets, target_components),
        "Precision": safe_div(tp, tp + fp),
        "FA_ppm": safe_div(fp, pred.size) * 1_000_000.0,
        "FP_components": int(fp_components),
    }


def finalize_stats(stats: dict) -> dict:
    precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
    fa = safe_div(stats["fp"], stats["pixels"])
    return {
        "mIoU": safe_div(stats["inter"], stats["union"]),
        "nIoU": safe_div(stats["niou_sum"], stats["count"]),
        "Pd": safe_div(stats["matched_targets"], stats["target_components"]),
        "detected_targets": stats["matched_targets"],
        "target_components": stats["target_components"],
        "FA": fa,
        "FA_ppm": fa * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": safe_div(2.0 * precision * recall, precision + recall),
        "FP_pixels": stats["fp"],
        "GT_pixels": stats["tp"] + stats["fn"],
        "FP_components": stats["fp_components"],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def torch_load_checkpoint(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(net: Net, checkpoint_path: str, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def resolve_image_ids(args, dataset):
    if args.image_list:
        path = Path(args.image_list)
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


def append_if_finite(values: list[float], value: float) -> None:
    if np.isfinite(value):
        values.append(float(value))


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, image_source = resolve_image_ids(args, dataset)
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)

    evidence_stats = init_stats()
    final_stats = init_stats()
    per_image_rows = []
    residual_totals = {
        "residual_new_fp_pixels": 0,
        "residual_new_fp_components": 0,
        "residual_removed_fp_pixels": 0,
        "residual_removed_fp_components": 0,
        "residual_lost_target_pixels": 0,
        "residual_boundary_excess_pixels": 0,
    }
    delta_target = []
    delta_boundary = []
    delta_far_bg = []
    delta_positive_far_bg_ratio = []

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0

            export = net.export_logits_features(img)
            evidence_logits = export["target_logit"][:, :, :h, :w]
            final_logits = export["logit"][:, :, :h, :w]
            delta = export["delta_logit"][:, :, :h, :w]
            evidence_prob = foreground_probability(evidence_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            final_prob = foreground_probability(final_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            delta_np = delta[0, 0].detach().cpu().numpy().astype(np.float32)

            evidence_metrics = update_stats(evidence_stats, evidence_prob, gt, args.threshold)
            final_metrics = update_stats(final_stats, final_prob, gt, args.threshold)

            evidence_mask = evidence_prob > args.threshold
            final_mask = final_prob > args.threshold
            bg = ~gt
            target_dilate = binary_dilate(gt, args.target_dilate_radius)
            boundary = target_dilate & (~gt)
            far_bg = ~binary_dilate(gt, args.far_dilate_radius)

            new_fp = final_mask & (~evidence_mask) & bg
            removed_fp = evidence_mask & (~final_mask) & bg
            lost_target = evidence_mask & (~final_mask) & gt
            boundary_excess = final_mask & (~evidence_mask) & boundary

            new_fp_pixels = int(new_fp.sum())
            removed_fp_pixels = int(removed_fp.sum())
            lost_target_pixels = int(lost_target.sum())
            boundary_excess_pixels = int(boundary_excess.sum())
            new_fp_components = count_components(new_fp)
            removed_fp_components = count_components(removed_fp)

            residual_totals["residual_new_fp_pixels"] += new_fp_pixels
            residual_totals["residual_new_fp_components"] += new_fp_components
            residual_totals["residual_removed_fp_pixels"] += removed_fp_pixels
            residual_totals["residual_removed_fp_components"] += removed_fp_components
            residual_totals["residual_lost_target_pixels"] += lost_target_pixels
            residual_totals["residual_boundary_excess_pixels"] += boundary_excess_pixels

            if gt.any():
                append_if_finite(delta_target, float(delta_np[gt].mean()))
            if boundary.any():
                append_if_finite(delta_boundary, float(delta_np[boundary].mean()))
            if far_bg.any():
                append_if_finite(delta_far_bg, float(delta_np[far_bg].mean()))
                append_if_finite(delta_positive_far_bg_ratio, float((delta_np[far_bg] > 0).mean()))

            per_image_rows.append(
                {
                    "image_name": name,
                    "evidence_mIoU": evidence_metrics["mIoU"],
                    "final_mIoU": final_metrics["mIoU"],
                    "evidence_Pd": evidence_metrics["Pd"],
                    "final_Pd": final_metrics["Pd"],
                    "evidence_Precision": evidence_metrics["Precision"],
                    "final_Precision": final_metrics["Precision"],
                    "evidence_FA_ppm": evidence_metrics["FA_ppm"],
                    "final_FA_ppm": final_metrics["FA_ppm"],
                    "evidence_FP_components": evidence_metrics["FP_components"],
                    "final_FP_components": final_metrics["FP_components"],
                    "residual_new_fp_pixels": new_fp_pixels,
                    "residual_new_fp_components": new_fp_components,
                    "residual_removed_fp_pixels": removed_fp_pixels,
                    "residual_removed_fp_components": removed_fp_components,
                    "residual_lost_target_pixels": lost_target_pixels,
                    "residual_boundary_excess_pixels": boundary_excess_pixels,
                    "delta_mean_target": float(delta_np[gt].mean()) if gt.any() else float("nan"),
                    "delta_mean_boundary": float(delta_np[boundary].mean()) if boundary.any() else float("nan"),
                    "delta_mean_far_bg": float(delta_np[far_bg].mean()) if far_bg.any() else float("nan"),
                    "delta_positive_far_bg_ratio": float((delta_np[far_bg] > 0).mean()) if far_bg.any() else float("nan"),
                }
            )

            if (idx + 1) % 100 == 0:
                print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    evidence = finalize_stats(evidence_stats)
    final = finalize_stats(final_stats)
    summary = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "split": args.split,
        "image_source": image_source,
        "num_images": len(per_image_rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "evidence": evidence,
        "final": final,
        "evidence_mIoU": evidence["mIoU"],
        "final_mIoU": final["mIoU"],
        "evidence_Pd": evidence["Pd"],
        "final_Pd": final["Pd"],
        "evidence_Precision": evidence["Precision"],
        "final_Precision": final["Precision"],
        "evidence_FA_ppm": evidence["FA_ppm"],
        "final_FA_ppm": final["FA_ppm"],
        "evidence_FP_components": evidence["FP_components"],
        "final_FP_components": final["FP_components"],
        **residual_totals,
        "delta_mean_target": mean_or_nan(delta_target),
        "delta_mean_boundary": mean_or_nan(delta_boundary),
        "delta_mean_far_bg": mean_or_nan(delta_far_bg),
        "delta_positive_far_bg_ratio": mean_or_nan(delta_positive_far_bg_ratio),
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "worst_images": str(output_dir / "worst_images.csv"),
        },
    }

    fields = [
        "image_name",
        "evidence_mIoU",
        "final_mIoU",
        "evidence_Pd",
        "final_Pd",
        "evidence_Precision",
        "final_Precision",
        "evidence_FA_ppm",
        "final_FA_ppm",
        "evidence_FP_components",
        "final_FP_components",
        "residual_new_fp_pixels",
        "residual_new_fp_components",
        "residual_removed_fp_pixels",
        "residual_removed_fp_components",
        "residual_lost_target_pixels",
        "residual_boundary_excess_pixels",
        "delta_mean_target",
        "delta_mean_boundary",
        "delta_mean_far_bg",
        "delta_positive_far_bg_ratio",
    ]
    write_csv(output_dir / "per_image.csv", per_image_rows, fields)
    worst_rows = sorted(
        per_image_rows,
        key=lambda row: (-int(row["residual_new_fp_components"]), -int(row["residual_lost_target_pixels"])),
    )[:50]
    write_csv(output_dir / "worst_images.csv", worst_rows, fields)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
