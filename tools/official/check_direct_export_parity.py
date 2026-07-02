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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def size_to_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")


def load_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def safe_div(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def region_iou(region, target_mask: np.ndarray) -> float:
    region_mask = np.zeros(target_mask.shape, dtype=bool)
    region_mask[region.coords[:, 0], region.coords[:, 1]] = True
    intersection = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(intersection, union)


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


def init_stats() -> dict:
    return {
        "inter": 0.0,
        "union": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "pixels": 0.0,
        "matched_targets": 0.0,
        "target_components": 0.0,
        "fp_components": 0.0,
    }


def update_stats(stats: dict, prob: np.ndarray, gt: np.ndarray, threshold: float) -> None:
    pred = prob > threshold
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
    stats["matched_targets"] += float(matched_targets)
    stats["target_components"] += float(target_components)
    stats["fp_components"] += float(fp_components)


def finalize_stats(stats: dict) -> dict:
    precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
    return {
        "mIoU": safe_div(stats["inter"], stats["union"]),
        "Pd": safe_div(stats["matched_targets"], stats["target_components"]),
        "detected_targets": stats["matched_targets"],
        "target_components": stats["target_components"],
        "FA_ppm": safe_div(stats["fp"], stats["pixels"]) * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": safe_div(2.0 * precision * recall, precision + recall),
        "FP_components": stats["fp_components"],
    }


def torch_load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_checkpoint(net: Net, checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint if isinstance(checkpoint, dict) else {}


def direct_probability(net: Net, img: torch.Tensor, h: int, w: int) -> np.ndarray:
    with torch.no_grad():
        logit = net.export_logits_features(img)["logit"]
        prob = foreground_probability(logit)[0, 0, :h, :w]
    return prob.detach().cpu().numpy().astype(np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate direct checkpoint evaluation against exported probabilities.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_prob_diff", type=float, default=1e-6)
    parser.add_argument("--max_miou_diff", type=float, default=1e-6)
    parser.add_argument("--max_pd_diff", type=float, default=1e-6)
    parser.add_argument("--max_fa_ppm_diff", type=float, default=1.0)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ecdv_beta_max", type=float, default=0.1)
    parser.add_argument("--ecdv_eval_beta", type=float, default=None)
    parser.add_argument("--ecdv_hidden_channels", type=int, default=32)
    parser.add_argument("--ecdv_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--ecdv_detach_verifier_input", action="store_true", default=True)
    parser.add_argument("--ecdv_no_detach_verifier_input", dest="ecdv_detach_verifier_input", action="store_false")
    parser.add_argument("--ecdv_contrast_kernel", type=int, default=9)
    parser.add_argument("--ecdv_highpass_kernel", type=int, default=9)
    parser.add_argument("--mscv_beta_max", type=float, default=0.1)
    parser.add_argument("--mscv_eval_beta", type=float, default=None)
    parser.add_argument("--mscv_hidden_channels", type=int, default=32)
    parser.add_argument("--mscv_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--mscv_detach_verifier_input", action="store_true", default=True)
    parser.add_argument("--mscv_no_detach_verifier_input", dest="mscv_detach_verifier_input", action="store_false")
    parser.add_argument("--mscv_contrast_kernel", type=int, default=9)
    parser.add_argument("--mscv_far_radius", type=int, default=7)
    parser.add_argument("--mscv_candidate_prob_thr", type=float, default=0.2)
    parser.add_argument("--mscv_candidate_std_thr", type=float, default=0.05)
    parser.add_argument("--mscv_nonflat_thr", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = Path(args.exports_dir)
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    train_dataset_name = args.train_dataset_name or args.dataset_name

    image_filter = None
    if args.image_list:
        image_filter = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    if image_filter is not None:
        test_set.test_list = image_filter
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, Path(args.checkpoint), device)

    direct_stats = init_stats()
    export_stats = init_stats()
    rows = []
    max_prob_diff = 0.0
    total_mask_diff = 0
    direct_target_sum = 0.0
    direct_bg_sum = 0.0
    export_target_sum = 0.0
    export_bg_sum = 0.0
    target_pixels = 0
    bg_pixels = 0

    for idx, (img, gt_mask, size, image_name) in enumerate(test_loader):
        img = img.to(device)
        h, w = size_to_int(size[0]), size_to_int(size[1])
        name = image_name[0] if isinstance(image_name, (tuple, list)) else str(image_name)
        direct_prob = direct_probability(net, img, h, w)
        export_prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[:h, :w]
        gt = load_mask(find_file(dataset_dir / "masks", name))[:h, :w]
        direct_mask = direct_prob > args.threshold
        export_mask = export_prob > args.threshold
        prob_diff = float(np.max(np.abs(direct_prob - export_prob)))
        mask_diff = int(np.not_equal(direct_mask, export_mask).sum())
        max_prob_diff = max(max_prob_diff, prob_diff)
        total_mask_diff += mask_diff

        direct_target_sum += float(direct_prob[gt].sum()) if gt.any() else 0.0
        export_target_sum += float(export_prob[gt].sum()) if gt.any() else 0.0
        target_pixels += int(gt.sum())
        bg = ~gt
        direct_bg_sum += float(direct_prob[bg].sum()) if bg.any() else 0.0
        export_bg_sum += float(export_prob[bg].sum()) if bg.any() else 0.0
        bg_pixels += int(bg.sum())

        update_stats(direct_stats, direct_prob, gt, args.threshold)
        update_stats(export_stats, export_prob, gt, args.threshold)
        rows.append({
            "image_name": name,
            "max_prob_diff": prob_diff,
            "mask_diff_pixels": mask_diff,
            "direct_target_mean": float(direct_prob[gt].mean()) if gt.any() else 0.0,
            "direct_bg_mean": float(direct_prob[bg].mean()) if bg.any() else 0.0,
            "export_target_mean": float(export_prob[gt].mean()) if gt.any() else 0.0,
            "export_bg_mean": float(export_prob[bg].mean()) if bg.any() else 0.0,
        })
        if (idx + 1) % 100 == 0:
            print(f"Checked [{idx + 1}/{len(test_loader)}]", flush=True)

    direct_metrics = finalize_stats(direct_stats)
    export_metrics = finalize_stats(export_stats)
    metric_diff = {
        "mIoU": abs(direct_metrics["mIoU"] - export_metrics["mIoU"]),
        "Pd": abs(direct_metrics["Pd"] - export_metrics["Pd"]),
        "FA_ppm": abs(direct_metrics["FA_ppm"] - export_metrics["FA_ppm"]),
    }
    direct_target_mean = safe_div(direct_target_sum, target_pixels)
    direct_bg_mean = safe_div(direct_bg_sum, bg_pixels)
    export_target_mean = safe_div(export_target_sum, target_pixels)
    export_bg_mean = safe_div(export_bg_sum, bg_pixels)
    checks = {
        "max_prob_diff": max_prob_diff <= args.max_prob_diff,
        "mask_diff_pixels": total_mask_diff == 0,
        "mIoU_diff": metric_diff["mIoU"] <= args.max_miou_diff,
        "Pd_diff": metric_diff["Pd"] <= args.max_pd_diff,
        "FA_ppm_diff": metric_diff["FA_ppm"] <= args.max_fa_ppm_diff,
        "direct_target_gt_background": direct_target_mean > direct_bg_mean,
        "export_target_gt_background": export_target_mean > export_bg_mean,
    }
    summary = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "model": args.model_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "exports_dir": os.path.abspath(args.exports_dir),
        "image_list": os.path.abspath(args.image_list) if args.image_list else None,
        "threshold": args.threshold,
        "num_images": len(rows),
        "max_prob_diff": max_prob_diff,
        "mask_diff_pixels": total_mask_diff,
        "direct_metrics": direct_metrics,
        "export_metrics": export_metrics,
        "metric_abs_diff": metric_diff,
        "direct_target_mean": direct_target_mean,
        "direct_background_mean": direct_bg_mean,
        "export_target_mean": export_target_mean,
        "export_background_mean": export_bg_mean,
        "checks": checks,
        "pass": all(checks.values()),
    }
    write_csv(output_dir / "direct_export_parity_per_image.csv", rows)
    (output_dir / "direct_export_parity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
