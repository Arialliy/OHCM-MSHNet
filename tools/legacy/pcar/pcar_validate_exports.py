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
from torch.autograd import Variable
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from utils import get_img_norm_cfg

MSHNET_NAMES = ("MSHNet", "MSHNetFocal", "MSHNetOHEM", "MSHNetTopKNeg", "MSHNetSPSOHEM")

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
METRIC_TOL = 1e-4
FA_PPM_TOL = 1.0
MAX_DIFF_TOL = 1e-6


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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


def metric_row(stats: dict, threshold: float) -> dict:
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


def read_curve_row(path: Path, threshold: float) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if abs(float(row["threshold"]) - threshold) < 1e-9:
                return {key: float(value) for key, value in row.items() if value != ""}
    raise ValueError(f"threshold {threshold} not found in {path}")


def metric_deltas(left: dict, right: dict) -> dict:
    return {
        "mIoU": abs(float(left["mIoU"]) - float(right["mIoU"])),
        "Pd": abs(float(left["Pd"]) - float(right["Pd"])),
        "FA_ppm": abs(float(left["FA_ppm"]) - float(right["FA_ppm"])),
    }


def metrics_pass(deltas: dict) -> bool:
    return deltas["mIoU"] < METRIC_TOL and deltas["Pd"] < METRIC_TOL and deltas["FA_ppm"] < FA_PPM_TOL


def load_checkpoint(net: Net, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    return checkpoint


def resolve_mshnet_head(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "output0" if model_name == "MSHNet" else "final"


def direct_probability(net: Net, img: torch.Tensor, args, h: int, w: int) -> np.ndarray:
    head = resolve_mshnet_head(args.model_name, args.mshnet_export_head)
    if args.model_name in MSHNET_NAMES and head == "output0":
        _, logit, _ = net.model(img, False, return_feature=True)
    else:
        logit = net.export_logits_features(img)["logit"]
    return torch.sigmoid(logit)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)


def evaluate_exported_probs(names: list[str], dataset_dir: Path, exports_dir: Path, threshold: float) -> dict:
    stats = empty_stats()
    for name in names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        update_stats(stats, prob, gt, threshold)
    return metric_row(stats, threshold)


def evaluate_direct_and_diff(args, names: list[str], dataset_dir: Path, exports_dir: Path, threshold: float) -> tuple[dict, float, str]:
    train_dataset_name = args.train_dataset_name or args.dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    test_set.test_list = names
    loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)
    net.eval()

    stats = empty_stats()
    max_abs_diff = 0.0
    max_diff_image = ""
    with torch.no_grad():
        for img, gt_mask, size, image_name in loader:
            img = Variable(img).to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            prob = direct_probability(net, img, args, h, w)
            gt = gt_mask[0, 0, :h, :w].detach().cpu().numpy() > 0
            exported_prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)[:h, :w]
            diff = float(np.max(np.abs(prob - exported_prob)))
            if diff > max_abs_diff:
                max_abs_diff = diff
                max_diff_image = name
            update_stats(stats, prob, gt, threshold)
    row = metric_row(stats, threshold)
    row["checkpoint_epoch"] = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return row, max_abs_diff, max_diff_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported probability maps before FP census.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--threshold_curve", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--method", default="model")
    parser.add_argument("--mshnet_export_head", default="auto", choices=["auto", "output0", "final"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohcm_warm_epoch", type=int, default=60)
    parser.add_argument("--ohcm_gamma_max", type=float, default=0.3)
    parser.add_argument("--ohcm_gamma_ramp_epochs", type=int, default=60)
    parser.add_argument("--ohcm_inhibition_start_epoch", type=int, default=None)
    parser.add_argument("--ohcm_tau", type=float, default=0.5)
    parser.add_argument("--ohcm_dilate_radius", type=int, default=5)
    parser.add_argument("--ohcm_topk", type=int, default=3)
    parser.add_argument("--ohcm_margin_m", type=float, default=0.1)
    parser.add_argument("--ohcm_margin_delta", type=float, default=0.5)
    parser.add_argument("--ohcm_gt_area_median", type=float, default=20.0)
    parser.add_argument("--ohcm_mining_mode", default="cc_area_lc_ms")
    parser.add_argument("--ohcm_force_no_proto", action="store_true")
    parser.add_argument("--lambda_clu", type=float, default=0.2)
    parser.add_argument("--lambda_sup", type=float, default=0.5)
    parser.add_argument("--lambda_margin", type=float, default=0.1)
    parser.add_argument("--lambda_proto", type=float, default=0.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    exports_dir = Path(args.exports_dir)
    names = read_names(dataset_dir, args.dataset_name, args.image_list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    summary_metrics = summary["metrics_at_threshold"]
    curve_metrics = read_curve_row(Path(args.threshold_curve), args.threshold)
    export_metrics = evaluate_exported_probs(names, dataset_dir, exports_dir, args.threshold)

    export_vs_summary = metric_deltas(export_metrics, summary_metrics)
    curve_vs_summary = metric_deltas(curve_metrics, summary_metrics)
    direction_pass = export_metrics["mean_prob_target"] > export_metrics["mean_prob_bg"]
    checks = {
        "export_vs_summary": metrics_pass(export_vs_summary),
        "curve_0p5_vs_summary": metrics_pass(curve_vs_summary),
        "gt_mean_gt_bg": direction_pass,
    }

    direct_metrics = None
    direct_vs_export = None
    max_abs_diff = None
    max_diff_image = None
    if args.checkpoint:
        direct_metrics, max_abs_diff, max_diff_image = evaluate_direct_and_diff(
            args, names, dataset_dir, exports_dir, args.threshold
        )
        direct_vs_export = metric_deltas(direct_metrics, export_metrics)
        checks["direct_vs_export_metrics"] = metrics_pass(direct_vs_export)
        checks["direct_export_max_diff"] = max_abs_diff < MAX_DIFF_TOL

    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "method": args.method,
        "exports_dir": str(exports_dir),
        "summary": args.summary,
        "threshold_curve": args.threshold_curve,
        "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
        "model_name": args.model_name,
        "mshnet_export_head": resolve_mshnet_head(args.model_name, args.mshnet_export_head),
        "image_list": os.path.abspath(args.image_list) if args.image_list else None,
        "num_images": len(names),
        "threshold": args.threshold,
        "tolerances": {
            "mIoU": METRIC_TOL,
            "Pd": METRIC_TOL,
            "FA_ppm": FA_PPM_TOL,
            "direct_export_max_diff": MAX_DIFF_TOL,
        },
        "checks": checks,
        "export_metrics": export_metrics,
        "summary_metrics": summary_metrics,
        "curve_0p5_metrics": curve_metrics,
        "export_vs_summary_delta": export_vs_summary,
        "curve_0p5_vs_summary_delta": curve_vs_summary,
        "direct_metrics": direct_metrics,
        "direct_vs_export_delta": direct_vs_export,
        "direct_export_max_abs_diff": max_abs_diff,
        "direct_export_max_diff_image": max_diff_image,
    }
    (output_dir / "export_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Export Validation",
        "",
        f"Status: **{status}**",
        f"Method: `{args.method}`",
        f"Exports: `{exports_dir}`",
        f"Images: {len(names)}",
        "",
        "## Checks",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend([
        "",
        "## Metrics",
        "",
        f"- export mIoU/Pd/FA_ppm: {export_metrics['mIoU']:.9f} / {export_metrics['Pd']:.9f} / {export_metrics['FA_ppm']:.6f}",
        f"- summary mIoU/Pd/FA_ppm: {float(summary_metrics['mIoU']):.9f} / {float(summary_metrics['Pd']):.9f} / {float(summary_metrics['FA_ppm']):.6f}",
        f"- curve 0.5 mIoU/Pd/FA_ppm: {float(curve_metrics['mIoU']):.9f} / {float(curve_metrics['Pd']):.9f} / {float(curve_metrics['FA_ppm']):.6f}",
        f"- mean prob target/background: {export_metrics['mean_prob_target']:.9f} / {export_metrics['mean_prob_bg']:.9f}",
    ])
    if direct_metrics is not None:
        lines.extend([
            f"- direct mIoU/Pd/FA_ppm: {direct_metrics['mIoU']:.9f} / {direct_metrics['Pd']:.9f} / {direct_metrics['FA_ppm']:.6f}",
            f"- direct/export max abs diff: {max_abs_diff:.12f} on `{max_diff_image}`",
        ])
    (output_dir / "EXPORT_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output_dir / "export_validation.json"), "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
