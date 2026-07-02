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
from scipy.ndimage import distance_transform_edt
from skimage import measure
from torch.autograd import Variable
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg

MSHNET_NAMES = ("MSHNet", "MSHNetFocal", "MSHNetOHEM", "MSHNetTopKNeg", "MSHNetSPSOHEM")


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def connected_regions(mask):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def region_iou(region, target_mask):
    region_mask = np.zeros(target_mask.shape, dtype=bool)
    region_mask[region.coords[:, 0], region.coords[:, 1]] = True
    intersection = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(intersection, union)


def match_components(pred_mask, gt_mask, distance_threshold=3.0):
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


def init_fp_census():
    return {
        "matched_target_components": 0.0,
        "boundary_excess_instances": 0.0,
        "boundary_excess_pixel_mass": 0.0,
        "detached_near_fp_components": 0.0,
        "detached_near_fp_pixel_mass": 0.0,
        "far_fp_components": 0.0,
        "far_fp_pixel_mass": 0.0,
        "unmatched_fp_components": 0.0,
        "fp_pixel_mass": 0.0,
        "confidence_mass": 0.0,
        "boundary_excess_confidence_mass": 0.0,
        "detached_near_fp_confidence_mass": 0.0,
        "far_fp_confidence_mass": 0.0,
    }


def update_fp_census(census, component_rows, image_name, prob, pred_mask, gt_mask, threshold):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)
    for idx, region in enumerate(connected_regions(pred), start=1):
        component = np.zeros_like(pred, dtype=bool)
        component[region.coords[:, 0], region.coords[:, 1]] = True
        overlaps_gt = bool(np.logical_and(component, gt).any())
        fp_mask = component & (~gt)
        if overlaps_gt:
            census["matched_target_components"] += 1.0
        if not fp_mask.any():
            continue

        if overlaps_gt:
            category = "boundary_excess"
            min_distance = 0.0
            unmatched = 0
            census["boundary_excess_instances"] += 1.0
        else:
            min_distance = float(dist_to_gt[fp_mask].min())
            if min_distance <= 10.0:
                category = "detached_near_fp"
                census["detached_near_fp_components"] += 1.0
            else:
                category = "far_fp"
                census["far_fp_components"] += 1.0
            unmatched = 1
            census["unmatched_fp_components"] += 1.0

        fp_pixels = float(fp_mask.sum())
        confidence = float(prob[fp_mask].sum())
        census["fp_pixel_mass"] += fp_pixels
        census["confidence_mass"] += confidence
        census[f"{category}_pixel_mass"] += fp_pixels
        census[f"{category}_confidence_mass"] += confidence
        cy, cx = (float(v) for v in region.centroid)
        component_rows.append({
            "image_name": image_name,
            "threshold": threshold,
            "component_id": idx,
            "category": category,
            "matched_target_component": int(overlaps_gt),
            "unmatched_fp_component": unmatched,
            "component_area": int(region.area),
            "fp_pixel_mass": int(fp_pixels),
            "mean_probability": float(prob[fp_mask].mean()),
            "max_probability": float(prob[fp_mask].max()),
            "confidence_mass": confidence,
            "minimum_distance_to_gt": min_distance,
            "component_center_y": cy,
            "component_center_x": cx,
            "bbox_y0": int(region.bbox[0]),
            "bbox_y1": int(region.bbox[2]),
            "bbox_x0": int(region.bbox[1]),
            "bbox_x1": int(region.bbox[3]),
        })


def finalize_fp_census(census):
    total_fp_pixels = census["fp_pixel_mass"]
    total_confidence = census["confidence_mass"]
    near_pixels = census["boundary_excess_pixel_mass"] + census["detached_near_fp_pixel_mass"]
    near_confidence = census["boundary_excess_confidence_mass"] + census["detached_near_fp_confidence_mass"]
    unmatched = census["unmatched_fp_components"]
    census["target_near_pixel_mass"] = near_pixels
    census["target_near_confidence_mass"] = near_confidence
    census["R_near_pixel"] = near_pixels / max(1.0, total_fp_pixels)
    census["R_near_component"] = census["detached_near_fp_components"] / max(1.0, unmatched)
    census["R_near_confidence"] = near_confidence / max(1e-12, total_confidence)
    return census


def update_stats(stats, prob, gt_mask, threshold):
    pred = prob > threshold
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)

    item = stats[threshold]
    item["inter"] += float(inter)
    item["union"] += float(union)
    item["tp"] += float(tp)
    item["fp"] += float(fp)
    item["fn"] += float(fn)
    item["pixels"] += float(pred.size)
    item["niou_sum"] += safe_div(inter, union)
    item["count"] += 1
    item["matched_targets"] += float(matched_targets)
    item["target_components"] += float(target_components)
    item["fp_components"] += float(fp_components)


def stats_rows(stats):
    rows = []
    for threshold in sorted(stats):
        item = stats[threshold]
        precision = safe_div(item["tp"], item["tp"] + item["fp"])
        recall = safe_div(item["tp"], item["tp"] + item["fn"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        fa = safe_div(item["fp"], item["pixels"])
        rows.append(
            {
                "threshold": threshold,
                "mIoU": safe_div(item["inter"], item["union"]),
                "nIoU": safe_div(item["niou_sum"], item["count"]),
                "Pd": safe_div(item["matched_targets"], item["target_components"]),
                "detected_targets": item["matched_targets"],
                "target_components": item["target_components"],
                "FA": fa,
                "FA_ppm": fa * 1_000_000.0,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "FP_pixels": item["fp"],
                "GT_pixels": item["tp"] + item["fn"],
                "FP_components": item["fp_components"],
            }
        )
    return rows


def image_metrics(image_name, prob, pred, gt, threshold):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "image_name": image_name,
        "threshold": threshold,
        "IoU": safe_div(inter, union),
        "nIoU": safe_div(inter, union),
        "Pd": safe_div(matched_targets, target_components),
        "detected_targets": matched_targets,
        "target_components": target_components,
        "FA": safe_div(fp, pred.size),
        "FA_ppm": safe_div(fp, pred.size) * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": safe_div(2.0 * precision * recall, precision + recall),
        "FP_pixels": fp,
        "GT_pixels": int(gt.sum()),
        "FP_components": fp_components,
        "target_area": int(gt.sum()),
        "mean_prob_target": float(prob[gt].mean()) if gt.any() else 0.0,
        "mean_prob_bg": float(prob[~gt].mean()) if (~gt).any() else 0.0,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def torch_load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_checkpoint(net, checkpoint_path, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    return checkpoint


def resolve_mshnet_head(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "final"


def direct_probability(net, img, args, h, w):
    head = resolve_mshnet_head(args.model_name, args.mshnet_export_head)
    if args.model_name in MSHNET_NAMES and head == "output0":
        _, logit, _ = net.model(img, False, return_feature=True)
    else:
        logit = net.export_logits_features(img)["logit"]
    return foreground_probability(logit)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint directly without exporting prediction files.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--mshnet_export_head", default="auto", choices=["auto", "output0", "final"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--method", default="MSHNetOHEM")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset_name = args.train_dataset_name or args.dataset_name
    subset_name = "full_test"
    image_filter = None
    if args.image_list:
        image_filter = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]
        subset_name = Path(args.image_list).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    if image_filter is not None:
        test_set.test_list = image_filter
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)
    net.eval()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    threshold_stats = {
        threshold: {
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
        for threshold in thresholds
    }
    per_image_rows = []
    fp_census = init_fp_census()
    fp_component_rows = []

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(test_loader):
            img = Variable(img).to(device)
            gt_mask = gt_mask.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            prob = direct_probability(net, img, args, h, w)
            gt = gt_mask[0, 0, :h, :w].detach().cpu().numpy() > 0
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            pred = prob > args.threshold
            per_image_rows.append(image_metrics(name, prob, pred, gt, args.threshold))
            update_fp_census(fp_census, fp_component_rows, name, prob, pred, gt, args.threshold)
            for threshold in thresholds:
                update_stats(threshold_stats, prob, gt, threshold)
            if (idx + 1) % 100 == 0:
                print(f"Evaluated [{idx + 1}/{len(test_loader)}]", flush=True)

    per_image_fields = [
        "image_name",
        "threshold",
        "IoU",
        "nIoU",
        "Pd",
        "detected_targets",
        "target_components",
        "FA",
        "FA_ppm",
        "Precision",
        "Recall",
        "F1",
        "FP_pixels",
        "GT_pixels",
        "FP_components",
        "target_area",
        "mean_prob_target",
        "mean_prob_bg",
    ]
    write_csv(output_dir / "metrics_per_image.csv", per_image_rows, per_image_fields)
    fp_component_fields = [
        "image_name",
        "threshold",
        "component_id",
        "category",
        "matched_target_component",
        "unmatched_fp_component",
        "component_area",
        "fp_pixel_mass",
        "mean_probability",
        "max_probability",
        "confidence_mass",
        "minimum_distance_to_gt",
        "component_center_y",
        "component_center_x",
        "bbox_y0",
        "bbox_y1",
        "bbox_x0",
        "bbox_x1",
    ]
    write_csv(output_dir / "fp_components.csv", fp_component_rows, fp_component_fields)
    rows = stats_rows(threshold_stats)
    metric_fields = [
        "threshold",
        "mIoU",
        "nIoU",
        "Pd",
        "detected_targets",
        "target_components",
        "FA",
        "FA_ppm",
        "Precision",
        "Recall",
        "F1",
        "FP_pixels",
        "GT_pixels",
        "FP_components",
    ]
    write_csv(output_dir / "threshold_curve.csv", rows, metric_fields)
    summary = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "method": args.method,
        "model": args.model_name,
        "mshnet_export_head": resolve_mshnet_head(args.model_name, args.mshnet_export_head),
        "seed": args.seed,
        "subset": subset_name,
        "image_list": os.path.abspath(args.image_list) if args.image_list else None,
        "num_images": len(per_image_rows),
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "metrics_at_threshold": next(row for row in rows if abs(row["threshold"] - args.threshold) < 1e-9),
        "outputs": {
            "per_image_metrics": str(output_dir / "metrics_per_image.csv"),
            "threshold_curve": str(output_dir / "threshold_curve.csv"),
            "fp_components": str(output_dir / "fp_components.csv"),
        },
        "fp_census_at_threshold": finalize_fp_census(fp_census),
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics_at_threshold"], indent=2), flush=True)


if __name__ == "__main__":
    main()
