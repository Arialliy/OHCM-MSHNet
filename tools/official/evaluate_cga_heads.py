#!/usr/bin/env python3
"""Evaluate CGA final mask and auxiliary geometry diagnostics.

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
from utils.component_geometry import build_center_heatmap, build_core_boundary_maps, component_area_bins
from utils.local_peak import local_peak_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CGA final and geometry heads.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="test", choices=["test", "full", "train"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--model_name", default="CGAMSHNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate_scope", choices=["none", "full", "hcval"], default="none")
    parser.add_argument("--mshnet_warm_epoch", type=int, default=0)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--cga_num_scale_bins", type=int, default=4)
    parser.add_argument("--target_dilate_radius", type=int, default=3)
    parser.add_argument("--far_dilate_radius", type=int, default=10)
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


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_checkpoint(net: Net, checkpoint_path: str, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
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


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def geometry_diagnostics(export: dict, gt_mask: torch.Tensor, final_logits: torch.Tensor, threshold: float) -> dict:
    center_target, _scale_map, _scale_valid_float = build_center_heatmap(gt_mask.float())
    scale_target, scale_valid, _counts = component_area_bins(gt_mask.float())
    _core_target, boundary_target, _ignore = build_core_boundary_maps(gt_mask.float())

    center_prob = torch.sigmoid(export["center_logit"])
    center_peaks = center_target >= 0.999
    center_peak_prob = float(center_prob[center_peaks].mean().item()) if center_peaks.any() else float("nan")
    center_peak_hit_rate = (
        float((center_prob[center_peaks] > threshold).float().mean().item()) if center_peaks.any() else float("nan")
    )

    scale_pred = export["geometry_scale_logits"].argmax(dim=1)
    scale_valid_2d = scale_valid[:, 0]
    scale_acc = (
        float((scale_pred[scale_valid_2d] == scale_target[scale_valid_2d]).float().mean().item())
        if scale_valid_2d.any()
        else float("nan")
    )

    final_prob = foreground_probability(final_logits)
    peaks = local_peak_mask(final_prob) & (gt_mask < 0.5)
    local_peak_false_alarm_count = int(((final_prob > threshold) & peaks).sum().item())
    boundary_prob = foreground_probability(export["boundary_logit"])
    boundary_prob_mean = float(boundary_prob[boundary_target > 0.5].mean().item()) if boundary_target.any() else float("nan")

    return {
        "center_peak_prob": center_peak_prob,
        "center_peak_hit_rate": center_peak_hit_rate,
        "scale_acc_on_target": scale_acc,
        "local_peak_false_alarm_count": local_peak_false_alarm_count,
        "boundary_prob_mean": boundary_prob_mean,
    }


def gate_decision(scope: str, cga: dict, ohem: dict) -> tuple[bool, list[str], dict]:
    if scope == "none":
        return True, [], {}
    if scope == "full":
        checks = {
            "mIoU": cga["mIoU"] >= ohem["mIoU"] - 0.001,
            "Pd": cga["Pd"] >= ohem["Pd"],
            "Precision": cga["Precision"] >= ohem["Precision"] - 0.002,
            "FA_ppm": cga["FA_ppm"] <= ohem["FA_ppm"] + 2.0,
            "FP_components": cga["FP_components"] <= ohem["FP_components"] + 5.0,
        }
    elif scope == "hcval":
        checks = {
            "mIoU": cga["mIoU"] >= ohem["mIoU"] + 0.005,
            "Pd": cga["Pd"] >= ohem["Pd"],
            "Precision": cga["Precision"] >= ohem["Precision"] + 0.005,
            "FA_ppm": cga["FA_ppm"] <= ohem["FA_ppm"] - 10.0,
        }
    else:
        raise ValueError(scope)
    fail_reasons = [f"{scope}_{name}_gate_failed" for name, passed in checks.items() if not passed]
    return len(fail_reasons) == 0, fail_reasons, checks


def main() -> int:
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

    cga_net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(cga_net, args.checkpoint, device)
    ohem_net = Net(model_name="MSHNetOHEM", mode="test", loss_cfg=vars(args)).to(device)
    load_checkpoint(ohem_net, args.ohem_checkpoint, device)

    cga_stats = init_stats()
    ohem_stats = init_stats()
    per_image_rows = []
    totals = {
        "new_fp_pixels": 0,
        "new_fp_components": 0,
        "removed_fp_pixels": 0,
        "removed_fp_components": 0,
        "lost_target_pixels": 0,
        "lost_target_count": 0,
        "boundary_excess_pixels": 0,
    }
    center_peak_probs = []
    center_peak_hit_rates = []
    scale_accs = []
    local_peak_false_alarms = []
    boundary_prob_means = []

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt_t = gt_mask[:, :, :h, :w].to(device)
            gt_np = gt_mask[0, 0, :h, :w].numpy() > 0

            export = cga_net.export_logits_features(img)
            cga_logits = export["logit"][:, :, :h, :w]
            cga_prob = foreground_probability(cga_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            ohem_prob = ohem_net(img, epoch=999)[:, :, :h, :w][0, 0].detach().cpu().numpy().astype(np.float32)

            cga_metrics = update_stats(cga_stats, cga_prob, gt_np, args.threshold)
            ohem_metrics = update_stats(ohem_stats, ohem_prob, gt_np, args.threshold)

            cga_mask = cga_prob > args.threshold
            ohem_mask = ohem_prob > args.threshold
            bg = ~gt_np
            boundary = binary_dilate(gt_np, args.target_dilate_radius) & (~gt_np)
            new_fp = cga_mask & (~ohem_mask) & bg
            removed_fp = ohem_mask & (~cga_mask) & bg
            lost_target = ohem_mask & (~cga_mask) & gt_np
            boundary_excess = cga_mask & (~ohem_mask) & boundary
            new_fp_components = count_components(new_fp)
            removed_fp_components = count_components(removed_fp)
            lost_target_count = max(0, ohem_metrics["Pd"] > cga_metrics["Pd"])

            geo = geometry_diagnostics(
                {
                    "center_logit": export["center_logit"][:, :, :h, :w],
                    "geometry_scale_logits": export["geometry_scale_logits"][:, :, :h, :w],
                    "boundary_logit": export["boundary_logit"][:, :, :h, :w],
                },
                gt_t,
                cga_logits,
                args.threshold,
            )
            for values, key in (
                (center_peak_probs, "center_peak_prob"),
                (center_peak_hit_rates, "center_peak_hit_rate"),
                (scale_accs, "scale_acc_on_target"),
                (boundary_prob_means, "boundary_prob_mean"),
            ):
                if np.isfinite(geo[key]):
                    values.append(float(geo[key]))
            local_peak_false_alarms.append(int(geo["local_peak_false_alarm_count"]))

            totals["new_fp_pixels"] += int(new_fp.sum())
            totals["new_fp_components"] += int(new_fp_components)
            totals["removed_fp_pixels"] += int(removed_fp.sum())
            totals["removed_fp_components"] += int(removed_fp_components)
            totals["lost_target_pixels"] += int(lost_target.sum())
            totals["lost_target_count"] += int(lost_target_count)
            totals["boundary_excess_pixels"] += int(boundary_excess.sum())

            per_image_rows.append(
                {
                    "image_name": name,
                    "ohem_mIoU": ohem_metrics["mIoU"],
                    "cga_mIoU": cga_metrics["mIoU"],
                    "ohem_Pd": ohem_metrics["Pd"],
                    "cga_Pd": cga_metrics["Pd"],
                    "ohem_Precision": ohem_metrics["Precision"],
                    "cga_Precision": cga_metrics["Precision"],
                    "ohem_FA_ppm": ohem_metrics["FA_ppm"],
                    "cga_FA_ppm": cga_metrics["FA_ppm"],
                    "ohem_FP_components": ohem_metrics["FP_components"],
                    "cga_FP_components": cga_metrics["FP_components"],
                    "new_fp_pixels": int(new_fp.sum()),
                    "new_fp_components": int(new_fp_components),
                    "removed_fp_pixels": int(removed_fp.sum()),
                    "removed_fp_components": int(removed_fp_components),
                    "lost_target_pixels": int(lost_target.sum()),
                    "boundary_excess_pixels": int(boundary_excess.sum()),
                    **geo,
                }
            )

            if (idx + 1) % 100 == 0:
                print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    ohem = finalize_stats(ohem_stats)
    cga = finalize_stats(cga_stats)
    gate_pass, fail_reasons, gate_checks = gate_decision(args.gate_scope, cga, ohem)
    summary = {
        "gate_pass": gate_pass,
        "fail_reasons": fail_reasons,
        "gate_scope": args.gate_scope,
        "gate_checks": gate_checks,
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "split": args.split,
        "image_source": image_source,
        "num_images": len(per_image_rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "ohem_checkpoint": str(Path(args.ohem_checkpoint).resolve()),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "ohem": ohem,
        "cga": cga,
        "ohem_mIoU": ohem["mIoU"],
        "cga_mIoU": cga["mIoU"],
        "ohem_Pd": ohem["Pd"],
        "cga_Pd": cga["Pd"],
        "ohem_Precision": ohem["Precision"],
        "cga_Precision": cga["Precision"],
        "ohem_FA_ppm": ohem["FA_ppm"],
        "cga_FA_ppm": cga["FA_ppm"],
        "ohem_FP_components": ohem["FP_components"],
        "cga_FP_components": cga["FP_components"],
        **totals,
        "center_peak_prob_mean": mean_or_nan(center_peak_probs),
        "center_peak_hit_rate_mean": mean_or_nan(center_peak_hit_rates),
        "scale_acc_on_target_mean": mean_or_nan(scale_accs),
        "local_peak_false_alarm_count": int(sum(local_peak_false_alarms)),
        "boundary_prob_mean": mean_or_nan(boundary_prob_means),
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "worst_images": str(output_dir / "worst_images.csv"),
        },
    }

    fields = [
        "image_name",
        "ohem_mIoU",
        "cga_mIoU",
        "ohem_Pd",
        "cga_Pd",
        "ohem_Precision",
        "cga_Precision",
        "ohem_FA_ppm",
        "cga_FA_ppm",
        "ohem_FP_components",
        "cga_FP_components",
        "new_fp_pixels",
        "new_fp_components",
        "removed_fp_pixels",
        "removed_fp_components",
        "lost_target_pixels",
        "boundary_excess_pixels",
        "center_peak_prob",
        "center_peak_hit_rate",
        "scale_acc_on_target",
        "local_peak_false_alarm_count",
        "boundary_prob_mean",
    ]
    write_csv(output_dir / "per_image.csv", per_image_rows, fields)
    worst_rows = sorted(
        per_image_rows,
        key=lambda row: (-int(row["new_fp_components"]), -int(row["lost_target_pixels"])),
    )[:50]
    write_csv(output_dir / "worst_images.csv", worst_rows, fields)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
