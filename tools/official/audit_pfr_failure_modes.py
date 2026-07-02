#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import measure
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit PFR-MSHNet failure modes against MSHNetOHEM.")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--dataset", "--dataset_name", dest="dataset_name", default="NUDT-SIRST")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="full", choices=["full", "train"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--ohem_model_name", default="MSHNetOHEM")
    parser.add_argument("--candidate_model_name", default="PFRMSHNet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_dilate_radius", type=int, default=3)
    parser.add_argument("--far_dilate_radius", type=int, default=10)
    parser.add_argument("--near_fp_radius", type=float, default=10.0)
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


def region_to_mask(region, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[region.coords[:, 0], region.coords[:, 1]] = True
    return mask


def region_iou(region, target_mask: np.ndarray) -> float:
    region_mask = region_to_mask(region, target_mask.shape)
    inter = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(inter, union)


def match_targets(pred_mask: np.ndarray, gt_mask: np.ndarray, distance_threshold: float = 3.0):
    pred_regions = connected_regions(pred_mask)
    gt_regions = connected_regions(gt_mask)
    used_pred = set()
    matched = []
    for gt_region in gt_regions:
        gt_centroid = np.asarray(gt_region.centroid)
        matched_this = False
        for pred_idx, pred_region in enumerate(pred_regions):
            if pred_idx in used_pred:
                continue
            pred_centroid = np.asarray(pred_region.centroid)
            if np.linalg.norm(pred_centroid - gt_centroid) < distance_threshold:
                used_pred.add(pred_idx)
                matched_this = True
                break
        matched.append(matched_this)
    return matched, len(gt_regions)


def image_metrics(prob: np.ndarray, gt: np.ndarray, threshold: float):
    pred = prob > threshold
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched, target_components = match_targets(pred, gt)
    fp_components = sum(1 for region in connected_regions(pred) if region_iou(region, gt) <= 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "mIoU": safe_div(inter, union),
        "Pd": safe_div(sum(matched), target_components),
        "Precision": precision,
        "Recall": recall,
        "F1": safe_div(2.0 * precision * recall, precision + recall),
        "FA_ppm": safe_div(fp, pred.size) * 1_000_000.0,
        "FP_pixels": int(fp),
        "FP_components": int(fp_components),
        "target_components": int(target_components),
        "matched_targets": int(sum(matched)),
    }


def component_census(pred_mask: np.ndarray, gt_mask: np.ndarray, prob: np.ndarray, near_radius: float):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)
    totals = {
        "boundary_excess_components": 0,
        "boundary_excess_pixels": 0,
        "detached_near_fp_components": 0,
        "detached_near_fp_pixels": 0,
        "far_fp_components": 0,
        "far_fp_pixels": 0,
        "fp_components": 0,
        "fp_pixels": 0,
    }
    components = []
    for idx, region in enumerate(connected_regions(pred), start=1):
        component = region_to_mask(region, pred.shape)
        overlaps_gt = bool(np.logical_and(component, gt).any())
        fp_mask = component & (~gt)
        if not fp_mask.any():
            continue
        if overlaps_gt:
            category = "boundary_excess"
            min_distance = 0.0
        else:
            min_distance = float(dist_to_gt[fp_mask].min())
            category = "detached_near_fp" if min_distance <= near_radius else "far_fp"
            totals["fp_components"] += 1
        fp_pixels = int(fp_mask.sum())
        totals[f"{category}_components"] += 1
        totals[f"{category}_pixels"] += fp_pixels
        totals["fp_pixels"] += fp_pixels
        components.append(
            {
                "component_id": idx,
                "category": category,
                "area": int(region.area),
                "fp_pixels": fp_pixels,
                "mean_probability": float(prob[fp_mask].mean()),
                "max_probability": float(prob[fp_mask].max()),
                "minimum_distance_to_gt": min_distance,
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "mask": component,
            }
        )
    return totals, components


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(net: Net, checkpoint_path: str, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def forward_prob(net: Net, img: torch.Tensor, h: int, w: int):
    export = net.export_logits_features(img)
    logit = export["logit"][:, :, :h, :w]
    return foreground_probability(logit)[0, 0].detach().cpu().numpy().astype(np.float32)


def resolve_image_list(args, dataset):
    if args.image_list:
        path = Path(args.image_list)
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path.resolve())
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path.resolve())
    return list(dataset.test_list), "full_test"


def compare_fp_components(image_name, ohem_components, candidate_components):
    rows = []
    matched_candidate = set()
    for o_idx, comp in enumerate(ohem_components):
        best_overlap = 0
        best_candidate = None
        for c_idx, cand in enumerate(candidate_components):
            overlap = int(np.logical_and(comp["mask"], cand["mask"]).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_candidate = c_idx
        status = "retained" if best_overlap > 0 else "removed"
        if status == "retained":
            matched_candidate.add(best_candidate)
        rows.append(
            {
                "image_name": image_name,
                "source": "ohem",
                "status": status,
                "component_id": comp["component_id"],
                "matched_component_id": (
                    candidate_components[best_candidate]["component_id"]
                    if best_candidate is not None and best_overlap > 0
                    else ""
                ),
                "category": comp["category"],
                "area": comp["area"],
                "fp_pixels": comp["fp_pixels"],
                "overlap_pixels": best_overlap,
                "mean_probability": comp["mean_probability"],
                "max_probability": comp["max_probability"],
                "minimum_distance_to_gt": comp["minimum_distance_to_gt"],
                "centroid_y": comp["centroid_y"],
                "centroid_x": comp["centroid_x"],
            }
        )
    for c_idx, comp in enumerate(candidate_components):
        if c_idx in matched_candidate:
            continue
        rows.append(
            {
                "image_name": image_name,
                "source": "candidate",
                "status": "new",
                "component_id": comp["component_id"],
                "matched_component_id": "",
                "category": comp["category"],
                "area": comp["area"],
                "fp_pixels": comp["fp_pixels"],
                "overlap_pixels": 0,
                "mean_probability": comp["mean_probability"],
                "max_probability": comp["max_probability"],
                "minimum_distance_to_gt": comp["minimum_distance_to_gt"],
                "centroid_y": comp["centroid_y"],
                "centroid_x": comp["centroid_x"],
            }
        )
    return rows


def audit_image(name, ohem_prob, candidate_prob, gt, args):
    target = gt.astype(bool)
    target_dilate = binary_dilate(target, args.target_dilate_radius)
    far_bg = ~binary_dilate(target, args.far_dilate_radius)
    ohem_pred = ohem_prob > args.threshold
    candidate_pred = candidate_prob > args.threshold
    removed_by_candidate = ohem_pred & (~candidate_pred)

    ohem_metrics = image_metrics(ohem_prob, target, args.threshold)
    candidate_metrics = image_metrics(candidate_prob, target, args.threshold)
    ohem_matched, target_components = match_targets(ohem_pred, target)
    candidate_matched, _ = match_targets(candidate_pred, target)
    target_lost_count = sum(1 for before, after in zip(ohem_matched, candidate_matched) if before and not after)
    target_gained_count = sum(1 for before, after in zip(ohem_matched, candidate_matched) if (not before) and after)

    ohem_census, ohem_components = component_census(ohem_pred, target, ohem_prob, args.near_fp_radius)
    candidate_census, candidate_components = component_census(
        candidate_pred, target, candidate_prob, args.near_fp_radius
    )
    component_rows = compare_fp_components(name, ohem_components, candidate_components)
    removed_fp_components = sum(1 for row in component_rows if row["source"] == "ohem" and row["status"] == "removed")
    retained_fp_components = sum(1 for row in component_rows if row["source"] == "ohem" and row["status"] == "retained")
    new_fp_components = sum(1 for row in component_rows if row["source"] == "candidate" and row["status"] == "new")

    target_shrink_pixels = int((removed_by_candidate & target).sum())
    target_over_suppressed_pixels = int((removed_by_candidate & target_dilate).sum())
    removed_far_fp_pixels = int((removed_by_candidate & far_bg & (~target)).sum())

    row = {
        "image_name": name,
        "target_components": target_components,
        "ohem_mIoU": ohem_metrics["mIoU"],
        "candidate_mIoU": candidate_metrics["mIoU"],
        "per_image_delta_mIoU": candidate_metrics["mIoU"] - ohem_metrics["mIoU"],
        "ohem_Pd": ohem_metrics["Pd"],
        "candidate_Pd": candidate_metrics["Pd"],
        "per_image_delta_Pd": candidate_metrics["Pd"] - ohem_metrics["Pd"],
        "ohem_Precision": ohem_metrics["Precision"],
        "candidate_Precision": candidate_metrics["Precision"],
        "per_image_delta_precision": candidate_metrics["Precision"] - ohem_metrics["Precision"],
        "ohem_FA_ppm": ohem_metrics["FA_ppm"],
        "candidate_FA_ppm": candidate_metrics["FA_ppm"],
        "per_image_delta_FA": candidate_metrics["FA_ppm"] - ohem_metrics["FA_ppm"],
        "ohem_FP_components": ohem_metrics["FP_components"],
        "candidate_FP_components": candidate_metrics["FP_components"],
        "delta_FP_components": candidate_metrics["FP_components"] - ohem_metrics["FP_components"],
        "target_lost_count": target_lost_count,
        "target_gained_count": target_gained_count,
        "target_shrink_pixels": target_shrink_pixels,
        "target_over_suppressed_pixels": target_over_suppressed_pixels,
        "removed_far_fp_pixels": removed_far_fp_pixels,
        "new_fp_components": new_fp_components,
        "removed_fp_components": removed_fp_components,
        "retained_fp_components": retained_fp_components,
        "ohem_boundary_excess_pixels": ohem_census["boundary_excess_pixels"],
        "candidate_boundary_excess_pixels": candidate_census["boundary_excess_pixels"],
        "boundary_excess_delta": candidate_census["boundary_excess_pixels"] - ohem_census["boundary_excess_pixels"],
        "ohem_far_fp_pixels": ohem_census["far_fp_pixels"],
        "candidate_far_fp_pixels": candidate_census["far_fp_pixels"],
        "far_fp_delta": candidate_census["far_fp_pixels"] - ohem_census["far_fp_pixels"],
        "ohem_detached_near_fp_pixels": ohem_census["detached_near_fp_pixels"],
        "candidate_detached_near_fp_pixels": candidate_census["detached_near_fp_pixels"],
        "detached_near_fp_delta": candidate_census["detached_near_fp_pixels"]
        - ohem_census["detached_near_fp_pixels"],
        "target_prob_ohem": float(ohem_prob[target].mean()) if target.any() else float("nan"),
        "target_prob_candidate": float(candidate_prob[target].mean()) if target.any() else float("nan"),
        "target_prob_delta": (
            float(candidate_prob[target].mean() - ohem_prob[target].mean()) if target.any() else float("nan")
        ),
        "far_bg_prob_ohem": float(ohem_prob[far_bg].mean()) if far_bg.any() else float("nan"),
        "far_bg_prob_candidate": float(candidate_prob[far_bg].mean()) if far_bg.any() else float("nan"),
        "far_bg_prob_delta": (
            float(candidate_prob[far_bg].mean() - ohem_prob[far_bg].mean()) if far_bg.any() else float("nan")
        ),
    }
    return row, component_rows


def aggregate(rows, key: str, op="sum"):
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    if not values:
        return float("nan")
    if op == "mean":
        return float(np.mean(values))
    return float(np.sum(values))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, image_source = resolve_image_list(args, dataset)
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    ohem_net = Net(model_name=args.ohem_model_name, mode="test", loss_cfg=vars(args)).to(device)
    candidate_net = Net(model_name=args.candidate_model_name, mode="test", loss_cfg=vars(args)).to(device)
    ohem_checkpoint = load_checkpoint(ohem_net, args.ohem_checkpoint, device)
    candidate_checkpoint = load_checkpoint(candidate_net, args.candidate_checkpoint, device)

    rows = []
    component_rows = []
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            ohem_prob = forward_prob(ohem_net, img, h, w)
            candidate_prob = forward_prob(candidate_net, img, h, w)
            row, comp_rows = audit_image(name, ohem_prob, candidate_prob, gt, args)
            rows.append(row)
            component_rows.extend(comp_rows)
            if (idx + 1) % 100 == 0:
                print("Audited [%d/%d]" % (idx + 1, len(loader)), flush=True)

    per_image_fields = [
        "image_name",
        "target_components",
        "ohem_mIoU",
        "candidate_mIoU",
        "per_image_delta_mIoU",
        "ohem_Pd",
        "candidate_Pd",
        "per_image_delta_Pd",
        "ohem_Precision",
        "candidate_Precision",
        "per_image_delta_precision",
        "ohem_FA_ppm",
        "candidate_FA_ppm",
        "per_image_delta_FA",
        "ohem_FP_components",
        "candidate_FP_components",
        "delta_FP_components",
        "target_lost_count",
        "target_gained_count",
        "target_shrink_pixels",
        "target_over_suppressed_pixels",
        "removed_far_fp_pixels",
        "new_fp_components",
        "removed_fp_components",
        "retained_fp_components",
        "ohem_boundary_excess_pixels",
        "candidate_boundary_excess_pixels",
        "boundary_excess_delta",
        "ohem_far_fp_pixels",
        "candidate_far_fp_pixels",
        "far_fp_delta",
        "ohem_detached_near_fp_pixels",
        "candidate_detached_near_fp_pixels",
        "detached_near_fp_delta",
        "target_prob_ohem",
        "target_prob_candidate",
        "target_prob_delta",
        "far_bg_prob_ohem",
        "far_bg_prob_candidate",
        "far_bg_prob_delta",
    ]
    write_csv(out_dir / "per_image.csv", rows, per_image_fields)
    worst_rows = sorted(rows, key=lambda row: (float(row["per_image_delta_mIoU"]), -float(row["per_image_delta_FA"])))[:50]
    write_csv(out_dir / "worst_images.csv", worst_rows, per_image_fields)

    component_fields = [
        "image_name",
        "source",
        "status",
        "component_id",
        "matched_component_id",
        "category",
        "area",
        "fp_pixels",
        "overlap_pixels",
        "mean_probability",
        "max_probability",
        "minimum_distance_to_gt",
        "centroid_y",
        "centroid_x",
    ]
    write_csv(out_dir / "component_delta.csv", component_rows, component_fields)

    total_target_lost = int(aggregate(rows, "target_lost_count"))
    total_target_shrink = int(aggregate(rows, "target_shrink_pixels"))
    total_target_over_suppressed = int(aggregate(rows, "target_over_suppressed_pixels"))
    total_new_fp = int(aggregate(rows, "new_fp_components"))
    total_removed_fp = int(aggregate(rows, "removed_fp_components"))
    total_retained_fp = int(aggregate(rows, "retained_fp_components"))
    total_boundary_delta = int(aggregate(rows, "boundary_excess_delta"))
    total_far_fp_delta = int(aggregate(rows, "far_fp_delta"))
    structural_failure = (
        total_target_lost > 0
        or total_target_shrink > 0
        or total_boundary_delta > 0
        or total_new_fp > total_removed_fp
        or total_far_fp_delta > 0
    )
    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "image_source": image_source,
        "num_images": len(rows),
        "threshold": args.threshold,
        "ohem_checkpoint": str(Path(args.ohem_checkpoint).resolve()),
        "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "ohem_epoch": ohem_checkpoint.get("epoch") if isinstance(ohem_checkpoint, dict) else None,
        "candidate_epoch": candidate_checkpoint.get("epoch") if isinstance(candidate_checkpoint, dict) else None,
        "mean_per_image_delta_mIoU": aggregate(rows, "per_image_delta_mIoU", op="mean"),
        "mean_per_image_delta_precision": aggregate(rows, "per_image_delta_precision", op="mean"),
        "mean_per_image_delta_FA": aggregate(rows, "per_image_delta_FA", op="mean"),
        "total_target_lost_count": total_target_lost,
        "total_target_shrink_pixels": total_target_shrink,
        "total_target_over_suppressed_pixels": total_target_over_suppressed,
        "total_new_fp_components": total_new_fp,
        "total_removed_fp_components": total_removed_fp,
        "total_retained_fp_components": total_retained_fp,
        "total_boundary_excess_delta": total_boundary_delta,
        "total_far_fp_delta": total_far_fp_delta,
        "mean_target_prob_delta": aggregate(rows, "target_prob_delta", op="mean"),
        "mean_far_bg_prob_delta": aggregate(rows, "far_bg_prob_delta", op="mean"),
        "failure_mode": (
            "structural_suppression_or_calibration_regression"
            if structural_failure
            else "no_structural_regression_detected"
        ),
        "structural_failure_flag": bool(structural_failure),
        "outputs": {
            "per_image": str(out_dir / "per_image.csv"),
            "worst_images": str(out_dir / "worst_images.csv"),
            "component_delta": str(out_dir / "component_delta.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
