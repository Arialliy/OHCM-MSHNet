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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from tools.official.check_bcv_gate_a import checkpoint_state_dict, torch_load_checkpoint
from tools.official.check_bcv_gate_b import binary_auc, fixed_context_background
from tools.official.check_bcv_gate_c_fp_residual import (
    binary_dilate,
    connected_components,
    finite_ge,
    median_or_zero,
    prediction_matches_target,
    quantile_or_zero,
    resolve_image_ids,
    size_to_int,
    suppressible_rate_at_target_recall,
)
from utils import get_img_norm_cfg
from utils.residual_shape_features import parse_shape_weights, residual_shape_features


def parse_args():
    parser = argparse.ArgumentParser(description="BCV Gate-D residual shape / morphology audit.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test", "full"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--bcv_checkpoint_or_init", default="fixed_context_background")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--target_near_radius", type=int, default=7)
    parser.add_argument("--match_distance", type=float, default=3.0)
    parser.add_argument("--background_kernel", type=int, default=31)
    parser.add_argument("--shape_ring_radius", type=int, default=3)
    parser.add_argument("--dog_sigma_small", type=float, default=1.0)
    parser.add_argument("--dog_sigma_large", type=float, default=2.0)
    parser.add_argument("--shape_weights", default=None)
    parser.add_argument("--min_far_fp_components", type=int, default=20)
    parser.add_argument("--min_shape_auc", type=float, default=0.70)
    parser.add_argument("--hard_stop_min_shape_auc", type=float, default=0.65)
    parser.add_argument("--min_single_feature_auc", type=float, default=0.70)
    parser.add_argument("--min_suppressible_far_fp_rate_99", type=float, default=0.20)
    parser.add_argument("--min_suppressible_far_fp_rate_995", type=float, default=0.10)
    parser.add_argument("--hard_stop_min_suppressible_far_fp_rate_99", type=float, default=0.10)
    return parser.parse_args()


def finite_values(rows: list[dict], component_type: str, key: str, sign: float = 1.0) -> list[float]:
    values = []
    for row in rows:
        if row.get("component_type") != component_type:
            continue
        value = float(row[key])
        if np.isfinite(value):
            values.append(sign * value)
    return values


def feature_auc(rows: list[dict], key: str, higher_target_is_better: bool = True) -> float:
    sign = 1.0 if higher_target_is_better else -1.0
    target = finite_values(rows, "gt_target", key, sign=sign)
    far_fp = finite_values(rows, "far_fp", key, sign=sign)
    return binary_auc(target, far_fp)


def component_shape_row(
    image_name: str,
    component_type: str,
    component_id: int,
    component,
    residual: np.ndarray,
    args,
    weights,
) -> dict:
    features = residual_shape_features(
        residual=residual,
        coords=component.coords,
        ring_radius=args.shape_ring_radius,
        sigma_small=args.dog_sigma_small,
        sigma_large=args.dog_sigma_large,
        weights=weights,
    )
    centroid = component.centroid
    row = {
        "image_name": image_name,
        "component_type": component_type,
        "component_id": int(component_id),
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
    }
    row.update(features)
    return row


def audit_image_shape_components(
    image_name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    residual: np.ndarray,
    args,
    weights,
) -> tuple[list[dict], dict]:
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    if gt_mask.shape != pred_mask.shape or gt_mask.shape != residual.shape:
        raise ValueError("gt_mask, pred_mask, and residual must have the same shape")

    gt_components = connected_components(gt_mask)
    pred_components = connected_components(pred_mask)
    target_near = binary_dilate(gt_mask, args.target_near_radius)
    rows = []
    counts = {
        "image_name": image_name,
        "target_component_count": len(gt_components),
        "matched_target_prediction_component_count": 0,
        "near_fp_component_count": 0,
        "far_fp_component_count": 0,
        "target_leakage_pixels": 0,
    }

    for component_id, component in enumerate(gt_components, start=1):
        rows.append(component_shape_row(image_name, "gt_target", component_id, component, residual, args, weights))

    for component_id, component in enumerate(pred_components, start=1):
        if prediction_matches_target(component, gt_components, gt_mask, args.match_distance):
            counts["matched_target_prediction_component_count"] += 1
            rows.append(
                component_shape_row(
                    image_name,
                    "matched_target_prediction",
                    component_id,
                    component,
                    residual,
                    args,
                    weights,
                )
            )
            continue

        leakage = int(gt_mask[component.coords[:, 0], component.coords[:, 1]].sum())
        counts["target_leakage_pixels"] += leakage
        if bool(target_near[component.coords[:, 0], component.coords[:, 1]].any()):
            counts["near_fp_component_count"] += 1
            rows.append(component_shape_row(image_name, "near_fp", component_id, component, residual, args, weights))
        else:
            counts["far_fp_component_count"] += 1
            rows.append(component_shape_row(image_name, "far_fp", component_id, component, residual, args, weights))

    return rows, counts


def build_gate_d_summary(component_rows: list[dict], image_rows: list[dict], args, metadata: dict | None = None) -> dict:
    target_shape = finite_values(component_rows, "gt_target", "shape_score")
    far_shape = finite_values(component_rows, "far_fp", "shape_score")
    near_shape = finite_values(component_rows, "near_fp", "shape_score")
    suppress99, theta99 = suppressible_rate_at_target_recall(target_shape, far_shape, 0.99)
    suppress995, theta995 = suppressible_rate_at_target_recall(target_shape, far_shape, 0.995)
    shape_auc_far = binary_auc(target_shape, far_shape)
    shape_auc_near = binary_auc(target_shape, near_shape)
    target_q10 = quantile_or_zero(target_shape, 0.10)
    far_median = median_or_zero(far_shape)
    target_leakage = int(sum(int(row["target_leakage_pixels"]) for row in image_rows))

    feature_aucs = {
        "compactness_auc": feature_auc(component_rows, "compactness", higher_target_is_better=True),
        "bbox_fill_ratio_auc": feature_auc(component_rows, "bbox_fill_ratio", higher_target_is_better=True),
        "anisotropy_auc": feature_auc(component_rows, "anisotropy", higher_target_is_better=False),
        "center_surround_auc": feature_auc(component_rows, "center_surround", higher_target_is_better=True),
        "radial_symmetry_auc": feature_auc(component_rows, "radial_symmetry", higher_target_is_better=True),
        "dog_peakness_auc": feature_auc(component_rows, "dog_peakness", higher_target_is_better=True),
    }
    max_single_feature_auc = max(feature_aucs.values()) if feature_aucs else 0.5
    checks = {
        "far_fp_component_count": len(far_shape) >= args.min_far_fp_components,
        "shape_auc_target_vs_far_fp": finite_ge(shape_auc_far, args.min_shape_auc),
        "suppressible_far_fp_rate_at_target_recall_99": finite_ge(suppress99, args.min_suppressible_far_fp_rate_99),
        "suppressible_far_fp_rate_at_target_recall_995": finite_ge(suppress995, args.min_suppressible_far_fp_rate_995),
        "single_feature_auc": finite_ge(max_single_feature_auc, args.min_single_feature_auc),
        "target_leakage": target_leakage == 0,
    }
    stop_conditions = {
        "far_fp_component_count_too_low": len(far_shape) < args.min_far_fp_components,
        "shape_auc_below_hard_stop": bool(shape_auc_far < args.hard_stop_min_shape_auc),
        "suppressible_far_fp_rate_99_below_hard_stop": bool(
            suppress99 < args.hard_stop_min_suppressible_far_fp_rate_99
        ),
        "distribution_overlap_target_q10_le_far_fp_median": bool(target_q10 <= far_median),
    }
    summary = {
        "gate": "BCV_Gate_D_residual_shape_morphology_audit",
        "target_component_count": len(target_shape),
        "matched_target_prediction_component_count": len(finite_values(component_rows, "matched_target_prediction", "shape_score")),
        "far_fp_component_count": len(far_shape),
        "near_fp_component_count": len(near_shape),
        "shape_auc_target_vs_far_fp": shape_auc_far,
        "shape_auc_target_vs_near_fp": shape_auc_near,
        "target_shape_q10": target_q10,
        "target_shape_median": median_or_zero(target_shape),
        "far_fp_shape_median": far_median,
        "near_fp_shape_median": median_or_zero(near_shape),
        "suppressible_far_fp_rate_at_target_recall_99": suppress99,
        "suppressible_far_fp_rate_at_target_recall_995": suppress995,
        "target_protection_shape_threshold": theta99,
        "target_protection_shape_threshold_995": theta995,
        **feature_aucs,
        "max_single_feature_auc": max_single_feature_auc,
        "target_leakage_pixels_total": target_leakage,
        "checks": checks,
        "stop_conditions": stop_conditions,
        "thresholds": {
            "min_far_fp_components": args.min_far_fp_components,
            "shape_auc_target_vs_far_fp_min": args.min_shape_auc,
            "hard_stop_min_shape_auc": args.hard_stop_min_shape_auc,
            "suppressible_far_fp_rate_at_target_recall_99_min": args.min_suppressible_far_fp_rate_99,
            "suppressible_far_fp_rate_at_target_recall_995_min": args.min_suppressible_far_fp_rate_995,
            "hard_stop_min_suppressible_far_fp_rate_99": args.hard_stop_min_suppressible_far_fp_rate_99,
            "single_feature_auc_min": args.min_single_feature_auc,
            "target_leakage_pixels_total": 0,
        },
        "gate_pass": bool(all(checks.values())),
    }
    summary["overall_decision"] = "PROCEED_TO_DETERMINISTIC_SHAPE_CALIBRATION" if summary["gate_pass"] else "STOP_BCV"
    if metadata:
        summary.update(metadata)
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_shape_weights(args.shape_weights)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, image_source = resolve_image_ids(args, dataset)
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    net = Net("MSHNetOHEM", mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = torch_load_checkpoint(args.ohem_checkpoint, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    net.eval()

    component_rows = []
    image_rows = []
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            export = net.export_logits_features(img)
            logit = export["logit"][:, :, :h, :w]
            prob = foreground_probability(logit)[0, 0].detach().cpu().numpy().astype(np.float32)
            pred = prob > args.threshold
            bg = fixed_context_background(img, args.background_kernel)[:, :, :h, :w]
            img_crop = img[:, :1, :h, :w]
            residual = torch.abs(img_crop - bg)
            residual_norm = residual / (residual.mean(dim=(-2, -1), keepdim=True) + 1e-6)
            residual_np = residual_norm[0, 0].detach().cpu().numpy().astype(np.float32)

            rows, counts = audit_image_shape_components(
                image_name=name,
                gt_mask=gt,
                pred_mask=pred,
                residual=residual_np,
                args=args,
                weights=weights,
            )
            component_rows.extend(rows)
            image_rows.append(counts)
            if (idx + 1) % 100 == 0:
                print(f"Gate-BCV-D audited [{idx + 1}/{len(loader)}]", flush=True)

    metadata = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "split": args.split,
        "image_source": image_source,
        "num_images": len(image_rows),
        "ohem_checkpoint": os.path.abspath(args.ohem_checkpoint),
        "bcv_checkpoint_or_init": args.bcv_checkpoint_or_init,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "residual_source": "fixed_context_background_residual_norm",
        "background_kernel": args.background_kernel,
        "target_near_radius": args.target_near_radius,
        "match_distance": args.match_distance,
        "shape_weights": weights.__dict__,
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "per_component": str(output_dir / "per_component.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary = build_gate_d_summary(component_rows, image_rows, args, metadata=metadata)
    write_csv(
        output_dir / "per_image.csv",
        image_rows,
        [
            "image_name",
            "target_component_count",
            "matched_target_prediction_component_count",
            "near_fp_component_count",
            "far_fp_component_count",
            "target_leakage_pixels",
        ],
    )
    write_csv(
        output_dir / "per_component.csv",
        component_rows,
        [
            "image_name",
            "component_type",
            "component_id",
            "centroid_y",
            "centroid_x",
            "area",
            "compactness",
            "bbox_fill_ratio",
            "anisotropy",
            "center_surround",
            "radial_symmetry",
            "dog_peakness",
            "shape_score",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
