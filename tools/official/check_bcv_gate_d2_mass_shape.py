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
from tools.official.check_bcv_gate_b import fixed_context_background
from tools.official.check_bcv_gate_c_fp_residual import (
    binary_dilate,
    connected_components,
    finite_ge,
    prediction_matches_target,
    quantile_or_zero,
    resolve_image_ids,
    size_to_int,
)
from utils import get_img_norm_cfg
from utils.residual_shape_features import ResidualShapeWeights, parse_shape_weights, residual_shape_features


def parse_args():
    parser = argparse.ArgumentParser(description="BCV Gate-D2 mass-weighted residual/shape audit.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test", "full"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--bcv_checkpoint_or_init", default="fixed_context_background")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gate_d_summary", required=True)
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
    parser.add_argument("--min_pixel_mass_rate_component_99", type=float, default=0.15)
    parser.add_argument("--min_confidence_mass_rate_component_99", type=float, default=0.15)
    parser.add_argument("--min_pixel_mass_rate_pixel_995", type=float, default=0.10)
    parser.add_argument("--min_confidence_mass_rate_pixel_995", type=float, default=0.10)
    parser.add_argument("--hard_stop_min_mass_rate", type=float, default=0.10)
    return parser.parse_args()


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    cutoff = float(q) * float(weights.sum())
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def load_shape_weights(args, gate_d_summary: dict) -> ResidualShapeWeights:
    if args.shape_weights:
        return parse_shape_weights(args.shape_weights)
    summary_weights = gate_d_summary.get("shape_weights")
    if isinstance(summary_weights, dict):
        return ResidualShapeWeights(
            compactness=float(summary_weights.get("compactness", 1.0)),
            fill_ratio=float(summary_weights.get("fill_ratio", 0.5)),
            anisotropy=float(summary_weights.get("anisotropy", 0.15)),
            center_surround=float(summary_weights.get("center_surround", 0.5)),
            radial_symmetry=float(summary_weights.get("radial_symmetry", 0.5)),
            dog_peakness=float(summary_weights.get("dog_peakness", 0.5)),
        )
    return ResidualShapeWeights()


def finite_rows(rows: list[dict], component_type: str) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("component_type") == component_type and np.isfinite(float(row.get("shape_score", np.nan)))
    ]


def component_mass_row(
    image_name: str,
    component_type: str,
    component_id: int,
    component,
    residual: np.ndarray,
    prob: np.ndarray,
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
    yy, xx = component.coords[:, 0], component.coords[:, 1]
    prob_values = prob[yy, xx].astype(np.float64)
    centroid = component.centroid
    return {
        "image_name": image_name,
        "component_type": component_type,
        "component_id": int(component_id),
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
        "area": int(component.area),
        "confidence_mass": float(prob_values.sum()),
        "peak_prob": float(prob_values.max()) if prob_values.size else 0.0,
        "prob_mean": float(prob_values.mean()) if prob_values.size else 0.0,
        **features,
    }


def audit_image_mass_components(
    image_name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    prob: np.ndarray,
    residual: np.ndarray,
    args,
    weights,
) -> tuple[list[dict], dict]:
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    if gt_mask.shape != pred_mask.shape or gt_mask.shape != residual.shape or gt_mask.shape != prob.shape:
        raise ValueError("gt_mask, pred_mask, prob, and residual must have the same shape")

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
        rows.append(component_mass_row(image_name, "gt_target", component_id, component, residual, prob, args, weights))

    for component_id, component in enumerate(pred_components, start=1):
        if prediction_matches_target(component, gt_components, gt_mask, args.match_distance):
            counts["matched_target_prediction_component_count"] += 1
            rows.append(
                component_mass_row(
                    image_name,
                    "matched_target_prediction",
                    component_id,
                    component,
                    residual,
                    prob,
                    args,
                    weights,
                )
            )
            continue

        leakage = int(gt_mask[component.coords[:, 0], component.coords[:, 1]].sum())
        counts["target_leakage_pixels"] += leakage
        if bool(target_near[component.coords[:, 0], component.coords[:, 1]].any()):
            counts["near_fp_component_count"] += 1
            rows.append(component_mass_row(image_name, "near_fp", component_id, component, residual, prob, args, weights))
        else:
            counts["far_fp_component_count"] += 1
            rows.append(component_mass_row(image_name, "far_fp", component_id, component, residual, prob, args, weights))

    return rows, counts


def mass_rates(far_rows: list[dict], threshold: float) -> dict:
    suppressible = [row for row in far_rows if float(row["shape_score"]) < threshold]
    total_area = sum(float(row["area"]) for row in far_rows)
    total_confidence = sum(float(row["confidence_mass"]) for row in far_rows)
    suppressed_area = sum(float(row["area"]) for row in suppressible)
    suppressed_confidence = sum(float(row["confidence_mass"]) for row in suppressible)
    return {
        "component_rate": safe_div(len(suppressible), len(far_rows)),
        "pixel_mass_rate": safe_div(suppressed_area, total_area),
        "confidence_mass_rate": safe_div(suppressed_confidence, total_confidence),
        "suppressed_count": len(suppressible),
        "suppressed_area": suppressed_area,
        "suppressed_confidence_mass": suppressed_confidence,
        "top_suppressible_fp_area_mean": float(np.mean([row["area"] for row in suppressible])) if suppressible else 0.0,
        "top_suppressible_fp_peak_prob_mean": float(np.mean([row["peak_prob"] for row in suppressible])) if suppressible else 0.0,
    }


def build_gate_d2_summary(component_rows: list[dict], image_rows: list[dict], args, metadata: dict | None = None) -> dict:
    target_rows = finite_rows(component_rows, "gt_target")
    far_rows = finite_rows(component_rows, "far_fp")
    target_scores = [float(row["shape_score"]) for row in target_rows]
    target_areas = [float(row["area"]) for row in target_rows]

    component_threshold = quantile_or_zero(target_scores, 0.01)
    pixel_threshold = weighted_quantile(target_scores, target_areas, 0.005)
    component_rates = mass_rates(far_rows, component_threshold)
    pixel_rates = mass_rates(far_rows, pixel_threshold)

    far_area_mean = float(np.mean([row["area"] for row in far_rows])) if far_rows else 0.0
    far_peak_prob_mean = float(np.mean([row["peak_prob"] for row in far_rows])) if far_rows else 0.0
    target_leakage = int(sum(int(row["target_leakage_pixels"]) for row in image_rows))
    suppressed_low_conf_small_noise = bool(
        component_rates["suppressed_count"] > 0
        and component_rates["top_suppressible_fp_area_mean"] < far_area_mean
        and component_rates["top_suppressible_fp_peak_prob_mean"] < far_peak_prob_mean
    )

    checks = {
        "far_fp_component_count": len(far_rows) >= args.min_far_fp_components,
        "far_fp_pixel_mass_at_component_recall_99": finite_ge(
            component_rates["pixel_mass_rate"], args.min_pixel_mass_rate_component_99
        ),
        "far_fp_confidence_mass_at_component_recall_99": finite_ge(
            component_rates["confidence_mass_rate"], args.min_confidence_mass_rate_component_99
        ),
        "far_fp_pixel_mass_at_target_pixel_recall_995": finite_ge(
            pixel_rates["pixel_mass_rate"], args.min_pixel_mass_rate_pixel_995
        ),
        "far_fp_confidence_mass_at_target_pixel_recall_995": finite_ge(
            pixel_rates["confidence_mass_rate"], args.min_confidence_mass_rate_pixel_995
        ),
        "target_leakage": target_leakage == 0,
    }
    stop_conditions = {
        "far_fp_component_count_too_low": len(far_rows) < args.min_far_fp_components,
        "pixel_mass_rate_component_99_below_hard_stop": component_rates["pixel_mass_rate"] < args.hard_stop_min_mass_rate,
        "confidence_mass_rate_component_99_below_hard_stop": component_rates["confidence_mass_rate"] < args.hard_stop_min_mass_rate,
        "pixel_mass_rate_pixel_995_below_hard_stop": pixel_rates["pixel_mass_rate"] < args.hard_stop_min_mass_rate,
        "confidence_mass_rate_pixel_995_below_hard_stop": pixel_rates["confidence_mass_rate"] < args.hard_stop_min_mass_rate,
        "suppressed_fp_low_conf_small_noise": suppressed_low_conf_small_noise,
    }
    summary = {
        "gate": "BCV_Gate_D2_mass_weighted_residual_shape_audit",
        "target_component_count": len(target_rows),
        "far_fp_component_count": len(far_rows),
        "near_fp_component_count": len(finite_rows(component_rows, "near_fp")),
        "matched_target_prediction_component_count": len(finite_rows(component_rows, "matched_target_prediction")),
        "target_component_recall_threshold": component_threshold,
        "target_pixel_recall_threshold": pixel_threshold,
        "suppressible_far_fp_component_rate_at_target_component_recall_99": component_rates["component_rate"],
        "suppressible_far_fp_pixel_mass_rate_at_target_component_recall_99": component_rates["pixel_mass_rate"],
        "suppressible_far_fp_confidence_mass_rate_at_target_component_recall_99": component_rates["confidence_mass_rate"],
        "suppressible_far_fp_pixel_mass_rate_at_target_pixel_recall_995": pixel_rates["pixel_mass_rate"],
        "suppressible_far_fp_confidence_mass_rate_at_target_pixel_recall_995": pixel_rates["confidence_mass_rate"],
        "suppressible_far_fp_component_count_at_target_component_recall_99": component_rates["suppressed_count"],
        "suppressible_far_fp_pixel_mass_at_target_component_recall_99": component_rates["suppressed_area"],
        "suppressible_far_fp_confidence_mass_at_target_component_recall_99": component_rates["suppressed_confidence_mass"],
        "top_suppressible_fp_area_mean": component_rates["top_suppressible_fp_area_mean"],
        "top_suppressible_fp_peak_prob_mean": component_rates["top_suppressible_fp_peak_prob_mean"],
        "far_fp_area_mean": far_area_mean,
        "far_fp_peak_prob_mean": far_peak_prob_mean,
        "target_leakage_pixels_total": target_leakage,
        "checks": checks,
        "stop_conditions": stop_conditions,
        "thresholds": {
            "min_far_fp_components": args.min_far_fp_components,
            "pixel_mass_rate_component_99_min": args.min_pixel_mass_rate_component_99,
            "confidence_mass_rate_component_99_min": args.min_confidence_mass_rate_component_99,
            "pixel_mass_rate_pixel_995_min": args.min_pixel_mass_rate_pixel_995,
            "confidence_mass_rate_pixel_995_min": args.min_confidence_mass_rate_pixel_995,
            "hard_stop_min_mass_rate": args.hard_stop_min_mass_rate,
            "target_leakage_pixels_total": 0,
        },
        "mass_gate_pass": bool(all(checks.values())),
    }
    summary["overall_decision"] = "PROCEED_TO_DETERMINISTIC_FORMULA_CALIBRATION" if summary["mass_gate_pass"] else "STOP_BCV"
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
    gate_d_summary_path = Path(args.gate_d_summary)
    gate_d_summary = json.loads(gate_d_summary_path.read_text(encoding="utf-8"))
    weights = load_shape_weights(args, gate_d_summary)
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
            rows, counts = audit_image_mass_components(
                image_name=name,
                gt_mask=gt,
                pred_mask=pred,
                prob=prob,
                residual=residual_np,
                args=args,
                weights=weights,
            )
            component_rows.extend(rows)
            image_rows.append(counts)
            if (idx + 1) % 100 == 0:
                print(f"Gate-BCV-D2 audited [{idx + 1}/{len(loader)}]", flush=True)

    metadata = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "split": args.split,
        "image_source": image_source,
        "num_images": len(image_rows),
        "ohem_checkpoint": os.path.abspath(args.ohem_checkpoint),
        "bcv_checkpoint_or_init": args.bcv_checkpoint_or_init,
        "gate_d_summary": str(gate_d_summary_path),
        "gate_d_overall_decision": gate_d_summary.get("overall_decision"),
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
    summary = build_gate_d2_summary(component_rows, image_rows, args, metadata=metadata)
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
            "confidence_mass",
            "peak_prob",
            "prob_mean",
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
    if not summary["mass_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
