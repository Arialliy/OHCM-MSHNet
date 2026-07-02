#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
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
from utils import get_img_norm_cfg

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - pure numpy fallback is covered by tests.
    ndi = None


@dataclass
class Component:
    label: int
    coords: np.ndarray

    @property
    def area(self) -> int:
        return int(self.coords.shape[0])

    @property
    def centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="BCV Gate-C OHEM false-positive residual audit.")
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
    parser.add_argument("--min_far_fp_components", type=int, default=20)
    parser.add_argument("--min_target_vs_far_fp_auroc", type=float, default=0.65)
    parser.add_argument("--hard_stop_min_auroc", type=float, default=0.60)
    parser.add_argument("--min_suppressible_far_fp_rate_99", type=float, default=0.20)
    parser.add_argument("--min_suppressible_far_fp_rate_995", type=float, default=0.10)
    parser.add_argument("--hard_stop_min_suppressible_far_fp_rate_99", type=float, default=0.10)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def finite_ge(value, threshold):
    return bool(np.isfinite(value) and value >= threshold)


def quantile_or_zero(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def median_or_zero(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0:
        return mask
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    return F_max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))[0, 0].numpy() > 0


def F_max_pool2d(tensor, kernel_size, stride, padding):
    return torch.nn.functional.max_pool2d(tensor, kernel_size=kernel_size, stride=stride, padding=padding)


def connected_components(mask: np.ndarray) -> list[Component]:
    mask = mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError("connected_components expects a 2D mask")
    if not mask.any():
        return []
    if ndi is not None:
        labeled, count = ndi.label(mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
        regions = []
        for label in range(1, int(count) + 1):
            coords = np.argwhere(labeled == label)
            if coords.size:
                regions.append(Component(label=label, coords=coords.astype(np.int64)))
        return regions

    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    regions = []
    label = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        label += 1
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        coords = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        regions.append(Component(label=label, coords=np.asarray(coords, dtype=np.int64)))
    return regions


def component_values(values: np.ndarray, component: Component) -> np.ndarray:
    return values[component.coords[:, 0], component.coords[:, 1]].astype(np.float64)


def component_intersects(component: Component, mask: np.ndarray) -> bool:
    return bool(mask[component.coords[:, 0], component.coords[:, 1]].any())


def component_mask_sum(component: Component, mask: np.ndarray) -> int:
    return int(mask[component.coords[:, 0], component.coords[:, 1]].sum())


def component_row(image_name: str, component_type: str, component_id: int, component: Component, residual: np.ndarray) -> dict:
    values = component_values(residual, component)
    centroid = component.centroid
    return {
        "image_name": image_name,
        "component_type": component_type,
        "component_id": int(component_id),
        "area": component.area,
        "residual_mean": float(values.mean()) if values.size else 0.0,
        "residual_median": float(np.median(values)) if values.size else 0.0,
        "residual_q10": quantile_or_zero(values, 0.10),
        "residual_q90": quantile_or_zero(values, 0.90),
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
    }


def prediction_matches_target(pred_component: Component, gt_components: list[Component], gt_mask: np.ndarray, match_distance: float) -> bool:
    if component_intersects(pred_component, gt_mask):
        return True
    if not gt_components:
        return False
    pred_centroid = pred_component.centroid
    for gt_component in gt_components:
        if np.linalg.norm(pred_centroid - gt_component.centroid) <= match_distance:
            return True
    return False


def audit_image_components(
    image_name: str,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    residual: np.ndarray,
    target_near_radius: int = 7,
    match_distance: float = 3.0,
) -> tuple[list[dict], dict]:
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    residual = residual.astype(np.float64)
    if gt_mask.shape != pred_mask.shape or gt_mask.shape != residual.shape:
        raise ValueError("gt_mask, pred_mask, and residual must have the same shape")

    gt_components = connected_components(gt_mask)
    pred_components = connected_components(pred_mask)
    target_near = binary_dilate(gt_mask, target_near_radius)
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
        rows.append(component_row(image_name, "gt_target", component_id, component, residual))

    for component_id, component in enumerate(pred_components, start=1):
        if prediction_matches_target(component, gt_components, gt_mask, match_distance):
            counts["matched_target_prediction_component_count"] += 1
            rows.append(component_row(image_name, "matched_target_prediction", component_id, component, residual))
            continue

        leakage = component_mask_sum(component, gt_mask)
        counts["target_leakage_pixels"] += leakage
        if component_intersects(component, target_near):
            counts["near_fp_component_count"] += 1
            rows.append(component_row(image_name, "near_fp", component_id, component, residual))
        else:
            counts["far_fp_component_count"] += 1
            rows.append(component_row(image_name, "far_fp", component_id, component, residual))

    return rows, counts


def component_residuals(rows: list[dict], component_type: str) -> list[float]:
    return [
        float(row["residual_mean"])
        for row in rows
        if row.get("component_type") == component_type and np.isfinite(float(row["residual_mean"]))
    ]


def suppressible_rate_at_target_recall(target_values: list[float], fp_values: list[float], recall: float) -> tuple[float, float]:
    if not target_values or not fp_values:
        return 0.0, 0.0
    threshold = quantile_or_zero(target_values, max(0.0, 1.0 - float(recall)))
    fp = np.asarray(fp_values, dtype=np.float64)
    return float((fp < threshold).mean()), float(threshold)


def build_summary(component_rows: list[dict], image_rows: list[dict], args, metadata: dict | None = None) -> dict:
    target = component_residuals(component_rows, "gt_target")
    matched_target_pred = component_residuals(component_rows, "matched_target_prediction")
    far_fp = component_residuals(component_rows, "far_fp")
    near_fp = component_residuals(component_rows, "near_fp")
    suppress99, theta99 = suppressible_rate_at_target_recall(target, far_fp, 0.99)
    suppress995, theta995 = suppressible_rate_at_target_recall(target, far_fp, 0.995)
    auroc = binary_auc(target, far_fp)
    target_q10 = quantile_or_zero(target, 0.10)
    far_fp_median = median_or_zero(far_fp)
    target_leakage = int(sum(int(row["target_leakage_pixels"]) for row in image_rows))
    checks = {
        "far_fp_component_count": len(far_fp) >= args.min_far_fp_components,
        "target_vs_far_fp_residual_auroc": finite_ge(auroc, args.min_target_vs_far_fp_auroc),
        "suppressible_far_fp_rate_at_target_recall_99": finite_ge(suppress99, args.min_suppressible_far_fp_rate_99),
        "suppressible_far_fp_rate_at_target_recall_995": finite_ge(suppress995, args.min_suppressible_far_fp_rate_995),
        "target_leakage": target_leakage == 0,
    }
    diagnostics = {
        "target_q10_gt_far_fp_median": bool(target_q10 > far_fp_median),
    }
    stop_conditions = {
        "far_fp_component_count_too_low": len(far_fp) < args.min_far_fp_components,
        "auroc_below_hard_stop": bool(auroc < args.hard_stop_min_auroc),
        "suppressible_far_fp_rate_99_below_hard_stop": bool(
            suppress99 < args.hard_stop_min_suppressible_far_fp_rate_99
        ),
        "distribution_overlap_target_q10_le_far_fp_median": bool(target_q10 <= far_fp_median),
    }
    summary = {
        "gate": "BCV_Gate_C_ohem_fp_residual_audit",
        "target_component_count": len(target),
        "matched_target_prediction_component_count": len(matched_target_pred),
        "far_fp_component_count": len(far_fp),
        "near_fp_component_count": len(near_fp),
        "target_residual_median": median_or_zero(target),
        "target_residual_q10": target_q10,
        "far_fp_residual_median": far_fp_median,
        "far_fp_residual_q90": quantile_or_zero(far_fp, 0.90),
        "target_vs_far_fp_residual_auroc": auroc,
        "suppressible_far_fp_rate_at_target_recall_99": suppress99,
        "suppressible_far_fp_rate_at_target_recall_995": suppress995,
        "target_protection_threshold": theta99,
        "target_protection_threshold_995": theta995,
        "target_leakage_pixels_total": target_leakage,
        "checks": checks,
        "diagnostics": diagnostics,
        "stop_conditions": stop_conditions,
        "thresholds": {
            "min_far_fp_components": args.min_far_fp_components,
            "target_vs_far_fp_residual_auroc_min": args.min_target_vs_far_fp_auroc,
            "hard_stop_min_auroc": args.hard_stop_min_auroc,
            "suppressible_far_fp_rate_at_target_recall_99_min": args.min_suppressible_far_fp_rate_99,
            "suppressible_far_fp_rate_at_target_recall_995_min": args.min_suppressible_far_fp_rate_995,
            "hard_stop_min_suppressible_far_fp_rate_99": args.hard_stop_min_suppressible_far_fp_rate_99,
            "target_leakage_pixels_total": 0,
        },
        "gate_pass": bool(all(checks.values())),
    }
    summary["overall_decision"] = (
        "PROCEED_TO_DETERMINISTIC_CALIBRATION" if summary["gate_pass"] else "STOP_BCV"
    )
    if metadata:
        summary.update(metadata)
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_ids(args, dataset):
    if args.image_list:
        path = Path(args.image_list)
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


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

            rows, counts = audit_image_components(
                image_name=name,
                gt_mask=gt,
                pred_mask=pred,
                residual=residual_np,
                target_near_radius=args.target_near_radius,
                match_distance=args.match_distance,
            )
            component_rows.extend(rows)
            image_rows.append(counts)
            if (idx + 1) % 100 == 0:
                print(f"Gate-BCV-C audited [{idx + 1}/{len(loader)}]", flush=True)

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
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "per_component": str(output_dir / "per_component.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary = build_summary(component_rows, image_rows, args, metadata=metadata)
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
            "area",
            "residual_mean",
            "residual_median",
            "residual_q10",
            "residual_q90",
            "centroid_y",
            "centroid_x",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
