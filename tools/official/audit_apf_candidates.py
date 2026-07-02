#!/usr/bin/env python3
"""Audit APF-OHEM train-only candidates before any APF training."""
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
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit APF-OHEM candidate masks.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--anchor_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_dilation_radius", type=int, default=5)
    parser.add_argument("--tau_low", type=float, default=0.25)
    parser.add_argument("--tau_high", type=float, default=0.60)
    parser.add_argument("--hard_top_q", type=float, default=0.01)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))[0, 0].numpy() > 0


def component_count(mask: np.ndarray) -> int:
    return int(measure.label(mask.astype(np.uint8), connectivity=2).max())


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def select_candidate_mask(
    prob: np.ndarray,
    gt_mask: np.ndarray,
    target_dilation_radius: int,
    tau_low: float,
    tau_high: float,
    hard_top_q: float,
) -> tuple[np.ndarray, dict]:
    gt = gt_mask.astype(bool)
    target_protect = binary_dilate(gt, target_dilation_radius)
    far_bg = ~target_protect
    candidate = far_bg & (prob >= tau_low) & (prob <= tau_high)

    far_indices = np.flatnonzero(far_bg.reshape(-1))
    budget = max(1, int(np.ceil(len(far_indices) * hard_top_q))) if len(far_indices) else 0
    if budget > 0:
        far_scores = prob.reshape(-1)[far_indices]
        budget = min(budget, far_scores.size)
        top_local = np.argpartition(far_scores, -budget)[-budget:]
        hard_flat = np.zeros(prob.size, dtype=bool)
        hard_flat[far_indices[top_local]] = True
        candidate |= hard_flat.reshape(prob.shape)

    candidate &= far_bg
    stats = {
        "far_bg_pixels": int(far_bg.sum()),
        "ohem_budget_pixels": int(budget),
        "target_protect_pixels": int(target_protect.sum()),
    }
    return candidate.astype(bool), stats


def fp_component_coverage(pred_mask: np.ndarray, gt_mask: np.ndarray, candidate: np.ndarray) -> tuple[int, int]:
    covered = 0
    total = 0
    gt = gt_mask.astype(bool)
    for region in connected_regions(pred_mask):
        comp = np.zeros_like(pred_mask, dtype=bool)
        comp[region.coords[:, 0], region.coords[:, 1]] = True
        if np.logical_and(comp, gt).any():
            continue
        total += 1
        if np.logical_and(comp, candidate).any():
            covered += 1
    return covered, total


def component_rows_for_image(
    image_id: str,
    prob: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    candidate: np.ndarray,
    target_dilation_radius: int,
    tau_low: float,
) -> list[dict]:
    rows = []
    gt = gt_mask.astype(bool)
    target_protect = binary_dilate(gt, target_dilation_radius)
    boundary = target_protect & (~gt)
    dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)
    seen_candidate = np.zeros_like(candidate, dtype=bool)
    component_id = 1

    for region in connected_regions(pred_mask):
        comp = np.zeros_like(pred_mask, dtype=bool)
        comp[region.coords[:, 0], region.coords[:, 1]] = True
        overlaps_gt = bool((comp & gt).any())
        is_boundary_excess = bool((comp & boundary).any()) and not overlaps_gt
        is_detached_far_fp = bool((comp & (~target_protect)).any()) and not overlaps_gt
        overlap_candidate = bool((comp & candidate).any())
        rows.append(
            {
                "image_id": image_id,
                "component_id": component_id,
                "area": int(comp.sum()),
                "max_prob": float(prob[comp].max()),
                "mean_prob": float(prob[comp].mean()),
                "mean_anchor_prob": float(prob[comp].mean()),
                "mean_current_prob": float(prob[comp].mean()),
                "distance_to_nearest_gt": float(dist_to_gt[comp].min()) if comp.any() else float("nan"),
                "is_boundary_excess": int(is_boundary_excess),
                "is_detached_far_fp": int(is_detached_far_fp),
                "is_flat_bg": int(float(prob[comp].mean()) < tau_low),
                "overlap_ohem_fp": int(not overlaps_gt),
                "overlap_candidate": int(overlap_candidate),
                "selected_by_apf": int(overlap_candidate),
            }
        )
        seen_candidate |= comp & candidate
        component_id += 1

    candidate_only = candidate & (~seen_candidate)
    for region in connected_regions(candidate_only):
        comp = np.zeros_like(candidate, dtype=bool)
        comp[region.coords[:, 0], region.coords[:, 1]] = True
        rows.append(
            {
                "image_id": image_id,
                "component_id": component_id,
                "area": int(comp.sum()),
                "max_prob": float(prob[comp].max()),
                "mean_prob": float(prob[comp].mean()),
                "mean_anchor_prob": float(prob[comp].mean()),
                "mean_current_prob": float(prob[comp].mean()),
                "distance_to_nearest_gt": float(dist_to_gt[comp].min()) if comp.any() else float("nan"),
                "is_boundary_excess": int(bool((comp & boundary).any())),
                "is_detached_far_fp": int(bool((comp & (~target_protect)).any())),
                "is_flat_bg": int(float(prob[comp].mean()) < tau_low),
                "overlap_ohem_fp": 0,
                "overlap_candidate": 1,
                "selected_by_apf": 1,
            }
        )
        component_id += 1
    return rows


def split_ids(args, dataset) -> tuple[list[str], str]:
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)
    output_dir = Path(args.output_dir)
    candidate_dir = Path(args.candidate_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, split_source = split_ids(args, dataset)
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    rows = []
    component_rows = []
    coverage_values = []
    fail_reasons = []
    waterfall_totals = {
        "num_pixels_total": 0,
        "num_valid_background_pixels": 0,
        "num_far_background_pixels": 0,
        "num_anchor_high_pixels": 0,
        "num_anchor_disagreement_pixels": 0,
        "num_nonflat_pixels": 0,
        "num_ohem_fp_component_pixels": 0,
        "num_candidate_pixels": 0,
        "num_selected_pixels": 0,
    }
    candidate_type = {
        "detached_far_fp_pixels": 0,
        "boundary_excess_pixels": 0,
        "flat_bg_pixels": 0,
        "near_gt_ambiguous_pixels": 0,
    }
    for idx, (_img, gt_mask, size, image_name) in enumerate(loader):
        h, w = size_to_int(size[0]), size_to_int(size[1])
        name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
        gt = gt_mask[0, 0, :h, :w].numpy() > 0
        anchor_path = anchor_dir / f"{name}.npz"
        if not anchor_path.exists():
            fail_reasons.append(f"missing_anchor:{name}")
            continue
        anchor = np.load(anchor_path, allow_pickle=False)
        prob = anchor["prob_ohem"].astype(np.float32)
        if prob.shape != gt.shape:
            fail_reasons.append(f"shape_mismatch:{name}")
            continue
        if not np.isfinite(prob).all():
            fail_reasons.append(f"nonfinite_anchor:{name}")
            continue

        candidate, candidate_stats = select_candidate_mask(
            prob=prob,
            gt_mask=gt,
            target_dilation_radius=args.target_dilation_radius,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            hard_top_q=args.hard_top_q,
        )
        target_protect = binary_dilate(gt, args.target_dilation_radius)
        boundary = target_protect & (~gt)
        far_bg = ~target_protect
        pred_ohem = prob > args.threshold
        covered_fp, total_fp = fp_component_coverage(pred_ohem, gt, candidate)
        if total_fp > 0:
            coverage_values.append(safe_div(covered_fp, total_fp))

        candidate_pixels = int(candidate.sum())
        budget = int(candidate_stats["ohem_budget_pixels"])
        ratio = safe_div(candidate_pixels, budget)
        target_leakage = int((candidate & gt).sum())
        boundary_overlap = int((candidate & boundary).sum())
        flat_bg_pixels = int((candidate & (prob < args.tau_low)).sum())
        flat_bg_ratio = safe_div(flat_bg_pixels, candidate_pixels)
        candidate_probs = prob[candidate]
        ohem_fp_mask = pred_ohem & far_bg
        nonflat_mask = far_bg & (prob >= args.tau_low)
        fail_reason = []
        if candidate_pixels == 0:
            fail_reason.append("candidate_empty")
        if ratio < 1.0:
            fail_reason.append("under_budget")
        if flat_bg_ratio > 0.35:
            fail_reason.append("flat_bg_ratio_high")
        if safe_div(covered_fp, total_fp) < 0.40 and total_fp > 0:
            fail_reason.append("low_ohem_fp_component_coverage")

        waterfall_totals["num_pixels_total"] += int(prob.size)
        waterfall_totals["num_valid_background_pixels"] += int((~gt).sum())
        waterfall_totals["num_far_background_pixels"] += int(far_bg.sum())
        waterfall_totals["num_anchor_high_pixels"] += int((far_bg & (prob >= args.tau_low)).sum())
        waterfall_totals["num_anchor_disagreement_pixels"] += int(ohem_fp_mask.sum())
        waterfall_totals["num_nonflat_pixels"] += int(nonflat_mask.sum())
        waterfall_totals["num_ohem_fp_component_pixels"] += int(ohem_fp_mask.sum())
        waterfall_totals["num_candidate_pixels"] += candidate_pixels
        waterfall_totals["num_selected_pixels"] += budget

        candidate_type["detached_far_fp_pixels"] += int((candidate & ohem_fp_mask).sum())
        candidate_type["boundary_excess_pixels"] += int((candidate & boundary).sum())
        candidate_type["flat_bg_pixels"] += flat_bg_pixels
        candidate_type["near_gt_ambiguous_pixels"] += int((candidate & boundary & (~gt)).sum())

        row = {
            "image_id": name,
            "budget_pixels": budget,
            "candidate_pixels": candidate_pixels,
            "candidate_to_ohem_budget_ratio": ratio,
            "candidate_components": component_count(candidate),
            "candidate_far_fp_component_overlap": safe_div(covered_fp, total_fp),
            "candidate_target_leakage_pixels": target_leakage,
            "candidate_boundary_overlap_pixels": boundary_overlap,
            "candidate_flat_bg_ratio": flat_bg_ratio,
            "candidate_prob_mean": float(candidate_probs.mean()) if candidate_pixels else 0.0,
            "candidate_prob_p90": float(np.percentile(candidate_probs, 90)) if candidate_pixels else 0.0,
            "candidate_prob_p99": float(np.percentile(candidate_probs, 99)) if candidate_pixels else 0.0,
            "ohem_fp_components_covered": int(covered_fp),
            "ohem_fp_components_total": int(total_fp),
            "new_candidate_components_not_ohem_fp": max(0, component_count(candidate) - int(covered_fp)),
            "fail_reason": ";".join(fail_reason),
            "num_pixels_total": int(prob.size),
            "num_valid_background_pixels": int((~gt).sum()),
            "num_far_background_pixels": int(far_bg.sum()),
            "num_anchor_high_pixels": int((far_bg & (prob >= args.tau_low)).sum()),
            "num_anchor_disagreement_pixels": int(ohem_fp_mask.sum()),
            "num_nonflat_pixels": int(nonflat_mask.sum()),
            "num_ohem_fp_component_pixels": int(ohem_fp_mask.sum()),
            "num_selected_pixels": budget,
        }
        rows.append(row)
        component_rows.extend(
            component_rows_for_image(
                image_id=name,
                prob=prob,
                gt_mask=gt,
                pred_mask=pred_ohem,
                candidate=candidate,
                target_dilation_radius=args.target_dilation_radius,
                tau_low=args.tau_low,
            )
        )

        metadata = {
            "dataset_name": args.dataset_name,
            "split_name": args.split,
            "image_id": name,
            "anchor_dir": str(anchor_dir),
            "threshold": args.threshold,
            "target_dilation_radius": args.target_dilation_radius,
            "tau_low": args.tau_low,
            "tau_high": args.tau_high,
            "hard_top_q": args.hard_top_q,
        }
        np.savez_compressed(
            candidate_dir / f"{name}.npz",
            candidate_mask=candidate,
            target_protect_mask=target_protect,
            easy_bg_mask=(~target_protect) & (~candidate),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        if (idx + 1) % 100 == 0:
            print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    fields = [
        "image_id",
        "budget_pixels",
        "candidate_pixels",
        "candidate_to_ohem_budget_ratio",
        "candidate_components",
        "candidate_far_fp_component_overlap",
        "candidate_target_leakage_pixels",
        "candidate_boundary_overlap_pixels",
        "candidate_flat_bg_ratio",
        "candidate_prob_mean",
        "candidate_prob_p90",
        "candidate_prob_p99",
        "ohem_fp_components_covered",
        "ohem_fp_components_total",
        "new_candidate_components_not_ohem_fp",
        "fail_reason",
        "num_pixels_total",
        "num_valid_background_pixels",
        "num_far_background_pixels",
        "num_anchor_high_pixels",
        "num_anchor_disagreement_pixels",
        "num_nonflat_pixels",
        "num_ohem_fp_component_pixels",
        "num_selected_pixels",
    ]
    write_csv(output_dir / "per_image.csv", rows, fields)
    component_fields = [
        "image_id",
        "component_id",
        "area",
        "max_prob",
        "mean_prob",
        "mean_anchor_prob",
        "mean_current_prob",
        "distance_to_nearest_gt",
        "is_boundary_excess",
        "is_detached_far_fp",
        "is_flat_bg",
        "overlap_ohem_fp",
        "overlap_candidate",
        "selected_by_apf",
    ]
    write_csv(output_dir / "components.csv", component_rows, component_fields)

    num_images = len(rows)
    empty = sum(1 for row in rows if int(row["candidate_pixels"]) == 0)
    under_budget = sum(1 for row in rows if float(row["candidate_to_ohem_budget_ratio"]) < 1.0)
    target_leakage_total = sum(int(row["candidate_target_leakage_pixels"]) for row in rows)
    boundary_total = sum(int(row["candidate_boundary_overlap_pixels"]) for row in rows)
    ratios = [float(row["candidate_to_ohem_budget_ratio"]) for row in rows]
    positive_ratio_fraction = mean([1.0 if r > 0.0 else 0.0 for r in ratios])
    flat_mean = mean([float(row["candidate_flat_bg_ratio"]) for row in rows])
    coverage_mean = mean(coverage_values)
    candidate_component_mean = mean([float(row["candidate_components"]) for row in rows])

    gate_failures = list(fail_reasons)
    if target_leakage_total != 0:
        gate_failures.append("target_leakage_pixels_total_nonzero")
    if safe_div(empty, num_images) > 0.10:
        gate_failures.append("candidate_empty_ratio_too_high")
    if mean(ratios) < 1.5:
        gate_failures.append("candidate_to_budget_ratio_mean_too_low")
    if positive_ratio_fraction < 0.90:
        gate_failures.append("candidate_positive_ratio_fraction_too_low")
    if flat_mean > 0.35:
        gate_failures.append("flat_bg_ratio_mean_too_high")
    if coverage_mean < 0.40:
        gate_failures.append("ohem_fp_component_coverage_mean_too_low")

    summary = {
        "gate_pass": len(gate_failures) == 0,
        "fail_reasons": gate_failures,
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "anchor_dir": str(anchor_dir),
        "candidate_output_dir": str(candidate_dir),
        "num_images": num_images,
        "num_images_with_candidate_empty": empty,
        "num_images_candidate_under_budget": under_budget,
        "candidate_to_budget_ratio_mean": mean(ratios),
        "candidate_to_budget_ratio_min": float(min(ratios)) if ratios else 0.0,
        "candidate_to_budget_ratio_positive_fraction": positive_ratio_fraction,
        "target_leakage_pixels_total": int(target_leakage_total),
        "boundary_overlap_pixels_total": int(boundary_total),
        "flat_bg_ratio_mean": flat_mean,
        "ohem_fp_component_coverage_mean": coverage_mean,
        "candidate_component_count_mean": candidate_component_mean,
        "waterfall_totals": waterfall_totals,
        "waterfall_ratios": {
            "far_bg_ratio": safe_div(waterfall_totals["num_far_background_pixels"], waterfall_totals["num_valid_background_pixels"]),
            "anchor_high_ratio": safe_div(waterfall_totals["num_anchor_high_pixels"], waterfall_totals["num_far_background_pixels"]),
            "anchor_disagreement_ratio": safe_div(waterfall_totals["num_anchor_disagreement_pixels"], waterfall_totals["num_far_background_pixels"]),
            "nonflat_ratio": safe_div(waterfall_totals["num_nonflat_pixels"], waterfall_totals["num_far_background_pixels"]),
            "candidate_to_budget_ratio": safe_div(waterfall_totals["num_candidate_pixels"], waterfall_totals["num_selected_pixels"]),
            "flat_bg_ratio": safe_div(candidate_type["flat_bg_pixels"], waterfall_totals["num_candidate_pixels"]),
            "ohem_fp_component_coverage": coverage_mean,
        },
        "candidate_type_breakdown": candidate_type,
        "strong_gate_pass": (
            mean(ratios) >= 2.0
            and safe_div(empty, num_images) <= 0.05
            and coverage_mean >= 0.50
            and flat_mean <= 0.25
            and target_leakage_total == 0
        ),
        "params": {
            "threshold": args.threshold,
            "target_dilation_radius": args.target_dilation_radius,
            "tau_low": args.tau_low,
            "tau_high": args.tau_high,
            "hard_top_q": args.hard_top_q,
        },
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "components": str(output_dir / "components.csv"),
            "candidate_dir": str(candidate_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
