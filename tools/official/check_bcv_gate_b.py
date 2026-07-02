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
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TrainSetLoader
from net import Net
from probability import foreground_probability
from tools.official.check_bcv_gate_a import checkpoint_state_dict, torch_load_checkpoint
from utils import get_img_norm_cfg, seed_pytorch
from utils.mscv_candidate import dilate_mask


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def mean(values):
    return float(np.mean(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fixed_context_background(image, kernel_size):
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(image[:, :1], kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def local_contrast(image, kernel_size):
    bg = fixed_context_background(image, kernel_size)
    contrast = torch.abs(image[:, :1] - bg)
    denom = contrast.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return contrast / denom


def binary_auc(pos_values, neg_values, max_samples=4096):
    pos = np.asarray(pos_values, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg_values, dtype=np.float64).reshape(-1)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    if pos.size > max_samples:
        pos = pos[np.linspace(0, pos.size - 1, max_samples).astype(np.int64)]
    if neg.size > max_samples:
        neg = neg[np.linspace(0, neg.size - 1, max_samples).astype(np.int64)]
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, dtype=np.int32), np.zeros(neg.size, dtype=np.int32)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    pos_rank_sum = ranks[labels == 1].sum()
    auc = (pos_rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def add_split_decision(summary: dict, args) -> dict:
    background_residual_gate_pass = (
        summary["residual_auroc_target_vs_far_mean"] >= args.min_residual_auroc
        and summary["target_residual_bg_ratio_mean"] >= args.min_target_residual_bg_ratio
    )
    candidate_mining_gate_pass = (
        summary["candidate_to_budget_ratio_mean"] >= args.min_candidate_to_budget_ratio
    )
    summary["background_residual_gate_pass"] = bool(background_residual_gate_pass)
    summary["candidate_mining_gate_pass"] = bool(candidate_mining_gate_pass)
    summary["legacy_all_checks_pass"] = bool(all(summary["checks"].values()))
    summary["overall_decision"] = (
        "PROCEED_TO_FP_RESIDUAL_AUDIT" if background_residual_gate_pass else "STOP_BCV"
    )
    summary["gate_pass"] = bool(background_residual_gate_pass)
    return summary


def main():
    parser = argparse.ArgumentParser(description="BCV Gate-B background/residual trainability audit.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--bcv_far_radius", type=int, default=7)
    parser.add_argument("--bcv_candidate_prob_thr", type=float, default=0.3)
    parser.add_argument("--background_kernel", type=int, default=31)
    parser.add_argument("--flat_contrast_thr", type=float, default=0.05)
    parser.add_argument("--low_residual_thr", type=float, default=1.0)
    parser.add_argument("--max_bg_reconstruction_error", type=float, default=0.50)
    parser.add_argument("--min_target_residual_bg_ratio", type=float, default=1.5)
    parser.add_argument("--min_candidate_to_budget_ratio", type=float, default=1.0)
    parser.add_argument("--max_flat_candidate_ratio", type=float, default=0.30)
    parser.add_argument("--min_residual_auroc", type=float, default=0.65)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    dataset = TrainSetLoader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        patch_size=args.patch_size,
        img_norm_cfg=img_norm_cfg,
        return_meta=True,
    )
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    net = Net("MSHNetOHEM", mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    net.eval()

    rows = []
    leakage_total = 0
    num_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
    with torch.no_grad():
        for idx, (img, gt_mask, image_ids, _aug_ops) in enumerate(loader):
            if args.max_images and idx >= args.max_images:
                break
            img = img.to(device)
            gt_mask = gt_mask.to(device).float()
            image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else str(image_ids)
            export = net.export_logits_features(img)
            evidence_prob = foreground_probability(export["logit"]).detach()
            bg = fixed_context_background(img, args.background_kernel)
            residual = torch.abs(img[:, :1] - bg)
            residual_norm = residual / (residual.mean(dim=(-2, -1), keepdim=True) + 1e-6)
            contrast = local_contrast(img, args.background_kernel)
            target = gt_mask
            if target.shape[-2:] != residual.shape[-2:]:
                target = F.interpolate(target, size=residual.shape[-2:], mode="nearest")
            target_near = dilate_mask(target, args.bcv_far_radius).bool()
            far_mask = ~target_near
            high_evidence = evidence_prob > float(args.bcv_candidate_prob_thr)
            low_residual = residual_norm < float(args.low_residual_thr)
            candidate = far_mask & high_evidence & low_residual
            flat_candidate = candidate & (contrast < float(args.flat_contrast_thr))
            candidate_pixels = int(candidate.sum().item())
            far_pixels = int(far_mask.sum().item())
            budget = max(1, int(far_pixels * float(args.ohem_ratio))) if far_pixels > 0 else 0
            leakage = int((candidate & target_near).sum().item())
            leakage_total += leakage
            target_pixels = int((target > 0).sum().item())
            bg_recon_error = float((residual * far_mask.float()).sum().detach().cpu() / (far_mask.float().sum().detach().cpu() + 1e-6))
            target_residual_mean = float(residual[target > 0].mean().detach().cpu()) if target_pixels else 0.0
            bg_residual_mean = float(residual[far_mask].mean().detach().cpu()) if far_pixels else 0.0
            target_bg_ratio = safe_div(target_residual_mean, bg_residual_mean)
            candidate_to_budget = safe_div(candidate_pixels, budget)
            flat_candidate_ratio = safe_div(int(flat_candidate.sum().item()), candidate_pixels)
            auroc = binary_auc(
                residual[target > 0].detach().cpu().numpy() if target_pixels else np.asarray([], dtype=np.float32),
                residual[far_mask].detach().cpu().numpy() if far_pixels else np.asarray([], dtype=np.float32),
            )
            rows.append(
                {
                    "image_id": image_id,
                    "bg_reconstruction_error": bg_recon_error,
                    "target_residual_mean": target_residual_mean,
                    "background_residual_mean": bg_residual_mean,
                    "target_residual_bg_ratio": target_bg_ratio,
                    "candidate_pixels": candidate_pixels,
                    "far_pixels": far_pixels,
                    "ohem_budget": budget,
                    "candidate_to_budget_ratio": candidate_to_budget,
                    "target_leakage_pixels": leakage,
                    "flat_candidate_ratio": flat_candidate_ratio,
                    "residual_auroc_target_vs_far": auroc,
                }
            )
            if (idx + 1) % 100 == 0:
                print(f"Gate-BCV-B checked [{idx + 1}/{num_images}]", flush=True)

    summary = {
        "gate": "BCV_Gate_B_background_residual_trainability",
        "dataset": args.dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "num_images": len(rows),
        "background_reconstruction_error_mean": mean([row["bg_reconstruction_error"] for row in rows]),
        "target_residual_bg_ratio_mean": mean([row["target_residual_bg_ratio"] for row in rows]),
        "candidate_to_budget_ratio_mean": mean([row["candidate_to_budget_ratio"] for row in rows]),
        "target_leakage_pixels_total": int(leakage_total),
        "flat_candidate_ratio_mean": mean([row["flat_candidate_ratio"] for row in rows]),
        "residual_auroc_target_vs_far_mean": mean([row["residual_auroc_target_vs_far"] for row in rows]),
        "thresholds": {
            "background_reconstruction_error_mean_max": args.max_bg_reconstruction_error,
            "target_residual_bg_ratio_mean_min": args.min_target_residual_bg_ratio,
            "candidate_to_budget_ratio_mean_min": args.min_candidate_to_budget_ratio,
            "target_leakage_pixels_total": 0,
            "flat_candidate_ratio_mean_max": args.max_flat_candidate_ratio,
            "residual_auroc_target_vs_far_mean_min": args.min_residual_auroc,
        },
        "checks": {},
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    checks = {
        "background_reconstruction_error": summary["background_reconstruction_error_mean"] <= args.max_bg_reconstruction_error,
        "target_residual_bg_ratio": summary["target_residual_bg_ratio_mean"] >= args.min_target_residual_bg_ratio,
        "candidate_to_budget_ratio": summary["candidate_to_budget_ratio_mean"] >= args.min_candidate_to_budget_ratio,
        "target_leakage": summary["target_leakage_pixels_total"] == 0,
        "flat_candidate_ratio": summary["flat_candidate_ratio_mean"] <= args.max_flat_candidate_ratio,
        "residual_auroc": summary["residual_auroc_target_vs_far_mean"] >= args.min_residual_auroc,
    }
    summary["checks"] = checks
    add_split_decision(summary, args)
    write_csv(output_dir / "per_image.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["overall_decision"] == "STOP_BCV":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
