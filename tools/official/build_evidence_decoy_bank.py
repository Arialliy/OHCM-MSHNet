#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TrainSetLoader
from net import Net
from utils import get_img_norm_cfg, seed_pytorch
from utils.evidence_conditioned_decoy import generate_evidence_conditioned_decoy, sample_safe_centers


def torch_load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def checkpoint_state_dict(checkpoint):
    return checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint


def load_ohem_checkpoint(net, checkpoint_path, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return checkpoint if isinstance(checkpoint, dict) else {}


def to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.uint8)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def write_preview(path, image, residual, mask):
    image_np = image[0, 0].detach().cpu().numpy()
    residual_np = residual[0, 0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy() > 0
    base = to_uint8(image_np)
    residual_vis = to_uint8(np.abs(residual_np))
    overlay = np.stack([base, base, base], axis=-1).astype(np.float32)
    overlay[mask_np, 0] = 255.0
    overlay[mask_np, 1] *= 0.35
    overlay[mask_np, 2] *= 0.35
    panel = np.concatenate(
        [
            np.stack([base, base, base], axis=-1),
            np.stack([residual_vis, residual_vis, residual_vis], axis=-1),
            np.clip(overlay, 0, 255).astype(np.uint8),
        ],
        axis=1,
    )
    Image.fromarray(panel).save(path)


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def write_csv(path, rows):
    fieldnames = [
        "image_id",
        "decoy_id",
        "residual_path",
        "mask_path",
        "center_y",
        "center_x",
        "area",
        "prob_before_max",
        "prob_after_max",
        "prob_after_topk",
        "prob_gain",
        "contrast_z",
        "target_dilate_overlap_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(args, rows, attempts, num_images, overlaps, rejects):
    gains = [float(row["prob_gain"]) for row in rows]
    before = [float(row["prob_before_max"]) for row in rows]
    after = [float(row["prob_after_max"]) for row in rows]
    areas = [float(row["area"]) for row in rows]
    contrast = [float(row["contrast_z"]) for row in rows]
    area_ok = [args.min_area <= area <= args.max_area for area in areas]
    flat = [value < args.min_contrast_z for value in contrast]
    success_ratio = float(len(rows)) / float(attempts) if attempts else 0.0
    summary = {
        "gate": "ECDV_Gate_B_evidence_conditioned_decoy_bank",
        "dataset": args.dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_images": int(num_images),
        "candidate_attempts": int(attempts),
        "decoys_total": int(len(rows)),
        "decoys_per_image_mean": float(len(rows)) / float(max(1, num_images)),
        "target_dilate_overlap_pixels": int(sum(overlaps)),
        "evidence_response_success_ratio": success_ratio,
        "mean_prob_before": safe_mean(before),
        "mean_prob_after": safe_mean(after),
        "mean_prob_gain": safe_mean(gains),
        "area_in_target_range_ratio": float(np.mean(area_ok)) if area_ok else 0.0,
        "flat_artifact_ratio": float(np.mean(flat)) if flat else 1.0,
        "preview_audit_pass": bool(rows),
        "checks": {},
        "reject_counts": rejects,
        "thresholds": {
            "min_decoys_per_image": args.min_decoys_per_image,
            "min_evidence_success_ratio": args.min_evidence_success_ratio,
            "min_mean_prob_gain": args.min_mean_prob_gain,
            "min_area_in_target_range_ratio": args.min_area_in_target_range_ratio,
            "max_flat_artifact_ratio": args.max_flat_artifact_ratio,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "min_contrast_z": args.min_contrast_z,
        },
        "outputs": {
            "summary": str(Path(args.output_dir) / "summary.json"),
            "decoy_rows": str(Path(args.output_dir) / "decoy_rows.csv"),
            "residuals": str(Path(args.output_dir) / "residuals"),
            "masks": str(Path(args.output_dir) / "masks"),
            "previews": str(Path(args.output_dir) / "previews"),
        },
    }
    checks = {
        "target_dilate_overlap_pixels": summary["target_dilate_overlap_pixels"] == 0,
        "decoys_per_image_mean": summary["decoys_per_image_mean"] >= args.min_decoys_per_image,
        "evidence_response_success_ratio": summary["evidence_response_success_ratio"] >= args.min_evidence_success_ratio,
        "mean_prob_gain": summary["mean_prob_gain"] >= args.min_mean_prob_gain,
        "area_in_target_range_ratio": summary["area_in_target_range_ratio"] >= args.min_area_in_target_range_ratio,
        "flat_artifact_ratio": summary["flat_artifact_ratio"] <= args.max_flat_artifact_ratio,
        "preview_audit_pass": summary["preview_audit_pass"] is True,
    }
    summary["checks"] = checks
    summary["gate_pass"] = all(checks.values())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build a Gate-B evidence-conditioned decoy bank for ECDV-MSHNet.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_centers", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--patch_radius", type=int, default=5)
    parser.add_argument("--target_dilate_radius", type=int, default=9)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--l2_weight", type=float, default=0.001)
    parser.add_argument("--max_delta", type=float, default=0.5)
    parser.add_argument("--response_threshold", type=float, default=0.5)
    parser.add_argument("--min_gain", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min_area", type=float, default=10.0)
    parser.add_argument("--max_area", type=float, default=160.0)
    parser.add_argument("--min_contrast_z", type=float, default=0.20)
    parser.add_argument("--min_decoys_per_image", type=float, default=1.0)
    parser.add_argument("--min_evidence_success_ratio", type=float, default=0.50)
    parser.add_argument("--min_mean_prob_gain", type=float, default=0.20)
    parser.add_argument("--min_area_in_target_range_ratio", type=float, default=0.80)
    parser.add_argument("--max_flat_artifact_ratio", type=float, default=0.30)
    parser.add_argument("--preview_limit", type=int, default=128)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    residual_dir = output_dir / "residuals"
    mask_dir = output_dir / "masks"
    preview_dir = output_dir / "previews"
    for directory in (residual_dir, mask_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

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
    evidence = Net("MSHNetOHEM", mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_ohem_checkpoint(evidence, args.checkpoint, device)

    rows = []
    attempts = 0
    overlaps = []
    rejects = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    num_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))

    for idx, batch in enumerate(loader):
        if args.max_images and idx >= args.max_images:
            break
        img, gt_mask, image_ids, _aug_ops = batch
        image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else str(image_ids)
        img = img.to(device)
        gt_mask = gt_mask.to(device)
        centers = sample_safe_centers(
            gt_mask,
            num_centers=args.num_centers,
            patch_radius=args.patch_radius,
            target_dilate_radius=args.target_dilate_radius,
            generator=generator,
        )
        if centers.numel() == 0:
            rejects["no_safe_center"] = rejects.get("no_safe_center", 0) + 1
            continue
        for center_idx, center in enumerate(centers):
            attempts += 1
            image_aug, pseudo_mask, stats = generate_evidence_conditioned_decoy(
                img,
                gt_mask,
                evidence,
                center=(int(center[0].item()), int(center[1].item())),
                patch_radius=args.patch_radius,
                steps=args.steps,
                lr=args.lr,
                target_dilate_radius=args.target_dilate_radius,
                tv_weight=args.tv_weight,
                l2_weight=args.l2_weight,
                max_delta=args.max_delta,
                response_threshold=args.response_threshold,
                min_gain=args.min_gain,
                topk=args.topk,
            )
            overlaps.append(int(stats.get("target_dilate_overlap_pixels", 0)))
            if not stats.get("accepted", False):
                reason = stats.get("reject_reason", "unknown") or "unknown"
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            residual = stats["residual"].detach().cpu().numpy().astype(np.float32)[0, 0]
            mask = pseudo_mask.detach().cpu().numpy().astype(np.float32)[0, 0]
            decoy_id = f"{image_id}_{center_idx}_{len(rows):06d}"
            residual_name = f"{decoy_id}.npy"
            mask_name = f"{decoy_id}.npy"
            np.save(residual_dir / residual_name, residual)
            np.save(mask_dir / mask_name, mask)
            if len(rows) < args.preview_limit:
                write_preview(preview_dir / f"{decoy_id}.png", image_aug.detach().cpu(), stats["residual"].detach().cpu(), pseudo_mask.detach().cpu())
            rows.append(
                {
                    "image_id": image_id,
                    "decoy_id": decoy_id,
                    "residual_path": str(Path("residuals") / residual_name),
                    "mask_path": str(Path("masks") / mask_name),
                    "center_y": int(center[0].item()),
                    "center_x": int(center[1].item()),
                    "area": float(stats.get("area", 0.0)),
                    "prob_before_max": float(stats.get("prob_before_max", 0.0)),
                    "prob_after_max": float(stats.get("prob_after_max", 0.0)),
                    "prob_after_topk": float(stats.get("prob_after_topk", 0.0)),
                    "prob_gain": float(stats.get("prob_gain", 0.0)),
                    "contrast_z": float(stats.get("contrast_z", 0.0)),
                    "target_dilate_overlap_pixels": int(stats.get("target_dilate_overlap_pixels", 0)),
                }
            )
        if (idx + 1) % 25 == 0:
            print(f"ECDV decoy bank [{idx + 1}/{num_images}] accepted={len(rows)} attempts={attempts}", flush=True)

    write_csv(output_dir / "decoy_rows.csv", rows)
    summary = build_summary(args, rows, attempts, num_images, overlaps, rejects)
    summary["epoch"] = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
