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

from dataset import TrainSetLoader
from net import Net
from tools.official.check_mscv_gate_a import checkpoint_state_dict, load_mscv_evidence_checkpoint, torch_load_checkpoint
from utils import get_img_norm_cfg, seed_pytorch
from utils.mscv_candidate import build_mscv_candidate_mask, topk_mask


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="MSCV Gate-B candidate/verifier trainability audit.")
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
    parser.add_argument("--mscv_beta_max", type=float, default=0.1)
    parser.add_argument("--mscv_hidden_channels", type=int, default=32)
    parser.add_argument("--mscv_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--mscv_contrast_kernel", type=int, default=9)
    parser.add_argument("--mscv_far_radius", type=int, default=7)
    parser.add_argument("--mscv_candidate_prob_thr", type=float, default=0.2)
    parser.add_argument("--mscv_candidate_std_thr", type=float, default=0.05)
    parser.add_argument("--mscv_nonflat_thr", type=float, default=0.05)
    parser.add_argument("--top_target_k", type=int, default=20)
    parser.add_argument("--min_candidate_to_budget_ratio", type=float, default=2.0)
    parser.add_argument("--max_target_top20_rate", type=float, default=0.15)
    parser.add_argument("--min_selected_ohem_overlap", type=float, default=0.30)
    parser.add_argument("--max_selected_ohem_overlap", type=float, default=0.80)
    parser.add_argument("--min_nonflat_candidate_ratio", type=float, default=0.70)
    parser.add_argument("--min_pstd_enrichment_vs_ohem", type=float, default=1.5)
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
    mscv_cfg = dict(vars(args))
    mscv_cfg["mscv_eval_beta"] = 0.0
    net = Net("MSCVMSHNet", mode="test", loss_cfg=mscv_cfg).to(device)
    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    _ = checkpoint_state_dict(checkpoint)
    load_mscv_evidence_checkpoint(net, args.checkpoint, device)
    net.eval()

    rows = []
    leakage_total = 0
    num_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
    with torch.no_grad():
        for idx, (img, gt_mask, image_ids, _aug_ops) in enumerate(loader):
            if args.max_images and idx >= args.max_images:
                break
            img = img.to(device)
            gt_mask = gt_mask.to(device)
            image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else str(image_ids)
            export = net.export_logits_features(img)
            p_max = export["p_max"]
            p_std = export["p_std"]
            evidence_prob = torch.sigmoid(export["evidence_logit"].detach())
            local_contrast = export["local_contrast"]
            cand = build_mscv_candidate_mask(
                p_max,
                p_std,
                gt_mask,
                far_radius=args.mscv_far_radius,
                candidate_prob_thr=args.mscv_candidate_prob_thr,
                candidate_std_thr=args.mscv_candidate_std_thr,
                local_contrast=local_contrast,
                nonflat_thr=args.mscv_nonflat_thr,
            )
            candidate = cand["candidate"].bool()
            base_candidate = cand["base_candidate"].bool()
            target_near = cand["target_near"].bool()
            far_mask = cand["far_mask"].bool()
            nonflat_mask = cand["nonflat_mask"].bool()
            candidate_pixels = int(candidate.sum().item())
            base_pixels = int(base_candidate.sum().item())
            far_pixels = int(far_mask.sum().item())
            budget = max(1, int(far_pixels * float(args.ohem_ratio))) if far_pixels > 0 else 0
            score = (p_max.detach() * p_std.detach()).float()
            ohem_mask = topk_mask(evidence_prob, far_mask, budget)
            ohem_pixels = int(ohem_mask.sum().item())
            overlap_pixels = int((candidate & ohem_mask).sum().item())
            leakage = int((candidate & target_near).sum().item())
            leakage_total += leakage
            top_k = min(int(args.top_target_k), score.numel())
            top_all = topk_mask(score, torch.ones_like(score, dtype=torch.bool), top_k)
            target_top_rate = safe_div(int((top_all & target_near).sum().item()), int(top_all.sum().item()))
            candidate_to_budget = safe_div(candidate_pixels, budget)
            selected_ohem_overlap = safe_div(overlap_pixels, ohem_pixels)
            nonflat_candidate_ratio = safe_div(int((base_candidate & nonflat_mask).sum().item()), base_pixels)
            candidate_pstd_mean = float(p_std[candidate].mean().detach().cpu()) if candidate_pixels else 0.0
            ohem_pstd_mean = float(p_std[ohem_mask].mean().detach().cpu()) if ohem_pixels else 0.0
            pstd_enrichment = safe_div(candidate_pstd_mean, ohem_pstd_mean)
            rows.append(
                {
                    "image_id": image_id,
                    "candidate_pixels": candidate_pixels,
                    "base_candidate_pixels": base_pixels,
                    "far_pixels": far_pixels,
                    "ohem_budget": budget,
                    "ohem_pixels": ohem_pixels,
                    "candidate_to_budget_ratio": candidate_to_budget,
                    "target_leakage_pixels": leakage,
                    "target_top20_rate": target_top_rate,
                    "selected_ohem_overlap": selected_ohem_overlap,
                    "nonflat_candidate_ratio": nonflat_candidate_ratio,
                    "candidate_pstd_mean": candidate_pstd_mean,
                    "ohem_pstd_mean": ohem_pstd_mean,
                    "pstd_enrichment_vs_ohem": pstd_enrichment,
                }
            )
            if (idx + 1) % 100 == 0:
                print(f"Gate-MSCV-B checked [{idx + 1}/{num_images}]", flush=True)

    summary = {
        "gate": "MSCV_Gate_B_candidate_trainability",
        "dataset": args.dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "num_images": len(rows),
        "candidate_to_budget_ratio_mean": mean([row["candidate_to_budget_ratio"] for row in rows]),
        "target_leakage_pixels_total": int(leakage_total),
        "target_top20_rate": mean([row["target_top20_rate"] for row in rows]),
        "selected_ohem_overlap_mean": mean([row["selected_ohem_overlap"] for row in rows]),
        "nonflat_candidate_ratio_mean": mean([row["nonflat_candidate_ratio"] for row in rows]),
        "pstd_enrichment_vs_ohem": mean([row["pstd_enrichment_vs_ohem"] for row in rows]),
        "thresholds": {
            "candidate_to_budget_ratio_mean": args.min_candidate_to_budget_ratio,
            "target_top20_rate": args.max_target_top20_rate,
            "selected_ohem_overlap_range": [args.min_selected_ohem_overlap, args.max_selected_ohem_overlap],
            "nonflat_candidate_ratio_mean": args.min_nonflat_candidate_ratio,
            "pstd_enrichment_vs_ohem": args.min_pstd_enrichment_vs_ohem,
        },
        "checks": {},
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    checks = {
        "candidate_to_budget_ratio_mean": summary["candidate_to_budget_ratio_mean"] >= args.min_candidate_to_budget_ratio,
        "target_leakage_pixels_total": summary["target_leakage_pixels_total"] == 0,
        "target_top20_rate": summary["target_top20_rate"] <= args.max_target_top20_rate,
        "selected_ohem_overlap_min": summary["selected_ohem_overlap_mean"] >= args.min_selected_ohem_overlap,
        "selected_ohem_overlap_max": summary["selected_ohem_overlap_mean"] <= args.max_selected_ohem_overlap,
        "nonflat_candidate_ratio_mean": summary["nonflat_candidate_ratio_mean"] >= args.min_nonflat_candidate_ratio,
        "pstd_enrichment_vs_ohem": summary["pstd_enrichment_vs_ohem"] >= args.min_pstd_enrichment_vs_ohem,
    }
    summary["checks"] = checks
    summary["gate_pass"] = all(checks.values())
    write_csv(output_dir / "per_image.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
