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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TrainSetLoader
from loss import SelfPerturbationStabilityLoss
from net import Net
from utils import get_img_norm_cfg


def build_view(img: torch.Tensor, gt_mask: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, str]:
    op = args.sps_perturbation
    if op == "hflip":
        return torch.flip(img, dims=[-1]), torch.flip(gt_mask, dims=[-1]), "hflip"
    if op == "vflip":
        return torch.flip(img, dims=[-2]), torch.flip(gt_mask, dims=[-2]), "vflip"
    if op == "hvflip":
        return torch.flip(img, dims=[-2, -1]), torch.flip(gt_mask, dims=[-2, -1]), "hvflip"
    if op == "transpose":
        return img.transpose(-1, -2).contiguous(), gt_mask.transpose(-1, -2).contiguous(), "transpose"
    if op == "gain_offset":
        gain = random.uniform(args.sps_gain_min, args.sps_gain_max)
        offset = random.uniform(-args.sps_offset_abs, args.sps_offset_abs)
        return img * gain + offset, gt_mask, "identity"
    if op == "gaussian_noise":
        return img + torch.randn_like(img) * args.sps_noise_std, gt_mask, "identity"
    raise ValueError(f"Unsupported perturbation: {op}")


def load_checkpoint(net: Net, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"[warn] missing keys while loading checkpoint: {len(missing)}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Census SPS candidate coverage on training crops.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="MSHNetSPSOHEM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=40)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--lambda_variant", type=float, default=0.2)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--sps_objective", default="additive", choices=["additive", "rerank"])
    parser.add_argument("--sps_lambda", type=float, default=0.0)
    parser.add_argument("--sps_perturbation", default="gain_offset",
                        choices=["hflip", "vflip", "hvflip", "transpose", "gain_offset", "gaussian_noise"])
    parser.add_argument("--sps_gain_min", type=float, default=1.005)
    parser.add_argument("--sps_gain_max", type=float, default=1.005)
    parser.add_argument("--sps_offset_abs", type=float, default=0.0)
    parser.add_argument("--sps_noise_std", type=float, default=0.02)
    parser.add_argument("--sps_candidate_tau", type=float, default=0.5)
    parser.add_argument("--sps_candidate_topk_ratio", type=float, default=0.0)
    parser.add_argument("--sps_candidate_topk_metric", default="confidence",
                        choices=["confidence", "instability", "sps_score", "target_margin_instability", "target_margin_sps_score", "target_contrast_instability", "target_contrast_sps_score"])
    parser.add_argument("--sps_candidate_min_metric", type=float, default=None)
    parser.add_argument("--sps_candidate_min_confidence", type=float, default=0.0)
    parser.add_argument("--sps_candidate_fallback_topk_ratio", type=float, default=0.0)
    parser.add_argument("--sps_candidate_expand_radius", type=int, default=0)
    parser.add_argument("--sps_candidate_expand_min_confidence", type=float, default=0.0)
    parser.add_argument("--sps_target_margin_quantile", type=float, default=0.85)
    parser.add_argument("--sps_target_margin_temp", type=float, default=0.01)
    parser.add_argument("--sps_target_margin_min", type=float, default=0.0)
    parser.add_argument("--sps_rerank_strict_fallback", dest="sps_rerank_strict_fallback", action="store_true", default=True)
    parser.add_argument("--sps_no_rerank_strict_fallback", dest="sps_rerank_strict_fallback", action="store_false")
    parser.add_argument("--sps_budget_q", type=float, default=0.1)
    parser.add_argument("--sps_kmax", type=int, default=256)
    parser.add_argument("--sps_eta", type=float, default=1.0)
    parser.add_argument("--sps_mode", default="sps",
                        choices=["sps", "confidence_only", "instability_only", "target_margin", "global_consistency", "none"])
    parser.add_argument("--sps_dilate_radius", type=int, default=5)
    parser.add_argument("--sps_adaptive_radius", action="store_true", default=True)
    parser.add_argument("--sps_fixed_radius", dest="sps_adaptive_radius", action="store_false")
    parser.add_argument("--sps_radius_kappa", type=float, default=1.0)
    parser.add_argument("--sps_radius_r0", type=float, default=2.0)
    parser.add_argument("--sps_radius_min", type=int, default=3)
    parser.add_argument("--sps_radius_max", type=int, default=9)
    parser.add_argument("--sps_target_safe", action="store_true", default=False)
    parser.add_argument("--sps_target_safe_u_low", type=float, default=0.02)
    parser.add_argument("--sps_target_safe_u_high", type=float, default=0.08)
    parser.add_argument("--sps_target_safe_conf_min", type=float, default=0.55)
    parser.add_argument("--sps_target_safe_conf_floor", type=float, default=0.35)
    parser.add_argument("--sps_target_safe_alpha_floor", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    train_set = TrainSetLoader(args.dataset_dir, args.dataset_name, args.patch_size, img_norm_cfg=img_norm_cfg)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=generator)

    net = Net(model_name=args.model_name, mode="train", loss_cfg=vars(args)).to(device)
    load_checkpoint(net, Path(args.checkpoint), device)
    net.eval()
    sps_loss = SelfPerturbationStabilityLoss(
        dilate_radius=args.sps_dilate_radius,
        candidate_tau=args.sps_candidate_tau,
        candidate_topk_ratio=args.sps_candidate_topk_ratio,
        candidate_topk_metric=args.sps_candidate_topk_metric,
        candidate_min_metric=args.sps_candidate_min_metric,
        candidate_min_confidence=args.sps_candidate_min_confidence,
        candidate_fallback_topk_ratio=args.sps_candidate_fallback_topk_ratio,
        candidate_expand_radius=args.sps_candidate_expand_radius,
        candidate_expand_min_confidence=args.sps_candidate_expand_min_confidence,
        target_margin_quantile=args.sps_target_margin_quantile,
        target_margin_temp=args.sps_target_margin_temp,
        target_margin_min=args.sps_target_margin_min,
        rerank_strict_fallback=args.sps_rerank_strict_fallback,
        budget_q=args.sps_budget_q,
        kmax=args.sps_kmax,
        eta=args.sps_eta,
        mode=args.sps_mode,
        adaptive_radius=args.sps_adaptive_radius,
        radius_kappa=args.sps_radius_kappa,
        radius_r0=args.sps_radius_r0,
        radius_min=args.sps_radius_min,
        radius_max=args.sps_radius_max,
        target_safe=args.sps_target_safe,
        target_safe_u_low=args.sps_target_safe_u_low,
        target_safe_u_high=args.sps_target_safe_u_high,
        target_safe_conf_min=args.sps_target_safe_conf_min,
        target_safe_conf_floor=args.sps_target_safe_conf_floor,
        target_safe_alpha_floor=args.sps_target_safe_alpha_floor,
    )

    rows = []
    with torch.no_grad():
        for batch_idx, (img, gt_mask) in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            img = img.to(device)
            gt_mask = gt_mask.to(device)
            sps_img, _, op = build_view(img, gt_mask, args)
            final_logit = net.export_logits_features(img)["logit"]
            perturb_logit = net.export_logits_features(sps_img)["logit"]
            if args.sps_objective == "rerank" and args.sps_mode != "global_consistency":
                loss, stats = sps_loss.rerank_ohem_loss(
                    final_logit,
                    perturb_logit,
                    gt_mask,
                    op=op,
                    topk_ratio=args.ohem_ratio,
                    alpha=args.sps_lambda,
                )
            else:
                loss, stats = sps_loss(final_logit, perturb_logit, gt_mask, op=op)
            pixels = float(final_logit.numel())
            candidates = float(stats["sps_candidate_pixels"].detach().cpu())
            selected = float(stats["sps_hard_pixels"].detach().cpu())
            rows.append({
                "batch": batch_idx,
                "pixels": pixels,
                "candidate_pixels": candidates,
                "selected_pixels": selected,
                "candidate_ratio": candidates / max(1.0, pixels),
                "selected_ratio": selected / max(1.0, pixels),
                "sps_loss": float(loss.detach().cpu()),
                "instability_mean_selected": float(stats["sps_instability_mean"].detach().cpu()),
                "confidence_mean_selected": float(stats["sps_conf_mean"].detach().cpu()),
                "score_mean_selected": float(stats["sps_score_mean"].detach().cpu()),
                "ohem_jaccard": float(stats.get("sps_ohem_jaccard", torch.tensor(float("nan"))).detach().cpu()),
                "fallback_images": float(stats.get("sps_fallback_images", torch.tensor(0.0)).detach().cpu()),
                "target_alpha_scale": float(stats.get("sps_target_alpha_scale", torch.tensor(float("nan"))).detach().cpu()),
                "target_instability_mean": float(stats.get("sps_target_instability_mean", torch.tensor(float("nan"))).detach().cpu()),
                "target_confidence_mean": float(stats.get("sps_target_conf_mean", torch.tensor(float("nan"))).detach().cpu()),
            })
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches", flush=True)

    write_csv(output_dir / "sps_training_candidate_batches.csv", rows)
    candidate_values = np.asarray([row["candidate_pixels"] for row in rows], dtype=np.float64)
    selected_values = np.asarray([row["selected_pixels"] for row in rows], dtype=np.float64)
    pixel_values = np.asarray([row["pixels"] for row in rows], dtype=np.float64)
    jaccard_values = np.asarray([row["ohem_jaccard"] for row in rows], dtype=np.float64)
    fallback_values = np.asarray([row["fallback_images"] for row in rows], dtype=np.float64)
    summary = {
        "dataset": args.dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_name": args.model_name,
        "seed": args.seed,
        "batches": len(rows),
        "images": int(len(rows) * args.batch_size),
        "sps_mode": args.sps_mode,
        "sps_objective": args.sps_objective,
        "sps_lambda": args.sps_lambda,
        "sps_perturbation": args.sps_perturbation,
        "sps_candidate_tau": args.sps_candidate_tau,
        "sps_candidate_topk_ratio": args.sps_candidate_topk_ratio,
        "sps_candidate_topk_metric": args.sps_candidate_topk_metric,
        "sps_candidate_min_metric": args.sps_candidate_min_metric,
        "sps_candidate_min_confidence": args.sps_candidate_min_confidence,
        "sps_candidate_fallback_topk_ratio": args.sps_candidate_fallback_topk_ratio,
        "sps_candidate_expand_radius": args.sps_candidate_expand_radius,
        "sps_candidate_expand_min_confidence": args.sps_candidate_expand_min_confidence,
        "sps_target_margin_quantile": args.sps_target_margin_quantile,
        "sps_target_margin_temp": args.sps_target_margin_temp,
        "sps_target_margin_min": args.sps_target_margin_min,
        "sps_rerank_strict_fallback": args.sps_rerank_strict_fallback,
        "sps_budget_q": args.sps_budget_q,
        "sps_kmax": args.sps_kmax,
        "mean_candidate_pixels_per_batch": float(candidate_values.mean()) if rows else 0.0,
        "mean_selected_pixels_per_batch": float(selected_values.mean()) if rows else 0.0,
        "mean_candidate_ratio": float(candidate_values.sum() / max(1.0, pixel_values.sum())) if rows else 0.0,
        "mean_selected_ratio": float(selected_values.sum() / max(1.0, pixel_values.sum())) if rows else 0.0,
        "mean_ohem_jaccard": float(np.nanmean(jaccard_values)) if rows else 0.0,
        "mean_fallback_images_per_batch": float(fallback_values.mean()) if rows else 0.0,
        "nonzero_candidate_batch_fraction": float((candidate_values > 0).sum() / max(1, len(rows))),
        "nonzero_selected_batch_fraction": float((selected_values > 0).sum() / max(1, len(rows))),
        "candidate_pixel_quantiles": {
            "q50": float(np.quantile(candidate_values, 0.50)) if rows else 0.0,
            "q75": float(np.quantile(candidate_values, 0.75)) if rows else 0.0,
            "q90": float(np.quantile(candidate_values, 0.90)) if rows else 0.0,
        },
        "outputs": {
            "per_batch": str(output_dir / "sps_training_candidate_batches.csv"),
        },
    }
    (output_dir / "sps_training_candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
