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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from loss import build_training_masks, select_topk_far_background
from net import Net
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit PFR train-only far-background candidate masks.")
    parser.add_argument("--dataset", "--dataset_name", dest="dataset_name", default="NUDT-SIRST")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ohem_checkpoint", default="")
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--far_topk_ratio", type=float, default=0.005)
    parser.add_argument("--target_dilate", type=int, default=3)
    parser.add_argument("--far_dilate", type=int, default=9)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def train_ids(dataset_dir: str, dataset_name: str) -> list[str]:
    path = Path(dataset_dir) / dataset_name / "img_idx" / f"train_{dataset_name}.txt"
    if not path.exists():
        fallback = Path(dataset_dir) / dataset_name / "img_idx" / "train.txt"
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(str(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_checkpoint(net: Net, checkpoint_path: str, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def default_ohem_checkpoint(args) -> str:
    if args.ohem_checkpoint:
        return args.ohem_checkpoint
    return (
        f"/home/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/"
        f"MSHNetOHEM/{args.dataset_name}/seed_{args.seed}/checkpoints/{args.dataset_name}/MSHNetOHEM_400.pth.tar"
    )


def forward_logit(net: Net, img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    export = net.export_logits_features(img)
    return export["logit"][:, :, :h, :w]


def ohem_negative_mask(logits: torch.Tensor, target: torch.Tensor, ohem_ratio: float) -> torch.Tensor:
    target = target.float()
    loss_map = F.binary_cross_entropy_with_logits(logits.detach(), target, reduction="none")
    valid_bg = target <= 0
    out = torch.zeros_like(valid_bg, dtype=torch.bool)
    flat_loss = loss_map.reshape(loss_map.shape[0], -1)
    flat_bg = valid_bg.reshape(valid_bg.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    for b in range(logits.shape[0]):
        idx = torch.nonzero(flat_bg[b], as_tuple=False).flatten()
        if idx.numel() < 1:
            continue
        k = max(1, int(idx.numel() * float(ohem_ratio)))
        k = min(k, idx.numel())
        selected = torch.topk(flat_loss[b, idx], k=k, largest=True).indices
        flat_out[b, idx[selected]] = True
    return out


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "image_name",
        "candidate_pixels",
        "target_leakage_pixels",
        "target_protect_pixels",
        "boundary_protect_pixels",
        "ohem_negative_pixels",
        "ohem_overlap_pixels",
        "ohem_overlap_fraction",
        "far_hard_logit_mean",
        "far_hard_prob_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def main():
    args = parse_args()
    if args.split.lower() != "train":
        raise ValueError("PFR candidate audit is train-only.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = train_ids(args.dataset_dir, args.dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, args.dataset_name, args.dataset_name, img_norm_cfg)
    dataset.test_list = ids
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

    checkpoint_path = default_ohem_checkpoint(args)
    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, checkpoint_path, device)

    rows = []
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            target = gt_mask[:, :, :h, :w].to(device).float()
            logits = forward_logit(net, img, h, w)
            target_mask, boundary_mask, far_bg_mask = build_training_masks(
                target,
                target_dilate=args.target_dilate,
                far_dilate=args.far_dilate,
            )
            candidates = select_topk_far_background(logits, far_bg_mask, args.far_topk_ratio)
            ohem_neg = ohem_negative_mask(logits, target, args.ohem_ratio)
            overlap = candidates & ohem_neg
            cand_pixels = int(candidates.sum().item())
            ohem_pixels = int(ohem_neg.sum().item())
            prob = torch.sigmoid(logits)
            rows.append(
                {
                    "image_name": name,
                    "candidate_pixels": cand_pixels,
                    "target_leakage_pixels": int((candidates & target_mask).sum().item()),
                    "target_protect_pixels": int(target_mask.sum().item()),
                    "boundary_protect_pixels": int(boundary_mask.sum().item()),
                    "ohem_negative_pixels": ohem_pixels,
                    "ohem_overlap_pixels": int(overlap.sum().item()),
                    "ohem_overlap_fraction": float(overlap.sum().item()) / max(1, cand_pixels),
                    "far_hard_logit_mean": float(logits[candidates].mean().item()) if cand_pixels else 0.0,
                    "far_hard_prob_mean": float(prob[candidates].mean().item()) if cand_pixels else 0.0,
                }
            )
            if (idx + 1) % 100 == 0:
                print("Audited PFR candidates [%d/%d]" % (idx + 1, len(loader)), flush=True)

    candidate_counts = [row["candidate_pixels"] for row in rows]
    fail_reasons = []
    candidate_empty_image_ratio = sum(count <= 0 for count in candidate_counts) / max(1, len(candidate_counts))
    target_leakage_pixels = int(sum(row["target_leakage_pixels"] for row in rows))
    far_hard_candidate_count_mean = finite_mean(candidate_counts)
    boundary_protect_pixel_count_mean = finite_mean([row["boundary_protect_pixels"] for row in rows])
    target_protect_pixel_count_mean = finite_mean([row["target_protect_pixels"] for row in rows])
    if candidate_empty_image_ratio > 0.05:
        fail_reasons.append("candidate_empty_image_ratio_gt_0p05")
    if target_leakage_pixels != 0:
        fail_reasons.append("target_leakage_pixels_nonzero")
    if far_hard_candidate_count_mean <= 0:
        fail_reasons.append("far_hard_candidate_count_mean_le_0")
    if boundary_protect_pixel_count_mean <= 0:
        fail_reasons.append("boundary_protect_pixel_count_mean_le_0")

    summary = {
        "gate_pass": len(fail_reasons) == 0,
        "num_images": len(rows),
        "dataset": args.dataset_name,
        "split": args.split,
        "seed": args.seed,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "candidate_to_budget_ratio_mean": finite_mean(
            [
                row["candidate_pixels"] / max(1, row["ohem_negative_pixels"])
                for row in rows
            ]
        ),
        "candidate_empty_image_ratio": candidate_empty_image_ratio,
        "target_leakage_pixels": target_leakage_pixels,
        "far_hard_candidate_count_mean": far_hard_candidate_count_mean,
        "target_protect_pixel_count_mean": target_protect_pixel_count_mean,
        "boundary_protect_pixel_count_mean": boundary_protect_pixel_count_mean,
        "ohem_overlap_fraction_mean": finite_mean([row["ohem_overlap_fraction"] for row in rows]),
        "far_hard_prob_mean": finite_mean([row["far_hard_prob_mean"] for row in rows]),
        "fail_reasons": fail_reasons,
        "outputs": {"per_image": str(out_dir / "per_image.csv")},
    }
    write_csv(out_dir / "per_image.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
