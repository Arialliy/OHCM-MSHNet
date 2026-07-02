#!/usr/bin/env python3
"""Audit MSHNetOHEM cross-scale consensus before EACF training."""
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
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit MSHNetOHEM scale-consensus signal for EACF.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scale_threshold", type=float, default=0.5)
    parser.add_argument("--target_dilate_radius", type=int, default=5)
    parser.add_argument("--easy_bg_thr", type=float, default=0.05)
    parser.add_argument("--min_target_multiscale_support", type=float, default=0.90)
    parser.add_argument("--min_high_conf_bg_scale_var", type=float, default=1e-5)
    parser.add_argument("--min_single_scale_high_bg_ratio", type=float, default=1e-8)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
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


def split_ids(args, dataset) -> tuple[list[str], str]:
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


def torch_load_checkpoint(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(net: Net, checkpoint_path: str, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def upsample_scale_logits(masks: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
    upsampled = [
        F.interpolate(mask, size=size, mode="bilinear", align_corners=True)
        for mask in masks
    ]
    return torch.cat(upsampled, dim=1)


def update_iou(stats: dict, pred: np.ndarray, gt: np.ndarray) -> None:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    stats["inter"] += float(inter)
    stats["union"] += float(union)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, split_source = split_ids(args, dataset)
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)

    totals = {
        "target_pixels": 0.0,
        "target_multiscale_support_pixels": 0.0,
        "target_scale_var_sum": 0.0,
        "far_bg_pixels": 0.0,
        "far_bg_high_conf_pixels": 0.0,
        "far_bg_high_conf_scale_var_sum": 0.0,
        "single_scale_high_bg_pixels": 0.0,
        "easy_bg_pixels": 0.0,
        "easy_bg_scale_var_sum": 0.0,
    }
    base_iou = {"inter": 0.0, "union": 0.0}
    oracle_iou = {"inter": 0.0, "union": 0.0}
    rows = []

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            export = net.export_logits_features(img)
            masks = export.get("masks", [])
            if len(masks) != 4:
                raise RuntimeError(f"Expected four scale masks for {args.model_name}, got {len(masks)}")
            U = upsample_scale_logits(masks, size=(h, w))
            P = torch.sigmoid(U)[0].detach().cpu().numpy().astype(np.float32)
            base_prob = foreground_probability(export["logit"][:, :, :h, :w])[0, 0].detach().cpu().numpy().astype(np.float32)
            scale_var = P.var(axis=0)
            scale_max = P.max(axis=0)
            scale_min = P.min(axis=0)
            support_count = (P >= args.scale_threshold).sum(axis=0)
            far_bg = ~binary_dilate(gt, args.target_dilate_radius)
            high_conf_bg = far_bg & (scale_max >= args.scale_threshold)
            single_scale_high_bg = far_bg & (support_count == 1)
            easy_bg = far_bg & (scale_max < args.easy_bg_thr)
            target_support = gt & (support_count >= 2)

            totals["target_pixels"] += float(gt.sum())
            totals["target_multiscale_support_pixels"] += float(target_support.sum())
            totals["target_scale_var_sum"] += float(scale_var[gt].sum()) if gt.any() else 0.0
            totals["far_bg_pixels"] += float(far_bg.sum())
            totals["far_bg_high_conf_pixels"] += float(high_conf_bg.sum())
            totals["far_bg_high_conf_scale_var_sum"] += (
                float(scale_var[high_conf_bg].sum()) if high_conf_bg.any() else 0.0
            )
            totals["single_scale_high_bg_pixels"] += float(single_scale_high_bg.sum())
            totals["easy_bg_pixels"] += float(easy_bg.sum())
            totals["easy_bg_scale_var_sum"] += float(scale_var[easy_bg].sum()) if easy_bg.any() else 0.0

            base_pred = base_prob > args.threshold
            oracle_prob = np.where(gt, scale_max, scale_min)
            oracle_pred = oracle_prob > args.threshold
            update_iou(base_iou, base_pred, gt)
            update_iou(oracle_iou, oracle_pred, gt)

            rows.append(
                {
                    "image_id": name,
                    "target_pixels": int(gt.sum()),
                    "target_multiscale_support_ratio": safe_div(target_support.sum(), gt.sum()),
                    "target_scale_var_mean": float(scale_var[gt].mean()) if gt.any() else 0.0,
                    "far_bg_pixels": int(far_bg.sum()),
                    "far_bg_high_conf_pixels": int(high_conf_bg.sum()),
                    "far_bg_high_conf_scale_var_mean": (
                        float(scale_var[high_conf_bg].mean()) if high_conf_bg.any() else 0.0
                    ),
                    "single_scale_high_bg_ratio": safe_div(single_scale_high_bg.sum(), far_bg.sum()),
                    "easy_bg_scale_var_mean": float(scale_var[easy_bg].mean()) if easy_bg.any() else 0.0,
                }
            )
            if (idx + 1) % 100 == 0:
                print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    target_support_mean = safe_div(
        totals["target_multiscale_support_pixels"],
        totals["target_pixels"],
    )
    target_scale_var_mean = safe_div(totals["target_scale_var_sum"], totals["target_pixels"])
    high_conf_var_mean = safe_div(
        totals["far_bg_high_conf_scale_var_sum"],
        totals["far_bg_high_conf_pixels"],
    )
    single_scale_ratio = safe_div(totals["single_scale_high_bg_pixels"], totals["far_bg_pixels"])
    easy_bg_var_mean = safe_div(totals["easy_bg_scale_var_sum"], totals["easy_bg_pixels"])
    base_miou = safe_div(base_iou["inter"], base_iou["union"])
    oracle_miou = safe_div(oracle_iou["inter"], oracle_iou["union"])

    fail_reasons = []
    if target_support_mean < args.min_target_multiscale_support:
        fail_reasons.append("target_multiscale_support_too_low")
    if totals["far_bg_high_conf_pixels"] <= 0:
        fail_reasons.append("no_high_conf_background_scale_signal")
    if high_conf_var_mean <= args.min_high_conf_bg_scale_var:
        fail_reasons.append("high_conf_background_scale_var_too_low")
    if single_scale_ratio <= args.min_single_scale_high_bg_ratio:
        fail_reasons.append("single_scale_high_background_ratio_zero")
    if totals["easy_bg_pixels"] > 0 and easy_bg_var_mean > high_conf_var_mean:
        fail_reasons.append("easy_background_scale_variance_dominates")

    summary = {
        "gate_pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "model_name": args.model_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "num_images": len(rows),
        "target_scale_var_mean": target_scale_var_mean,
        "far_bg_high_conf_scale_var_mean": high_conf_var_mean,
        "easy_bg_scale_var_mean": easy_bg_var_mean,
        "single_scale_high_bg_ratio": single_scale_ratio,
        "multi_scale_target_support_mean": target_support_mean,
        "scale_weight_oracle_upper_bound_delta_miou": oracle_miou - base_miou,
        "base_mIoU": base_miou,
        "oracle_mIoU": oracle_miou,
        "counts": totals,
        "thresholds": {
            "threshold": args.threshold,
            "scale_threshold": args.scale_threshold,
            "target_dilate_radius": args.target_dilate_radius,
            "easy_bg_thr": args.easy_bg_thr,
            "min_target_multiscale_support": args.min_target_multiscale_support,
            "min_high_conf_bg_scale_var": args.min_high_conf_bg_scale_var,
            "min_single_scale_high_bg_ratio": args.min_single_scale_high_bg_ratio,
        },
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
        },
    }
    write_csv(
        output_dir / "per_image.csv",
        rows,
        [
            "image_id",
            "target_pixels",
            "target_multiscale_support_ratio",
            "target_scale_var_mean",
            "far_bg_pixels",
            "far_bg_high_conf_pixels",
            "far_bg_high_conf_scale_var_mean",
            "single_scale_high_bg_ratio",
            "easy_bg_scale_var_mean",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
