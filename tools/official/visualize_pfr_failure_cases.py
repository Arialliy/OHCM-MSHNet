#!/usr/bin/env python3
"""Visualize PFR evidence/final/residual failure cases.

This is a failure-analysis utility only. It does not train or tune PFR.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Visualize PFR failure-analysis cases.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head_audit_csv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--target_dilate_radius", type=int, default=3)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--pfr_beta", type=float, default=0.5)
    parser.add_argument("--pfr_feature_channels", type=int, default=16)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))[0, 0].numpy() > 0


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


def read_ranked_ids(path: str | None) -> list[str]:
    if not path:
        return []
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(
        key=lambda row: (
            -int(float(row.get("residual_new_fp_components", 0) or 0)),
            -int(float(row.get("residual_lost_target_pixels", 0) or 0)),
            -int(float(row.get("residual_boundary_excess_pixels", 0) or 0)),
        )
    )
    return [row["image_name"] for row in rows if row.get("image_name")]


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.repeat(normalize_for_display(base)[..., None], 3, axis=2)
    rgb[mask.astype(bool)] = color
    return rgb


def save_case(out_dir: Path, name: str, image: np.ndarray, gt: np.ndarray, evidence_prob: np.ndarray, final_prob: np.ndarray, delta: np.ndarray, threshold: float, target_dilate_radius: int) -> None:
    evidence_mask = evidence_prob > threshold
    final_mask = final_prob > threshold
    bg = ~gt
    boundary = binary_dilate(gt, target_dilate_radius) & (~gt)
    new_fp = final_mask & (~evidence_mask) & bg
    removed_fp = evidence_mask & (~final_mask) & bg
    lost_target = evidence_mask & (~final_mask) & gt
    boundary_excess = final_mask & (~evidence_mask) & boundary

    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(case_dir / "image.png", normalize_for_display(image), cmap="gray")
    plt.imsave(case_dir / "gt_mask.png", gt.astype(np.float32), cmap="gray")
    plt.imsave(case_dir / "evidence_prob.png", evidence_prob, cmap="magma", vmin=0.0, vmax=1.0)
    plt.imsave(case_dir / "final_prob.png", final_prob, cmap="magma", vmin=0.0, vmax=1.0)
    plt.imsave(case_dir / "residual_delta.png", delta, cmap="coolwarm")
    plt.imsave(case_dir / "new_fp_overlay.png", overlay_mask(image, new_fp, (1.0, 0.0, 0.0)))
    plt.imsave(case_dir / "removed_fp_overlay.png", overlay_mask(image, removed_fp, (0.0, 0.7, 1.0)))
    plt.imsave(case_dir / "lost_target_overlay.png", overlay_mask(image, lost_target, (1.0, 1.0, 0.0)))
    plt.imsave(case_dir / "boundary_excess_overlay.png", overlay_mask(image, boundary_excess, (1.0, 0.4, 0.0)))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    ranked_ids = read_ranked_ids(args.head_audit_csv)
    selected = ranked_ids[: args.max_images] if ranked_ids else list(dataset.test_list[: args.max_images])
    dataset.test_list = selected
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name="PFRMSHNet", mode="test", loss_cfg=vars(args)).to(device)
    load_checkpoint(net, args.checkpoint, device)

    with torch.no_grad():
        for img, gt_mask, size, image_name in loader:
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            export = net.export_logits_features(img)
            evidence_logits = export["target_logit"][:, :, :h, :w]
            final_logits = export["logit"][:, :, :h, :w]
            delta = export["delta_logit"][0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            evidence_prob = foreground_probability(evidence_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            final_prob = foreground_probability(final_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            image = img[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            save_case(out_dir, name, image, gt, evidence_prob, final_prob, delta, args.threshold, args.target_dilate_radius)


if __name__ == "__main__":
    main()
