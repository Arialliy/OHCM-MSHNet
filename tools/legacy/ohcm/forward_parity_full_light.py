#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from utils import get_img_norm_cfg, seed_pytorch


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def load_checkpoint(net: Net, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)


def tensor_diff(a: torch.Tensor, b: torch.Tensor, h: int, w: int) -> dict:
    a = a[:, :, :h, :w].detach()
    b = b[:, :, :h, :w].detach()
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OHCM-light and OHCM-full forward outputs from the same checkpoint.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_list", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohcm_warm_epoch", type=int, default=60)
    parser.add_argument("--ohcm_tau", type=float, default=0.5)
    parser.add_argument("--ohcm_dilate_radius", type=int, default=5)
    parser.add_argument("--ohcm_topk", type=int, default=3)
    parser.add_argument("--ohcm_gamma_max", type=float, default=0.3)
    parser.add_argument("--ohcm_gamma_ramp_epochs", type=int, default=60)
    parser.add_argument("--ohcm_inhibition_start_epoch", type=int, default=None)
    parser.add_argument("--ohcm_margin_m", type=float, default=0.1)
    parser.add_argument("--ohcm_margin_delta", type=float, default=0.5)
    parser.add_argument("--ohcm_gt_area_median", type=float, default=20.0)
    parser.add_argument("--ohcm_mining_mode", default="cc_area_lc_ms")
    parser.add_argument("--lambda_clu", type=float, default=0.2)
    parser.add_argument("--lambda_sup", type=float, default=0.5)
    parser.add_argument("--lambda_margin", type=float, default=0.1)
    parser.add_argument("--lambda_proto", type=float, default=0.0)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset_name = args.train_dataset_name or args.dataset_name
    image_set = {line.strip() for line in Path(args.image_list).read_text(encoding="utf-8").splitlines() if line.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    cfg = vars(args).copy()
    cfg["ohcm_use_proto"] = False
    light = Net("OHCMMSHNet", mode="test", loss_cfg=cfg).to(device)
    load_checkpoint(light, Path(args.checkpoint), device)
    light.eval()

    full_cfg = vars(args).copy()
    full_cfg["ohcm_use_proto"] = True
    full_cfg["ohcm_force_no_proto"] = False
    full_cfg["lambda_proto"] = 0.0
    full = Net("OHCMMSHNetFull", mode="test", loss_cfg=full_cfg).to(device)
    load_checkpoint(full, Path(args.checkpoint), device)
    full.eval()

    rows = []
    with torch.no_grad():
        for img, _gt_mask, size, image_name in test_loader:
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            if name not in image_set:
                continue
            img = Variable(img).to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            light_out = light.export_logits_features(img)
            full_out = full.export_logits_features(img)

            row = {"image_name": name}
            for key, out_key in [
                ("z_t", "target_logit"),
                ("z_c", "clutter_logit"),
                ("z_final", "logit"),
            ]:
                diff = tensor_diff(light_out[out_key], full_out[out_key], h, w)
                row[f"{key}_max_abs"] = diff["max_abs"]
                row[f"{key}_mean_abs"] = diff["mean_abs"]

            p_light = torch.sigmoid(light_out["logit"][:, :, :h, :w])
            p_full = torch.sigmoid(full_out["logit"][:, :, :h, :w])
            prob_diff = (p_light - p_full).abs()
            row["p_final_max_abs"] = float(prob_diff.max().item())
            row["p_final_mean_abs"] = float(prob_diff.mean().item())
            mask_light = p_light > args.threshold
            mask_full = p_full > args.threshold
            row["binary_diff_pixels"] = int((mask_light != mask_full).sum().item())
            row["binary_pixels"] = int(mask_light.numel())
            row["binary_diff_fraction"] = float(row["binary_diff_pixels"] / max(1, row["binary_pixels"]))
            rows.append(row)
            if len(rows) >= args.max_images:
                break

    fieldnames = [
        "image_name",
        "z_t_max_abs",
        "z_t_mean_abs",
        "z_c_max_abs",
        "z_c_mean_abs",
        "z_final_max_abs",
        "z_final_mean_abs",
        "p_final_max_abs",
        "p_final_mean_abs",
        "binary_diff_pixels",
        "binary_pixels",
        "binary_diff_fraction",
    ]
    with (output_dir / "forward_parity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    max_values = {}
    for field in fieldnames[1:]:
        max_values[field] = max((float(row[field]) for row in rows), default=0.0)
    pass_parity = (
        max_values.get("z_t_max_abs", 1.0) <= 1e-7
        and max_values.get("z_c_max_abs", 1.0) <= 1e-7
        and max_values.get("z_final_max_abs", 1.0) <= 1e-7
        and max_values.get("binary_diff_pixels", 1.0) == 0.0
    )
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": args.dataset_name,
        "image_list": str(Path(args.image_list).resolve()),
        "num_images": len(rows),
        "threshold": args.threshold,
        "pass_parity": pass_parity,
        "max_values": max_values,
        "outputs": {"csv": str(output_dir / "forward_parity.csv")},
    }
    (output_dir / "forward_parity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
