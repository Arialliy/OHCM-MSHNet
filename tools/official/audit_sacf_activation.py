#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit SACF activation after a short sanity run.")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="SACFMSHNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=0)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--sacf_hidden_channels", type=int, default=16)
    parser.add_argument("--sacf_delta_max", type=float, default=1.0)
    parser.add_argument("--freeze_evidence", action="store_true", default=True)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def split_ids(args, dataset):
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values)) if values else 0.0


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

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    checkpoint_has_fusion_keys = any("fusion" in key for key in state_dict.keys())
    trainable_names = checkpoint.get("trainable_parameter_names", []) if isinstance(checkpoint, dict) else []
    optimizer_has_fusion_params = any("fusion" in name for name in trainable_names)

    rows = []
    diff_values = []
    changed_ratios = []
    gate_means = []
    gate_mins = []
    gate_maxs = []
    delta_abs_means = []
    entropy_means = []

    with torch.no_grad():
        for idx, (img, _gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            out = net(img, epoch=999, return_dict=True)
            final_prob = foreground_probability(out["final_logits"][:, :, :h, :w])
            base_prob = foreground_probability(out["base_logits"][:, :, :h, :w])
            gate = out["fusion_gate"][:, :, :h, :w]
            delta = out["fusion_delta"][:, :, :h, :w]
            weights = out["fusion_weights"][:, :, :h, :w]
            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1, keepdim=True)
            diff = (final_prob - base_prob).abs()
            changed = ((final_prob > args.threshold) != (base_prob > args.threshold)).float()

            row = {
                "image_id": name,
                "mean_abs_final_minus_base_prob": float(diff.mean().item()),
                "changed_pixel_ratio_at_0p5": float(changed.mean().item()),
                "fusion_gate_mean": float(gate.mean().item()),
                "fusion_gate_min": float(gate.min().item()),
                "fusion_gate_max": float(gate.max().item()),
                "fusion_delta_abs_mean": float(delta.abs().mean().item()),
                "fusion_weight_entropy_mean": float(entropy.mean().item()),
            }
            rows.append(row)
            diff_values.append(row["mean_abs_final_minus_base_prob"])
            changed_ratios.append(row["changed_pixel_ratio_at_0p5"])
            gate_means.append(row["fusion_gate_mean"])
            gate_mins.append(row["fusion_gate_min"])
            gate_maxs.append(row["fusion_gate_max"])
            delta_abs_means.append(row["fusion_delta_abs_mean"])
            entropy_means.append(row["fusion_weight_entropy_mean"])
            if (idx + 1) % 100 == 0:
                print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    summary = {
        "gate_pass": False,
        "fail_reasons": [],
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "num_images": len(rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "mean_abs_final_minus_base_prob": mean(diff_values),
        "changed_pixel_ratio_at_0p5": mean(changed_ratios),
        "fusion_gate_mean": mean(gate_means),
        "fusion_gate_min": min(gate_mins) if gate_mins else 0.0,
        "fusion_gate_max": max(gate_maxs) if gate_maxs else 0.0,
        "fusion_delta_abs_mean": mean(delta_abs_means),
        "fusion_weight_entropy_mean": mean(entropy_means),
        "checkpoint_has_fusion_keys": checkpoint_has_fusion_keys,
        "optimizer_has_fusion_params": optimizer_has_fusion_params,
        "outputs": {"per_image": str(output_dir / "per_image.csv")},
    }
    if summary["mean_abs_final_minus_base_prob"] <= 1e-4:
        summary["fail_reasons"].append("final_equals_base_identity_collapse")
    if summary["fusion_gate_mean"] <= 1e-3:
        summary["fail_reasons"].append("fusion_gate_not_active")
    if summary["fusion_delta_abs_mean"] <= 1e-5:
        summary["fail_reasons"].append("fusion_delta_not_active")
    if summary["changed_pixel_ratio_at_0p5"] <= 0.0:
        summary["fail_reasons"].append("no_threshold_changed_pixels")
    if not checkpoint_has_fusion_keys:
        summary["fail_reasons"].append("missing_fusion_keys")
    if not optimizer_has_fusion_params:
        summary["fail_reasons"].append("fusion_params_not_in_optimizer")
    summary["gate_pass"] = len(summary["fail_reasons"]) == 0

    write_csv(
        output_dir / "per_image.csv",
        rows,
        [
            "image_id",
            "mean_abs_final_minus_base_prob",
            "changed_pixel_ratio_at_0p5",
            "fusion_gate_mean",
            "fusion_gate_min",
            "fusion_gate_max",
            "fusion_delta_abs_mean",
            "fusion_weight_entropy_mean",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
