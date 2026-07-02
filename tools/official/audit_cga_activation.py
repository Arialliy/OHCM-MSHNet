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
from skimage import measure
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CGA activation after a 1-epoch sanity run.")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=0)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--cga_num_scale_bins", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=128)
    parser.add_argument("--easy_drift_limit", type=float, default=0.01)
    parser.add_argument("--hard_update_min", type=float, default=1e-5)
    parser.add_argument("--fp_delta_limit", type=int, default=10)
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


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_net_checkpoint(net: Net, path: str, device):
    checkpoint = torch_load_checkpoint(path, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    return checkpoint


def false_positive_components(prob: torch.Tensor, target: torch.Tensor, threshold: float) -> int:
    pred = (prob[0, 0].detach().cpu().numpy() > threshold).astype(np.uint8)
    gt = (target[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    labels = measure.label(pred, connectivity=2)
    count = 0
    for region in measure.regionprops(labels):
        coords = region.coords
        if gt[coords[:, 0], coords[:, 1]].sum() == 0:
            count += 1
    return count


def finite_float(value) -> bool:
    return math.isfinite(float(value))


def mean(values):
    return float(np.mean(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_gradient_probe(net: Net, loader: DataLoader, device, epoch: int):
    net.train()
    net.zero_grad(set_to_none=True)
    img, gt_mask, _size, _image_name = next(iter(loader))
    img = img.to(device)
    gt_mask = gt_mask.to(device)
    out = net(img, epoch=epoch, return_dict=True)
    loss_out = net.loss(out, gt_mask, epoch=epoch)
    loss_out["total"].backward()

    sq_sum = 0.0
    for name, param in net.named_parameters():
        if "geometry_heads" in name and param.grad is not None:
            sq_sum += float(param.grad.detach().pow(2).sum().cpu())
    grad_norm = math.sqrt(sq_sum)
    return {
        "geometry_head_grad_norm": grad_norm,
        "center_loss": float(loss_out["center"].detach().cpu()),
        "scale_loss": float(loss_out["scale"].detach().cpu()),
        "local_peak_bg_loss": float(loss_out["peak_bg"].detach().cpu()),
        "peak_count": float(loss_out["peak_count"].detach().cpu()),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, split_source = split_ids(args, dataset)
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    loss_cfg = {
        "mshnet_warm_epoch": args.mshnet_warm_epoch,
        "mshnet_in_channels": args.mshnet_in_channels,
        "cga_num_scale_bins": args.cga_num_scale_bins,
    }
    cga_net = Net(model_name="CGAMSHNet", mode="train", loss_cfg=loss_cfg).to(device)
    cga_checkpoint = load_net_checkpoint(cga_net, args.checkpoint, device)

    base_net = Net(model_name="MSHNetOHEM", mode="test", loss_cfg=loss_cfg).to(device)
    load_net_checkpoint(base_net, args.ohem_checkpoint, device)

    grad_probe = compute_gradient_probe(cga_net, loader, device, epoch=999)
    cga_net.eval()
    base_net.eval()

    state_dict = checkpoint_state_dict(cga_checkpoint)
    checkpoint_has_geometry_keys = any("geometry_heads" in key for key in state_dict.keys())
    trainable_names = cga_checkpoint.get("trainable_parameter_names", []) if isinstance(cga_checkpoint, dict) else []
    optimizer_has_geometry_params = any("geometry_heads" in name for name in trainable_names)

    rows = []
    easy_diffs = []
    hard_diffs = []
    all_diffs = []
    cga_fp_total = 0
    base_fp_total = 0

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            img = img.to(device)
            gt_mask = gt_mask[:, :, :h, :w].to(device)

            cga_out = cga_net(img, epoch=999, return_dict=True)
            cga_prob = foreground_probability(cga_out["final_logits"][:, :, :h, :w])
            base_prob = base_net(img, epoch=999)[:, :, :h, :w]

            diff = (cga_prob - base_prob).abs()
            easy_target = (gt_mask > 0.5) & (base_prob > 0.85)
            easy_bg = (gt_mask < 0.5) & (base_prob < 0.05)
            easy = easy_target | easy_bg
            hard = ~easy
            easy_diff = float(diff[easy].mean().item()) if easy.any() else 0.0
            hard_diff = float(diff[hard].mean().item()) if hard.any() else 0.0
            all_diff = float(diff.mean().item())
            base_fp = false_positive_components(base_prob, gt_mask, args.threshold)
            cga_fp = false_positive_components(cga_prob, gt_mask, args.threshold)

            rows.append(
                {
                    "image_id": name,
                    "mean_abs_final_minus_ohem_prob": all_diff,
                    "easy_anchor_drift": easy_diff,
                    "hard_region_update": hard_diff,
                    "base_fp_components": base_fp,
                    "cga_fp_components": cga_fp,
                }
            )
            easy_diffs.append(easy_diff)
            hard_diffs.append(hard_diff)
            all_diffs.append(all_diff)
            base_fp_total += base_fp
            cga_fp_total += cga_fp
            if (idx + 1) % 50 == 0:
                print(f"Audited [{idx + 1}/{len(loader)}]", flush=True)

    summary = {
        "gate_pass": False,
        "fail_reasons": [],
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "num_images": len(rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "ohem_checkpoint": str(Path(args.ohem_checkpoint).resolve()),
        "epoch": cga_checkpoint.get("epoch") if isinstance(cga_checkpoint, dict) else None,
        "geometry_head_grad_norm": grad_probe["geometry_head_grad_norm"],
        "center_loss": grad_probe["center_loss"],
        "scale_loss": grad_probe["scale_loss"],
        "local_peak_bg_loss": grad_probe["local_peak_bg_loss"],
        "peak_count": grad_probe["peak_count"],
        "checkpoint_has_geometry_keys": checkpoint_has_geometry_keys,
        "optimizer_has_geometry_params": optimizer_has_geometry_params,
        "mean_abs_final_minus_ohem_prob": mean(all_diffs),
        "final_easy_anchor_drift": mean(easy_diffs),
        "final_hard_region_update": mean(hard_diffs),
        "base_fp_components": int(base_fp_total),
        "cga_fp_components": int(cga_fp_total),
        "fp_component_delta": int(cga_fp_total - base_fp_total),
        "thresholds": {
            "easy_drift_limit": args.easy_drift_limit,
            "hard_update_min": args.hard_update_min,
            "fp_delta_limit": args.fp_delta_limit,
        },
        "outputs": {"per_image": str(output_dir / "per_image.csv")},
    }

    if summary["geometry_head_grad_norm"] <= 0.0:
        summary["fail_reasons"].append("geometry_head_grad_norm_not_positive")
    for key in ("center_loss", "scale_loss", "local_peak_bg_loss"):
        if not finite_float(summary[key]):
            summary["fail_reasons"].append(f"{key}_not_finite")
    if not checkpoint_has_geometry_keys:
        summary["fail_reasons"].append("missing_geometry_head_keys")
    if not optimizer_has_geometry_params:
        summary["fail_reasons"].append("geometry_params_not_in_optimizer")
    if summary["final_easy_anchor_drift"] >= args.easy_drift_limit:
        summary["fail_reasons"].append("easy_anchor_drift_too_large")
    if summary["final_hard_region_update"] <= args.hard_update_min:
        summary["fail_reasons"].append("hard_region_update_zero")
    if summary["fp_component_delta"] > args.fp_delta_limit:
        summary["fail_reasons"].append("fp_component_explosion")

    summary["gate_pass"] = len(summary["fail_reasons"]) == 0

    write_csv(
        output_dir / "per_image.csv",
        rows,
        [
            "image_id",
            "mean_abs_final_minus_ohem_prob",
            "easy_anchor_drift",
            "hard_region_update",
            "base_fp_components",
            "cga_fp_components",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
