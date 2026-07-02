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

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from tools.official.check_direct_export_parity import finalize_stats, init_stats, update_stats
from utils import get_img_norm_cfg


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


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
    return checkpoint if isinstance(checkpoint, dict) else {}


def load_ecdv_evidence_checkpoint(net, checkpoint_path, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("model.evidence_net."):
            mapped[key] = value
        elif key.startswith("model."):
            mapped["model.evidence_net." + key[len("model."):]] = value
    current_keys = set(net.state_dict().keys())
    mapped = {key: value for key, value in mapped.items() if key in current_keys}
    missing, unexpected = net.load_state_dict(mapped, strict=False)
    missing_evidence = [key for key in missing if key.startswith("model.evidence_net.")]
    if missing_evidence:
        raise RuntimeError(f"Missing ECDV evidence keys: {missing_evidence[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected ECDV evidence keys: {unexpected[:8]}")
    net.eval()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ECDV Gate-A beta=0 equivalence audit against MSHNetOHEM.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--max_prob_diff", type=float, default=1e-6)
    parser.add_argument("--max_miou_diff", type=float, default=0.0)
    parser.add_argument("--max_pd_diff", type=float, default=0.0)
    parser.add_argument("--max_fa_ppm_diff", type=float, default=0.0)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ecdv_beta_max", type=float, default=0.1)
    parser.add_argument("--ecdv_hidden_channels", type=int, default=32)
    parser.add_argument("--ecdv_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--ecdv_contrast_kernel", type=int, default=9)
    parser.add_argument("--ecdv_highpass_kernel", type=int, default=9)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset_name = args.train_dataset_name or args.dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    loader = DataLoader(TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg), num_workers=1, batch_size=1, shuffle=False)

    ohem = Net("MSHNetOHEM", mode="test", loss_cfg=vars(args)).to(device)
    ecdv_cfg = dict(vars(args))
    ecdv_cfg["ecdv_eval_beta"] = 0.0
    ecdv = Net("ECDVMSHNet", mode="test", loss_cfg=ecdv_cfg).to(device)
    checkpoint = load_ohem_checkpoint(ohem, args.checkpoint, device)
    load_ecdv_evidence_checkpoint(ecdv, args.checkpoint, device)

    ohem_stats, ecdv_stats = init_stats(), init_stats()
    rows = []
    max_prob_diff, total_mask_diff, max_suppression = 0.0, 0, 0.0
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            if args.max_images and idx >= args.max_images:
                break
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            ohem_logit = ohem.export_logits_features(img)["logit"]
            ecdv_export = ecdv.export_logits_features(img)
            ecdv_logit = ecdv_export["logit"]
            suppression = ecdv_export["suppression_map"]
            ohem_prob = foreground_probability(ohem_logit)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            ecdv_prob = foreground_probability(ecdv_logit)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            ohem_mask, ecdv_mask = ohem_prob > args.threshold, ecdv_prob > args.threshold
            prob_diff = float(np.max(np.abs(ohem_prob - ecdv_prob)))
            mask_diff = int(np.not_equal(ohem_mask, ecdv_mask).sum())
            suppression_max = float(suppression[:, :, :h, :w].abs().max().detach().cpu())
            max_prob_diff = max(max_prob_diff, prob_diff)
            max_suppression = max(max_suppression, suppression_max)
            total_mask_diff += mask_diff
            update_stats(ohem_stats, ohem_prob, gt, args.threshold)
            update_stats(ecdv_stats, ecdv_prob, gt, args.threshold)
            rows.append({"image_name": name, "max_prob_diff": prob_diff, "mask_diff_pixels": mask_diff, "suppression_max": suppression_max})
            if (idx + 1) % 100 == 0:
                print(f"Gate-ECDV-A checked [{idx + 1}/{len(loader)}]", flush=True)

    ohem_metrics, ecdv_metrics = finalize_stats(ohem_stats), finalize_stats(ecdv_stats)
    metric_diff = {key: abs(ohem_metrics[key] - ecdv_metrics[key]) for key in ("mIoU", "Pd", "FA_ppm")}
    checks = {
        "prob_max_diff": max_prob_diff < args.max_prob_diff,
        "binary_mask_diff": total_mask_diff == 0,
        "suppression_zero": max_suppression == 0.0,
        "full_miou_diff": metric_diff["mIoU"] <= args.max_miou_diff,
        "pd_diff": metric_diff["Pd"] <= args.max_pd_diff,
        "fa_ppm_diff": metric_diff["FA_ppm"] <= args.max_fa_ppm_diff,
    }
    summary = {
        "gate": "ECDV_Gate_A_beta0_equivalence",
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "num_images": len(rows),
        "max_prob_diff": max_prob_diff,
        "mask_diff_pixels": total_mask_diff,
        "max_suppression": max_suppression,
        "ohem_metrics": ohem_metrics,
        "ecdv_metrics": ecdv_metrics,
        "metric_abs_diff": metric_diff,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "outputs": {"per_image": str(output_dir / "per_image.csv"), "summary": str(output_dir / "summary.json")},
    }
    write_csv(output_dir / "per_image.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
