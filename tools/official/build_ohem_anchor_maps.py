#!/usr/bin/env python3
"""Build train-split MSHNetOHEM anchor maps for APF candidate audit."""
from __future__ import annotations

import argparse
import hashlib
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
    parser = argparse.ArgumentParser(description="Build frozen OHEM anchor maps for APF audit.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_dilation_radius", type=int, default=5)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    return checkpoint


def split_ids(args, dataset) -> tuple[list[str], str]:
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_hash = sha256_file(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, split_source = split_ids(args, dataset)
    dataset.test_list = image_ids
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)

    rows = []
    fail_reasons = []
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            export = net.export_logits_features(img)
            logit = export["logit"][:, :, :h, :w]
            prob = foreground_probability(logit)[0, 0].detach().cpu().numpy().astype(np.float32)
            logit_np = logit[0, 0].detach().cpu().numpy().astype(np.float32)

            if prob.shape != gt.shape or logit_np.shape != gt.shape:
                fail_reasons.append(f"shape_mismatch:{name}")
                continue
            if not np.isfinite(prob).all() or not np.isfinite(logit_np).all():
                fail_reasons.append(f"nonfinite_anchor:{name}")
                continue

            target_protect = binary_dilate(gt, args.target_dilation_radius)
            far_bg = ~target_protect
            metadata = {
                "dataset_name": args.dataset_name,
                "split_name": args.split,
                "image_id": name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "threshold": args.threshold,
                "target_dilation_radius": args.target_dilation_radius,
                "height": int(h),
                "width": int(w),
            }
            np.savez_compressed(
                out_dir / f"{name}.npz",
                prob_ohem=prob.astype(np.float16),
                logit_ohem=logit_np.astype(np.float16),
                mask_ohem_05=(prob > args.threshold),
                far_bg_mask=far_bg,
                target_protect_mask=target_protect,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            rows.append(
                {
                    "image_id": name,
                    "height": int(h),
                    "width": int(w),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                }
            )
            if (idx + 1) % 100 == 0:
                print(f"Built [{idx + 1}/{len(loader)}]", flush=True)

    if len(rows) != len(image_ids):
        fail_reasons.append("generated_count_mismatch")
    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "model_name": args.model_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "target_dilation_radius": args.target_dilation_radius,
        "num_images": len(image_ids),
        "num_written": len(rows),
        "gate_pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
