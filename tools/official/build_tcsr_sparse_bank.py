#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from net import Net  # noqa: E402
from probability import foreground_probability  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg  # noqa: E402


FORBIDDEN_SPLITS = {"test", "full", "hcval", "hctest", "hc-test", "blind", "external"}
IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only TCSR sparse reliability bank.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--tce_checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_high", type=float, default=0.50)
    parser.add_argument("--tce_low", type=float, default=0.35)
    parser.add_argument("--gap", type=float, default=0.15)
    parser.add_argument("--consensus_target", type=float, default=0.60)
    parser.add_argument("--far_bg_radius", type=int, default=7)
    parser.add_argument("--local_peak_kernel", type=int, default=7)
    parser.add_argument("--neg_dilate_radius", type=int, default=2)
    parser.add_argument("--target_protect_radius", type=int, default=2)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    return parser.parse_args()


def train_ids(dataset_dir: str, dataset_name: str) -> List[str]:
    root = Path(dataset_dir) / dataset_name
    candidates = [
        root / "img_idx" / f"train_{dataset_name}.txt",
        root / "img_idx" / "train.txt",
    ]
    for path in candidates:
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"Train split not found. Searched: {[str(path) for path in candidates]}")


def find_data_file(root: Path, subdir: str, image_id: str) -> Path:
    stem = Path(str(image_id)).stem
    for ext in IMAGE_EXTENSIONS:
        path = root / subdir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {subdir} file for image_id={image_id} under {root}")


def load_train_image_mask(dataset_root: Path, image_id: str, img_norm_cfg: Dict[str, float]):
    image_path = find_data_file(dataset_root, "images", image_id)
    mask_path = find_data_file(dataset_root, "masks", image_id)
    img = Image.open(image_path).convert("I")
    mask = Image.open(mask_path)
    img_np = Normalized(np.array(img, dtype=np.float32), img_norm_cfg)
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    if mask_np.ndim > 2:
        mask_np = mask_np[:, :, 0]
    h, w = img_np.shape
    img_pad = PadImg(img_np)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_pad[None, None, :, :])).float()
    gt = torch.from_numpy(np.ascontiguousarray((mask_np[:h, :w] > 0).astype(np.float32)))[None, :, :]
    return img_tensor, gt, h, w


def torch_load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_checkpoint(net: Net, checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def load_net(model_name: str, checkpoint_path: str | Path, device: torch.device, loss_cfg: Dict) -> Net:
    net = Net(model_name=model_name, mode="test", loss_cfg=loss_cfg).to(device)
    load_checkpoint(net, checkpoint_path, device)
    return net


def forward_prob(net: Net, img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    export = net.export_logits_features(img)
    logit = export["logit"][:, :, :h, :w]
    return foreground_probability(logit)[0, 0].detach().cpu().float()


def dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)


def build_bank_item(
    *,
    image_id: str,
    gt: torch.Tensor,
    p_ohem: torch.Tensor,
    p_tce_list: List[torch.Tensor],
    args: argparse.Namespace,
) -> Dict:
    gt4 = gt[None].float()
    p_ohem4 = p_ohem[None, None].float()
    p_stack = torch.stack(p_tce_list, dim=0).float()
    p_tce = p_stack.mean(dim=0)
    p_tce_std = p_stack.std(dim=0, unbiased=False)
    p_tce4 = p_tce[None, None]

    far_bg = dilate(gt4, args.far_bg_radius) == 0
    local_peak = p_ohem4 >= F.max_pool2d(
        p_ohem4,
        kernel_size=args.local_peak_kernel,
        stride=1,
        padding=args.local_peak_kernel // 2,
    )
    anchor_high = p_ohem4 >= args.anchor_high
    tce_low = p_tce4 <= args.tce_low
    gap = (p_ohem4 - p_tce4) >= args.gap
    neg_seed = far_bg & local_peak & anchor_high & tce_low & gap
    neg_weight = dilate(neg_seed.float(), args.neg_dilate_radius) * torch.clamp(p_ohem4 - p_tce4, 0.0, 1.0)

    target_support = dilate(gt4, args.target_protect_radius)
    consensus_target = ((p_tce4 >= args.consensus_target) & (target_support > 0)).float()
    protect_weight = torch.clamp(target_support + consensus_target, 0.0, 1.0)
    ref_prob = torch.maximum(gt4, torch.maximum(p_ohem4, p_tce4))

    raw_target_leakage = (neg_weight > 0) & (target_support > 0)
    neg_weight = neg_weight.masked_fill(raw_target_leakage, 0.0)
    raw_neg_protect_overlap = (neg_weight > 0) & (protect_weight > 0)
    neg_weight = neg_weight.masked_fill(raw_neg_protect_overlap, 0.0)

    target_leakage = (neg_weight > 0) & (target_support > 0)
    neg_protect_overlap = (neg_weight > 0) & (protect_weight > 0)

    neg_pixels = int((neg_weight > 0).sum().item())
    protect_pixels = int((protect_weight > 0).sum().item())
    meta = {
        "image_id": image_id,
        "height": int(gt.shape[-2]),
        "width": int(gt.shape[-1]),
        "neg_pixels": neg_pixels,
        "protect_pixels": protect_pixels,
        "target_leakage_pixels": int(target_leakage.sum().item()),
        "neg_protect_overlap_pixels": int(neg_protect_overlap.sum().item()),
        "raw_target_leakage_pixels": int(raw_target_leakage.sum().item()),
        "raw_neg_protect_overlap_pixels": int(raw_neg_protect_overlap.sum().item()),
        "neg_weight_sum": float(neg_weight.sum().item()),
        "protect_weight_sum": float(protect_weight.sum().item()),
        "p_ohem_mean": float(p_ohem.mean().item()),
        "p_tce_mean": float(p_tce.mean().item()),
        "p_tce_std_mean": float(p_tce_std.mean().item()),
    }
    return {
        "neg_weight": neg_weight[0].cpu(),
        "protect_weight": protect_weight[0].cpu(),
        "ref_prob": ref_prob[0].cpu(),
        "meta": meta,
    }


def main() -> None:
    args = parse_args()
    split = args.split.lower()
    if split != "train" or split in FORBIDDEN_SPLITS:
        raise ValueError("TCSR sparse bank may only be generated for the train split.")
    if len(args.tce_checkpoints) < 2:
        raise ValueError("--tce_checkpoints must include at least two checkpoints.")

    dataset_root = Path(args.dataset_dir) / args.dataset_name
    ids = train_ids(args.dataset_dir, args.dataset_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir.parent / "tcsr_bank_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)

    loss_cfg = vars(args).copy()
    net_by_path: Dict[str, Net] = {}
    all_checkpoint_paths = [args.ohem_checkpoint] + list(args.tce_checkpoints)
    for checkpoint_path in all_checkpoint_paths:
        resolved = str(Path(checkpoint_path).resolve())
        if resolved not in net_by_path:
            net_by_path[resolved] = load_net(args.model_name, checkpoint_path, device, loss_cfg)
    ohem_net = net_by_path[str(Path(args.ohem_checkpoint).resolve())]
    tce_nets = [net_by_path[str(Path(path).resolve())] for path in args.tce_checkpoints]

    items = []
    with torch.no_grad():
        for idx, image_id in enumerate(ids):
            img, gt, h, w = load_train_image_mask(dataset_root, image_id, img_norm_cfg)
            img = img.to(device)
            p_ohem = forward_prob(ohem_net, img, h, w)
            p_tce_list = [forward_prob(net, img, h, w) for net in tce_nets]
            item = build_bank_item(image_id=Path(image_id).stem, gt=gt, p_ohem=p_ohem, p_tce_list=p_tce_list, args=args)
            torch.save(item, output_dir / f"{Path(image_id).stem}.pt")
            meta = dict(item["meta"])
            meta["bank_path"] = str((output_dir / f"{Path(image_id).stem}.pt").resolve())
            items.append(meta)
            if (idx + 1) % 50 == 0 or idx + 1 == len(ids):
                print(f"Built TCSR sparse bank [{idx + 1}/{len(ids)}]", flush=True)

    summary = {
        "gate": "Gate-TCSR-A",
        "dataset": args.dataset_name,
        "split": "train",
        "train_only": True,
        "num_images": len(items),
        "train_images": len(ids),
        "num_images_with_neg": sum(1 for item in items if item["neg_pixels"] > 0),
        "neg_pixels_total": int(sum(item["neg_pixels"] for item in items)),
        "protect_pixels_total": int(sum(item["protect_pixels"] for item in items)),
        "target_leakage_pixels_total": int(sum(item["target_leakage_pixels"] for item in items)),
        "neg_protect_overlap_pixels_total": int(sum(item["neg_protect_overlap_pixels"] for item in items)),
        "raw_target_leakage_pixels_total": int(sum(item["raw_target_leakage_pixels"] for item in items)),
        "raw_neg_protect_overlap_pixels_total": int(sum(item["raw_neg_protect_overlap_pixels"] for item in items)),
        "output_dir": str(output_dir.resolve()),
        "ohem_checkpoint": str(Path(args.ohem_checkpoint).resolve()),
        "tce_checkpoints": [str(Path(path).resolve()) for path in args.tce_checkpoints],
        "thresholds": {
            "anchor_high": args.anchor_high,
            "tce_low": args.tce_low,
            "gap": args.gap,
            "consensus_target": args.consensus_target,
            "far_bg_radius": args.far_bg_radius,
            "local_peak_kernel": args.local_peak_kernel,
            "neg_dilate_radius": args.neg_dilate_radius,
            "target_protect_radius": args.target_protect_radius,
        },
        "items": items,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "num_images",
        "num_images_with_neg",
        "neg_pixels_total",
        "protect_pixels_total",
        "target_leakage_pixels_total",
        "neg_protect_overlap_pixels_total",
    )}, indent=2), flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
