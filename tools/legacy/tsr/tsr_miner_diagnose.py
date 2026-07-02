#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss import TargetScaleRegionLoss  # noqa: E402
from net import Net  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg, seed_pytorch  # noqa: E402


def read_image_and_mask(dataset_dir: Path, name: str, img_norm_cfg):
    image_path = dataset_dir / "images" / f"{name}.png"
    mask_path = dataset_dir / "masks" / f"{name}.png"
    if not image_path.exists():
        image_path = dataset_dir / "images" / f"{name}.bmp"
    if not mask_path.exists():
        mask_path = dataset_dir / "masks" / f"{name}.bmp"

    raw_img = np.array(Image.open(image_path).convert("I"), dtype=np.float32)
    mask = np.array(Image.open(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = raw_img.shape

    img = Normalized(raw_img, img_norm_cfg)
    mask = mask / 255.0
    img_pad = PadImg(img)
    mask_pad = PadImg(mask)

    # Padding is not real background; mark it as protected so the miner cannot
    # select windows outside the original image.
    if mask_pad.shape[0] > h:
        mask_pad[h:, :] = 1.0
    if mask_pad.shape[1] > w:
        mask_pad[:, w:] = 1.0

    img_tensor = torch.from_numpy(np.ascontiguousarray(img_pad[np.newaxis, np.newaxis, :])).float()
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_pad[np.newaxis, np.newaxis, :])).float()
    return image_path, raw_img, img_tensor, mask_tensor, (h, w)


def image_to_uint8(raw: np.ndarray) -> np.ndarray:
    arr = raw.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = float(arr.max() + 1.0)
        lo = float(arr.min())
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protected_overlap_ratio(miner: TargetScaleRegionLoss, mask_tensor: torch.Tensor, regions: list[dict]) -> float:
    if not regions:
        return 0.0
    safe = miner._safe_background(mask_tensor, mask_tensor.shape[-2:])[0, 0].detach().cpu().numpy()
    protected = safe <= 0
    bad = 0
    for region in regions:
        y0, y1, x0, x1 = region["box"]
        if protected[y0:y1, x0:x1].any():
            bad += 1
    return float(bad) / float(len(regions))


def visualize_candidates(rows: list[dict], output_dir: Path, vis_count: int, seed: int) -> None:
    if vis_count <= 0 or not rows:
        return
    rng = random.Random(seed)
    selected = rows if len(rows) <= vis_count else rng.sample(rows, vis_count)
    vis_dir = output_dir / "candidate_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, row in enumerate(selected):
        image_path = Path(row["image_path"])
        raw = np.array(Image.open(image_path).convert("I"), dtype=np.float32)
        canvas = Image.fromarray(image_to_uint8(raw)).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 40, 40), width=2)
        label = f"s={row['scale']} h={float(row['score']):.3f} w={float(row['weight']):.2f}"
        draw.text((max(0, x0), max(0, y0 - 11)), label, fill=(255, 255, 0), font=font)
        canvas.save(vis_dir / f"{idx:04d}_{row['image']}_cand{row['rank']}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose TSR-OHEM target-scale miner with a frozen MSHNetOHEM checkpoint.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_scales", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.3)
    parser.add_argument("--weight_temp", type=float, default=0.2)
    parser.add_argument("--dilate_radius", type=int, default=0)
    parser.add_argument("--no_consensus", action="store_true")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--vis_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    if args.image_list:
        names = [line.strip() for line in Path(args.image_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        list_path = dataset_dir / "img_idx" / f"{args.split}_{args.dataset_name}.txt"
        names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_images > 0:
        names = names[: args.max_images]

    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    net = Net(model_name="MSHNetOHEM", mode="test", loss_cfg={"mshnet_in_channels": 1}).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()

    miner = TargetScaleRegionLoss(
        target_scales=args.target_scales,
        beta=args.beta,
        topk=args.topk,
        nms_iou=args.nms_iou,
        weight_temp=args.weight_temp,
        dilation_radius=args.dilate_radius,
        use_consensus=not args.no_consensus,
    ).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    image_rows = []
    scale_counts = {}
    overlap_ratios = []

    with torch.no_grad():
        for idx, name in enumerate(names):
            image_path, _raw, img_tensor, mask_tensor, (h, w) = read_image_and_mask(dataset_dir, name, img_norm_cfg)
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            export = net.export_logits_features(img_tensor)
            hard_regions, stats = miner.mine_hard_regions(export["logit"], export["masks"], mask_tensor)
            regions = hard_regions[0]
            overlap = protected_overlap_ratio(miner, mask_tensor.detach().cpu(), regions)
            overlap_ratios.append(overlap)
            image_rows.append({
                "image": name,
                "candidate_count": len(regions),
                "empty": int(len(regions) == 0),
                "protected_overlap_ratio": overlap,
                "hard_region_score_mean": float(stats["hard_region_score_mean"].detach().cpu()),
                "hard_region_uncertainty_mean": float(stats["hard_region_uncertainty_mean"].detach().cpu()),
                "hard_region_scale_mean": float(stats["hard_region_scale_mean"].detach().cpu()),
            })
            for rank, region in enumerate(regions, start=1):
                y0, y1, x0, x1 = region["box"]
                if y1 > h or x1 > w:
                    continue
                scale_counts[str(region["scale"])] = scale_counts.get(str(region["scale"]), 0) + 1
                candidate_rows.append({
                    "image": name,
                    "image_path": str(image_path),
                    "rank": rank,
                    "scale": region["scale"],
                    "score": region["score"],
                    "uncertainty": region["uncertainty"],
                    "weight": region["weight"],
                    "center_y": region["y"],
                    "center_x": region["x"],
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                })
            if (idx + 1) % 100 == 0:
                print(f"Diagnosed [{idx + 1}/{len(names)}]", flush=True)

    write_csv(
        output_dir / "tsr_miner_candidates.csv",
        candidate_rows,
        ["image", "image_path", "rank", "scale", "score", "uncertainty", "weight", "center_y", "center_x", "y0", "y1", "x0", "x1"],
    )
    write_csv(
        output_dir / "tsr_miner_per_image.csv",
        image_rows,
        ["image", "candidate_count", "empty", "protected_overlap_ratio", "hard_region_score_mean", "hard_region_uncertainty_mean", "hard_region_scale_mean"],
    )
    visualize_candidates(candidate_rows, output_dir, args.vis_count, args.seed)

    candidate_counts = [row["candidate_count"] for row in image_rows]
    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "images": len(image_rows),
        "target_scales": args.target_scales,
        "topk": args.topk,
        "mean_candidates_per_image": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "min_candidates_per_image": int(np.min(candidate_counts)) if candidate_counts else 0,
        "max_candidates_per_image": int(np.max(candidate_counts)) if candidate_counts else 0,
        "empty_ratio": float(np.mean([row["empty"] for row in image_rows])) if image_rows else 1.0,
        "protected_overlap_ratio": float(np.mean(overlap_ratios)) if overlap_ratios else 0.0,
        "super_large_region_ratio": 0.0,
        "scale_counts": scale_counts,
        "candidate_csv": str(output_dir / "tsr_miner_candidates.csv"),
        "per_image_csv": str(output_dir / "tsr_miner_per_image.csv"),
        "visualization_dir": str(output_dir / "candidate_vis"),
    }
    (output_dir / "tsr_miner_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
