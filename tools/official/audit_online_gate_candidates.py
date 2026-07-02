#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import dilate_mask, far_background_mask, select_online_reliability_negatives
from net import Net
from probability import foreground_probability
from utils import Denormalization, Normalized, PadImg, get_img_norm_cfg


FORBIDDEN_SPLIT_TOKENS = ("test", "hc-test", "hctest", "blind", "external")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit train-only ERD online gate negative candidates.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--gate_far_radius", type=int, default=5)
    parser.add_argument("--gate_neg_q", type=float, default=0.01)
    parser.add_argument("--gate_neg_min_k", type=int, default=16)
    parser.add_argument("--gate_neg_max_k", type=int, default=512)
    parser.add_argument("--max_visuals", type=int, default=32)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def assert_train_only(path: Path, split: str):
    if split != "train":
        raise ValueError("online gate audit must use train split only: %s" % split)
    normalized = str(path).lower()
    if any(token in normalized for token in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("online gate audit split/source must be train-only: %s" % path)


def find_image(base: Path, image_id: str) -> Path:
    for ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = base / (image_id + ext)
        if path.exists():
            return path
    raise FileNotFoundError(str(base / image_id))


def load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr.astype(np.float32)


def to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.uint8)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def prob_to_rgb(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    red = (prob * 255).astype(np.uint8)
    blue = ((1.0 - prob) * 255).astype(np.uint8)
    green = (np.minimum(prob, 1.0 - prob) * 2.0 * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask = mask.astype(bool)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def torch_load_checkpoint(checkpoint_path: str, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_dir) / args.dataset_name
    split_path = dataset_root / "img_idx" / ("train_%s.txt" % args.dataset_name)
    assert_train_only(split_path, args.split)

    image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not image_ids:
        raise ValueError("empty train split: %s" % split_path)

    out_dir = Path(args.out_dir)
    visual_dir = out_dir / "candidate_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    net = Net(
        model_name=args.model_name,
        mode="test",
        loss_cfg=vars(args),
    ).to(device)
    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    net.eval()

    rows = []
    with torch.no_grad():
        for idx, image_id in enumerate(image_ids):
            image_path = find_image(dataset_root / "images", image_id)
            mask_path = find_image(dataset_root / "masks", image_id)
            raw_img = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
            gt = load_gray(mask_path)
            if raw_img.shape != gt.shape:
                raise ValueError("shape mismatch for %s: image=%s mask=%s" % (image_id, raw_img.shape, gt.shape))

            h, w = raw_img.shape
            norm_img = Normalized(raw_img, img_norm_cfg)
            img_pad = PadImg(norm_img)
            input_tensor = torch.from_numpy(np.ascontiguousarray(img_pad[None, None, :, :])).float().to(device)

            export = net.export_logits_features(input_tensor)
            logit = export["logit"][:, :, :h, :w]
            prob = foreground_probability(logit)
            gt_tensor = torch.from_numpy(np.ascontiguousarray(gt[None, None, :, :])).float().to(device)

            neg_mask, counts = select_online_reliability_negatives(
                evidence_logit=logit,
                gt_mask=gt_tensor,
                far_radius=args.gate_far_radius,
                q=args.gate_neg_q,
                min_k=args.gate_neg_min_k,
                max_k=args.gate_neg_max_k,
            )
            target_dilate = dilate_mask(gt_tensor, args.gate_far_radius)
            far_bg = far_background_mask(gt_tensor, args.gate_far_radius)

            neg_count = int(neg_mask.sum().item())
            leakage = int((neg_mask * target_dilate).sum().item())
            far_bg_valid_ratio = float(far_bg.mean().item())
            prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
            neg_np = neg_mask[0, 0].detach().cpu().numpy() > 0
            target_np = gt > 0
            easy_bg_np = (far_bg[0, 0].detach().cpu().numpy() > 0) & (~neg_np)
            row = {
                "image_id": image_id,
                "neg_pixels": neg_count,
                "target_leakage_neg_pixels": leakage,
                "far_bg_valid_ratio": far_bg_valid_ratio,
                "evidence_prob_on_neg_mean": float(prob_np[neg_np].mean()) if neg_np.any() else 0.0,
                "evidence_prob_on_target_mean": float(prob_np[target_np].mean()) if target_np.any() else 0.0,
                "evidence_prob_on_easy_bg_mean": float(prob_np[easy_bg_np].mean()) if easy_bg_np.any() else 0.0,
                "selected_k": int(counts[0]) if counts else 0,
            }
            rows.append(row)

            if idx < args.max_visuals:
                raw_u8 = np.clip(Denormalization(norm_img, img_norm_cfg), 0, 255).astype(np.uint8)
                panels = [
                    np.stack([raw_u8, raw_u8, raw_u8], axis=-1),
                    overlay_mask(raw_u8, target_np, (0, 255, 0)),
                    prob_to_rgb(prob_np),
                    overlay_mask(raw_u8, neg_np, (255, 0, 0)),
                ]
                Image.fromarray(np.concatenate(panels, axis=1)).save(visual_dir / ("%s.png" % image_id))

            if (idx + 1) % 100 == 0:
                print("Audited [%d/%d]" % (idx + 1, len(image_ids)), flush=True)

    write_csv(out_dir / "per_image.csv", rows)
    neg_pixels = np.asarray([row["neg_pixels"] for row in rows], dtype=np.float64)
    leakages = np.asarray([row["target_leakage_neg_pixels"] for row in rows], dtype=np.float64)
    no_neg = int((neg_pixels <= 0).sum())
    neg_mean = float(neg_pixels.mean())
    neg_min = int(neg_pixels.min()) if len(neg_pixels) else 0
    images_without_neg_ratio = float(no_neg) / float(len(rows))
    easy_bg_mean = float(np.mean([row["evidence_prob_on_easy_bg_mean"] for row in rows]))
    neg_prob_mean = float(np.mean([row["evidence_prob_on_neg_mean"] for row in rows]))
    gate_pass = (
        images_without_neg_ratio <= 0.02
        and int(leakages.sum()) == 0
        and neg_min >= max(1, int(args.gate_neg_min_k / 2))
        and neg_mean >= float(args.gate_neg_min_k)
        and neg_prob_mean > easy_bg_mean
    )
    summary = {
        "gate_pass": bool(gate_pass),
        "source_split": args.split,
        "num_images": len(rows),
        "images_without_neg": no_neg,
        "images_without_neg_ratio": images_without_neg_ratio,
        "neg_pixels_mean": neg_mean,
        "neg_pixels_min": neg_min,
        "target_leakage_neg_pixels": int(leakages.sum()),
        "far_bg_valid_ratio_mean": float(np.mean([row["far_bg_valid_ratio"] for row in rows])),
        "evidence_prob_on_neg_mean": neg_prob_mean,
        "evidence_prob_on_target_mean": float(np.mean([row["evidence_prob_on_target_mean"] for row in rows])),
        "evidence_prob_on_easy_bg_mean": easy_bg_mean,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_name": args.model_name,
        "gate_far_radius": int(args.gate_far_radius),
        "gate_neg_q": float(args.gate_neg_q),
        "gate_neg_min_k": int(args.gate_neg_min_k),
        "gate_neg_max_k": int(args.gate_neg_max_k),
        "candidate_visuals": str(visual_dir),
        "decision": "GO_ONLINE_ERD_SEED42" if gate_pass else "NO_GO_ONLINE_AUDIT_STOP",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
