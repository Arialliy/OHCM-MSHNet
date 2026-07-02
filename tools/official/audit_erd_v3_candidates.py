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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import ERDMSHNetV3Loss, binary_dilate
from net import Net
from probability import foreground_probability
from utils import Normalized, PadImg, get_img_norm_cfg


FORBIDDEN_SPLIT_TOKENS = ("test", "hc-test", "hctest", "blind", "external")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit train-only ERD-v3 TP-CS online candidates.")
    parser.add_argument("--model_names", default=["ERDMSHNetV3"], nargs="+")
    parser.add_argument("--pretrained_ohem", required=True)
    parser.add_argument("--dataset_names", default=["NUDT-SIRST"], nargs="+")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--erd_far_radius", type=int, default=7)
    parser.add_argument("--erd_target_protect_radius", type=int, default=2)
    parser.add_argument("--erd_neg_topk_ratio", type=float, default=0.01)
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def assert_train_only(path: Path, split: str):
    if split != "train":
        raise ValueError("ERD-v3 candidate audit must use train split only: %s" % split)
    normalized = str(path).lower()
    if any(token in normalized for token in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("ERD-v3 candidate audit source must be train-only: %s" % path)


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


def torch_load_checkpoint(checkpoint_path: str, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_checkpoint(net: Net, checkpoint_path: str, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    net.load_state_dict(checkpoint_state_dict(checkpoint))
    return checkpoint


def select_ohem_reference_mask(prob: torch.Tensor, target: torch.Tensor, ratio: float) -> torch.Tensor:
    with torch.no_grad():
        target = target.float()
        valid = target <= 0
        out = torch.zeros_like(prob, dtype=torch.bool)
        flat_prob = prob.reshape(prob.shape[0], -1)
        flat_valid = valid.reshape(valid.shape[0], -1)
        flat_out = out.reshape(out.shape[0], -1)
        for b in range(prob.shape[0]):
            idx = torch.nonzero(flat_valid[b], as_tuple=False).flatten()
            if idx.numel() < 1:
                continue
            k = max(1, int(math.floor(float(idx.numel()) * float(ratio))))
            k = min(k, int(idx.numel()))
            top = torch.topk(flat_prob[b, idx], k=k, largest=True).indices
            flat_out[b, idx[top]] = True
        return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else 0.0


def main():
    args = parse_args()
    if args.model_names != ["ERDMSHNetV3"]:
        raise ValueError("This audit is only for --model_names ERDMSHNetV3.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_probe = ERDMSHNetV3Loss(
        mshnet_warm_epoch=args.mshnet_warm_epoch,
        ohem_ratio=args.ohem_ratio,
        far_radius=args.erd_far_radius,
        target_protect_radius=args.erd_target_protect_radius,
        neg_topk_ratio=args.erd_neg_topk_ratio,
    )

    rows = []
    for dataset_name in args.dataset_names:
        dataset_root = Path(args.dataset_dir) / dataset_name
        split_path = dataset_root / "img_idx" / ("train_%s.txt" % dataset_name)
        assert_train_only(split_path, args.split)
        image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.max_images > 0:
            image_ids = image_ids[: args.max_images]
        if not image_ids:
            raise ValueError("empty train split: %s" % split_path)

        img_norm_cfg = get_img_norm_cfg(dataset_name, args.dataset_dir)
        net = Net(
            model_name="MSHNetOHEM",
            mode="test",
            loss_cfg={
                "mshnet_warm_epoch": args.mshnet_warm_epoch,
                "mshnet_in_channels": args.mshnet_in_channels,
                "ohem_ratio": args.ohem_ratio,
            },
        ).to(device)
        load_checkpoint(net, args.pretrained_ohem, device)
        net.eval()

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
                img_tensor = torch.from_numpy(np.ascontiguousarray(img_pad[None, None, :, :])).float().to(device)
                gt_tensor = torch.from_numpy(np.ascontiguousarray(gt[None, None, :, :])).float().to(device)

                export = net.export_logits_features(img_tensor)
                evidence_logits = export["logit"][:, :, :h, :w]
                evidence_prob = foreground_probability(evidence_logits)
                neg_mask, counts = loss_probe.select_online_negatives(evidence_logits, gt_tensor)
                ohem_mask = select_ohem_reference_mask(evidence_prob.detach(), gt_tensor, args.ohem_ratio)
                target_dilate = binary_dilate(gt_tensor, args.erd_far_radius)
                far_bg = (1.0 - target_dilate).bool()

                cand_pixels = int(neg_mask.sum().item())
                leakage = int((neg_mask.float() * target_dilate).sum().item())
                far_selected = int((neg_mask & far_bg).sum().item())
                far_ratio = float(far_selected) / float(max(1, cand_pixels))
                ohem_inter = int((neg_mask & ohem_mask).sum().item())
                ohem_union = int((neg_mask | ohem_mask).sum().item())
                cand_prob = float(evidence_prob[neg_mask].mean().item()) if cand_pixels > 0 else 0.0

                rows.append({
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "candidate_pixels": cand_pixels,
                    "candidate_pixels_selected_k": int(counts[0]) if counts else 0,
                    "candidate_target_leakage_pixels": leakage,
                    "candidate_far_bg_ratio": far_ratio,
                    "candidate_high_evidence_mean": cand_prob,
                    "candidate_ohem_overlap_fraction": float(ohem_inter) / float(max(1, ohem_union)),
                    "target_dilate_pixels": int(target_dilate.sum().item()),
                    "far_bg_pixels": int(far_bg.sum().item()),
                })

                if (idx + 1) % 100 == 0:
                    print("Audited %s [%d/%d]" % (dataset_name, idx + 1, len(image_ids)), flush=True)

    write_csv(out_dir / "per_image.csv", rows)

    cand = np.asarray([row["candidate_pixels"] for row in rows], dtype=np.float64)
    leak = int(sum(row["candidate_target_leakage_pixels"] for row in rows))
    no_candidate = int((cand <= 0).sum())
    no_candidate_ratio = float(no_candidate) / float(max(1, len(rows)))
    candidate_far_bg_ratio = finite_mean(row["candidate_far_bg_ratio"] for row in rows)
    gate_pass = (
        no_candidate_ratio <= 0.05
        and leak == 0
        and float(cand.mean()) > 0.0
        and candidate_far_bg_ratio >= 0.99
    )

    summary = {
        "gate_pass": bool(gate_pass),
        "method": "ERDMSHNetV3_TPCS",
        "split": args.split,
        "model_names": args.model_names,
        "num_images": len(rows),
        "candidate_pixels_mean": float(cand.mean()) if len(cand) else 0.0,
        "candidate_pixels_min": int(cand.min()) if len(cand) else 0,
        "images_without_candidate": no_candidate,
        "images_without_candidate_ratio": no_candidate_ratio,
        "candidate_target_leakage_pixels": leak,
        "candidate_far_bg_ratio": candidate_far_bg_ratio,
        "candidate_high_evidence_mean": finite_mean(row["candidate_high_evidence_mean"] for row in rows),
        "candidate_ohem_overlap_fraction": finite_mean(row["candidate_ohem_overlap_fraction"] for row in rows),
        "candidate_tce_removed_fp_alignment": None,
        "dataset_names": args.dataset_names,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "pretrained_ohem": str(Path(args.pretrained_ohem).resolve()),
        "erd_far_radius": int(args.erd_far_radius),
        "erd_neg_topk_ratio": float(args.erd_neg_topk_ratio),
        "outputs": {
            "per_image": str(out_dir / "per_image.csv"),
            "summary": str(out_dir / "summary.json"),
            "gate_pass": str(out_dir / "gate_pass.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "gate_pass.json").write_text(
        json.dumps(
            {
                "gate_pass": bool(gate_pass),
                "method": "ERDMSHNetV3_TPCS",
                "split": args.split,
                "summary": str(out_dir / "summary.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
