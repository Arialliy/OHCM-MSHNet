#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


FORBIDDEN_SPLIT_TOKENS = ("hctest", "hc-test", "test", "blind", "external")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit ERD-v2 seed42 HC-Val failure modes.")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--ohem_checkpoint", required=True)
    parser.add_argument("--erd_checkpoint", required=True)
    parser.add_argument("--ohem_model_name", default="MSHNetOHEM")
    parser.add_argument("--erd_model_name", default="ERDMSHNet")
    parser.add_argument("--split", default="hcval")
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_dilate_radius", type=int, default=3)
    parser.add_argument("--far_dilate_radius", type=int, default=5)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--erd_rho", type=float, default=0.25)
    parser.add_argument("--erd_gamma_max", type=float, default=1.0)
    parser.add_argument("--erd_gate_start_epoch", type=int, default=20)
    parser.add_argument("--erd_gate_ramp_epochs", type=int, default=30)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    k = 2 * int(radius) + 1
    return (F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=int(radius)) > 0).float()


def safe_mean(array: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not mask.any():
        return float("nan")
    return float(array[mask].mean())


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def region_iou(region, target_mask: np.ndarray) -> float:
    region_mask = np.zeros(target_mask.shape, dtype=bool)
    region_mask[region.coords[:, 0], region.coords[:, 1]] = True
    intersection = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(intersection, union)


def match_components(pred_mask: np.ndarray, gt_mask: np.ndarray, distance_threshold: float = 3.0):
    pred_regions = connected_regions(pred_mask)
    gt_regions = connected_regions(gt_mask)
    used_pred = set()
    matched_targets = 0
    for gt_region in gt_regions:
        gt_centroid = np.asarray(gt_region.centroid)
        for pred_idx, pred_region in enumerate(pred_regions):
            if pred_idx in used_pred:
                continue
            pred_centroid = np.asarray(pred_region.centroid)
            if np.linalg.norm(pred_centroid - gt_centroid) < distance_threshold:
                used_pred.add(pred_idx)
                matched_targets += 1
                break

    fp_components = 0
    for pred_idx, pred_region in enumerate(pred_regions):
        if pred_idx in used_pred:
            continue
        if region_iou(pred_region, gt_mask) <= 0:
            fp_components += 1
    return matched_targets, len(gt_regions), fp_components


def image_detection_metrics(prob: np.ndarray, gt: np.ndarray, threshold: float):
    pred = prob > threshold
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched, target_components, fp_components = match_components(pred, gt)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "miou": safe_div(inter, union),
        "pd": safe_div(matched, target_components),
        "precision": precision,
        "fa_ppm": safe_div(fp, pred.size) * 1_000_000.0,
        "fp_components": float(fp_components),
        "num_gt_components": float(target_components),
    }


def load_checkpoint(net: Net, checkpoint_path: str, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def resolve_image_list(args) -> Path:
    if args.image_list:
        path = Path(args.image_list)
    elif args.split == "hcval":
        path = Path("/home/AAAI/OHCM-MSHNet/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hcval_%s.txt" % args.dataset_name)
    elif args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / ("train_%s.txt" % args.dataset_name)
    else:
        path = Path(args.split)

    lowered = str(path).lower()
    if args.split != "hcval" and any(token in lowered for token in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("This audit cannot use test/blind/external sources: %s" % path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def export_prob_aux(net: Net, img: torch.Tensor, h: int, w: int):
    export = net.export_logits_features(img)
    logit = export["logit"][:, :, :h, :w]
    prob = foreground_probability(logit)[0, 0].detach().cpu().numpy().astype(np.float32)
    aux = {}
    if "gate" in export:
        aux["gate"] = export["gate"][0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
    if "target_logit" in export:
        aux["evidence_prob"] = foreground_probability(export["target_logit"][:, :, :h, :w])[0, 0].detach().cpu().numpy().astype(np.float32)
    return prob, aux


def compute_audit_row(image_id, ohem_prob, erd_prob, gt, erd_aux, threshold, target_dilate_radius, far_dilate_radius):
    target = gt.astype(bool)
    y = torch.from_numpy(target.astype(np.float32))[None, None]
    target_dilate = binary_dilate(y, target_dilate_radius)[0, 0].numpy() > 0
    far_bg = (1.0 - binary_dilate(y, far_dilate_radius))[0, 0].numpy() > 0
    high_evidence_far_bg = far_bg & (ohem_prob > threshold)
    ohem_pred = ohem_prob > threshold
    erd_pred = erd_prob > threshold
    removed_by_erd = ohem_pred & (~erd_pred)
    removed_gt = removed_by_erd & target_dilate
    removed_fp = removed_by_erd & far_bg

    ohem_metrics = image_detection_metrics(ohem_prob, target, threshold)
    erd_metrics = image_detection_metrics(erd_prob, target, threshold)

    target_core_recall_ohem = safe_div(np.logical_and(ohem_pred, target).sum(), target.sum())
    target_core_recall_erd = safe_div(np.logical_and(erd_pred, target).sum(), target.sum())
    target_dilate_recall_ohem = safe_div(np.logical_and(ohem_pred, target_dilate).sum(), target_dilate.sum())
    target_dilate_recall_erd = safe_div(np.logical_and(erd_pred, target_dilate).sum(), target_dilate.sum())

    row = {
        "image_id": image_id,
        "split": "hcval",
        "num_gt_components": ohem_metrics["num_gt_components"],
        "ohem_miou": ohem_metrics["miou"],
        "erd_miou": erd_metrics["miou"],
        "delta_miou": erd_metrics["miou"] - ohem_metrics["miou"],
        "ohem_pd": ohem_metrics["pd"],
        "erd_pd": erd_metrics["pd"],
        "delta_pd": erd_metrics["pd"] - ohem_metrics["pd"],
        "ohem_precision": ohem_metrics["precision"],
        "erd_precision": erd_metrics["precision"],
        "delta_precision": erd_metrics["precision"] - ohem_metrics["precision"],
        "ohem_fa_ppm": ohem_metrics["fa_ppm"],
        "erd_fa_ppm": erd_metrics["fa_ppm"],
        "delta_fa_ppm": erd_metrics["fa_ppm"] - ohem_metrics["fa_ppm"],
        "ohem_fp_components": ohem_metrics["fp_components"],
        "erd_fp_components": erd_metrics["fp_components"],
        "delta_fp_components": erd_metrics["fp_components"] - ohem_metrics["fp_components"],
        "target_core_recall_ohem": target_core_recall_ohem,
        "target_core_recall_erd": target_core_recall_erd,
        "target_core_recall_delta": target_core_recall_erd - target_core_recall_ohem,
        "target_dilate_recall_ohem": target_dilate_recall_ohem,
        "target_dilate_recall_erd": target_dilate_recall_erd,
        "target_dilate_recall_delta": target_dilate_recall_erd - target_dilate_recall_ohem,
        "target_prob_mass_ohem": safe_mean(ohem_prob, target_dilate),
        "target_prob_mass_erd": safe_mean(erd_prob, target_dilate),
        "target_prob_mass_delta": safe_mean(erd_prob, target_dilate) - safe_mean(ohem_prob, target_dilate),
        "far_bg_prob_mass_ohem": safe_mean(ohem_prob, far_bg),
        "far_bg_prob_mass_erd": safe_mean(erd_prob, far_bg),
        "far_bg_prob_mass_delta": safe_mean(erd_prob, far_bg) - safe_mean(ohem_prob, far_bg),
        "removed_fp_pixels": int(removed_fp.sum()),
        "removed_gt_pixels": int(removed_gt.sum()),
        "suppression_selectivity": safe_div(removed_fp.sum(), removed_fp.sum() + removed_gt.sum()),
        "gate_mean_target": float("nan"),
        "gate_mean_target_dilate": float("nan"),
        "gate_mean_far_bg": float("nan"),
        "gate_mean_high_evidence_far_bg": float("nan"),
    }
    gate = erd_aux.get("gate")
    if gate is not None:
        row.update({
            "gate_mean_target": safe_mean(gate, target),
            "gate_mean_target_dilate": safe_mean(gate, target_dilate),
            "gate_mean_far_bg": safe_mean(gate, far_bg),
            "gate_mean_high_evidence_far_bg": safe_mean(gate, high_evidence_far_bg),
        })
    return row


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows, key: str) -> float:
    values = [float(row[key]) for row in rows if not np.isnan(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_list_path = resolve_image_list(args)
    image_ids = [line.strip() for line in image_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    test_set.test_list = image_ids
    loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    ohem_net = Net(model_name=args.ohem_model_name, mode="test", loss_cfg=vars(args)).to(device)
    erd_net = Net(model_name=args.erd_model_name, mode="test", loss_cfg=vars(args)).to(device)
    load_checkpoint(ohem_net, args.ohem_checkpoint, device)
    load_checkpoint(erd_net, args.erd_checkpoint, device)

    rows = []
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            ohem_prob, _ = export_prob_aux(ohem_net, img, h, w)
            erd_prob, erd_aux = export_prob_aux(erd_net, img, h, w)
            rows.append(
                compute_audit_row(
                    name,
                    ohem_prob,
                    erd_prob,
                    gt,
                    erd_aux,
                    args.threshold,
                    args.target_dilate_radius,
                    args.far_dilate_radius,
                )
            )
            if (idx + 1) % 20 == 0:
                print("Audited [%d/%d]" % (idx + 1, len(loader)), flush=True)

    write_csv(out_dir / "per_image_erd_v2_failure_audit.csv", rows)
    summary = {
        "num_images": len(rows),
        "split": args.split,
        "image_list": str(image_list_path),
        "mean_delta_miou": finite_mean(rows, "delta_miou"),
        "mean_delta_precision": finite_mean(rows, "delta_precision"),
        "mean_delta_fa_ppm": finite_mean(rows, "delta_fa_ppm"),
        "mean_target_core_recall_delta": finite_mean(rows, "target_core_recall_delta"),
        "mean_target_dilate_recall_delta": finite_mean(rows, "target_dilate_recall_delta"),
        "mean_target_prob_mass_delta": finite_mean(rows, "target_prob_mass_delta"),
        "mean_far_bg_prob_mass_delta": finite_mean(rows, "far_bg_prob_mass_delta"),
        "total_removed_fp_pixels": int(sum(int(row["removed_fp_pixels"]) for row in rows)),
        "total_removed_gt_pixels": int(sum(int(row["removed_gt_pixels"]) for row in rows)),
        "mean_suppression_selectivity": finite_mean(rows, "suppression_selectivity"),
        "num_images_target_dilate_drop_gt_2pct": int(sum(float(row["target_dilate_recall_delta"]) < -0.02 for row in rows)),
        "target_dilate_drop_gt_2pct_ratio": safe_div(
            sum(float(row["target_dilate_recall_delta"]) < -0.02 for row in rows),
            len(rows),
        ),
        "mean_gate_target": finite_mean(rows, "gate_mean_target"),
        "mean_gate_target_dilate": finite_mean(rows, "gate_mean_target_dilate"),
        "mean_gate_far_bg": finite_mean(rows, "gate_mean_far_bg"),
        "mean_gate_high_evidence_far_bg": finite_mean(rows, "gate_mean_high_evidence_far_bg"),
    }
    target_damage = (
        summary["mean_target_dilate_recall_delta"] < -0.01
        or summary["target_dilate_drop_gt_2pct_ratio"] > 0.20
        or summary["mean_suppression_selectivity"] < 0.80
        or summary["mean_gate_target_dilate"] < 0.90
    )
    summary["failure_mode"] = "target_damage_or_support_shrinkage" if target_damage else "calibration_or_threshold_shift"
    summary["target_damage_flag"] = bool(target_damage)
    (out_dir / "summary_erd_v2_failure_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
