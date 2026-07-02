#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import measure
from torch.autograd import Variable
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import Denormalization, get_img_norm_cfg

MSHNET_NAMES = ("MSHNet", "MSHNetFocal", "MSHNetOHEM", "MSHNetTopKNeg", "MSHNetSPSOHEM")


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.uint8)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def prob_to_rgb(prob):
    prob = np.clip(prob, 0.0, 1.0)
    red = (prob * 255).astype(np.uint8)
    blue = ((1.0 - prob) * 255).astype(np.uint8)
    green = (np.minimum(prob, 1.0 - prob) * 2.0 * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def overlay_mask(gray, mask, color):
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask = mask.astype(bool)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def connected_regions(mask):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def region_iou(region, target_mask):
    region_mask = np.zeros(target_mask.shape, dtype=bool)
    region_mask[region.coords[:, 0], region.coords[:, 1]] = True
    intersection = np.logical_and(region_mask, target_mask).sum()
    union = np.logical_or(region_mask, target_mask).sum()
    return safe_div(intersection, union)


def match_components(pred_mask, gt_mask, distance_threshold=3.0):
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


def image_metrics(image_name, prob, logit, pred_mask, gt_mask, threshold):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fa = safe_div(fp, pred.size)
    target_area = int(gt.sum())
    bg = ~gt

    return {
        "image_name": image_name,
        "threshold": threshold,
        "IoU": safe_div(inter, union),
        "nIoU": safe_div(inter, union),
        "Pd": safe_div(matched_targets, target_components),
        "FA": fa,
        "FA_ppm": fa * 1_000_000.0,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FP_components": fp_components,
        "target_components": target_components,
        "target_area": target_area,
        "mean_prob_target": float(prob[gt].mean()) if target_area else 0.0,
        "mean_prob_bg": float(prob[bg].mean()) if bg.any() else 0.0,
        "mean_logit_target": float(logit[gt].mean()) if target_area else 0.0,
        "mean_logit_bg": float(logit[bg].mean()) if bg.any() else 0.0,
    }


def threshold_update(stats, prob, gt_mask, threshold):
    pred = prob > threshold
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    matched_targets, target_components, fp_components = match_components(pred, gt)

    item = stats[threshold]
    item["inter"] += float(inter)
    item["union"] += float(union)
    item["tp"] += float(tp)
    item["fp"] += float(fp)
    item["fn"] += float(fn)
    item["pixels"] += float(pred.size)
    item["niou_sum"] += safe_div(inter, union)
    item["count"] += 1
    item["matched_targets"] += float(matched_targets)
    item["target_components"] += float(target_components)
    item["fp_components"] += float(fp_components)


def threshold_rows(stats):
    rows = []
    for threshold in sorted(stats):
        item = stats[threshold]
        precision = safe_div(item["tp"], item["tp"] + item["fp"])
        recall = safe_div(item["tp"], item["tp"] + item["fn"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        fa = safe_div(item["fp"], item["pixels"])
        rows.append(
            {
                "threshold": threshold,
                "mIoU": safe_div(item["inter"], item["union"]),
                "nIoU": safe_div(item["niou_sum"], item["count"]),
                "Pd": safe_div(item["matched_targets"], item["target_components"]),
                "FA": fa,
                "FA_ppm": fa * 1_000_000.0,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "FP_components": item["fp_components"],
            }
        )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def torch_load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_checkpoint(net, checkpoint_path, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    return checkpoint


def resolve_mshnet_head(model_name: str, requested: str, allow_legacy_output0: bool = False) -> str:
    if model_name in MSHNET_NAMES and requested == "output0" and not allow_legacy_output0:
        raise ValueError(
            "Refusing to export MSHNet output0 head in non-legacy mode. "
            "Use the final head for direct/export parity, or pass "
            "--allow_legacy_mshnet_output0 only for reproducing old invalid exports."
        )
    if requested != "auto":
        return requested
    return "final" if model_name in MSHNET_NAMES else "final"


def export_forward(net, img, args):
    head = resolve_mshnet_head(args.model_name, args.mshnet_export_head, args.allow_legacy_mshnet_output0)
    if args.model_name in MSHNET_NAMES and head == "output0":
        masks, logit, feature = net.model(img, False, return_feature=True)
        return {
            "logit": logit,
            "target_logit": logit,
            "clutter_logit": torch.zeros_like(logit),
            "feature": feature,
            "masks": masks,
        }
    return net.export_logits_features(img)


def run_direct_export_parity_gate(args, output_dir: Path) -> None:
    parity_dir = output_dir / "direct_export_parity"
    cmd = [
        sys.executable,
        "tools/official/check_direct_export_parity.py",
        "--dataset_dir",
        args.dataset_dir,
        "--dataset_name",
        args.dataset_name,
        "--train_dataset_name",
        args.train_dataset_name or args.dataset_name,
        "--model_name",
        args.model_name,
        "--checkpoint",
        args.checkpoint,
        "--exports_dir",
        str(output_dir),
        "--output_dir",
        str(parity_dir),
        "--threshold",
        str(args.threshold),
        "--max_prob_diff",
        str(args.parity_max_prob_diff),
        "--max_miou_diff",
        str(args.parity_max_miou_diff),
        "--max_pd_diff",
        str(args.parity_max_pd_diff),
        "--max_fa_ppm_diff",
        str(args.parity_max_fa_ppm_diff),
        "--mshnet_warm_epoch",
        str(args.mshnet_warm_epoch),
        "--mshnet_in_channels",
        str(args.mshnet_in_channels),
    ]
    if args.model_name == "ECDVMSHNet":
        cmd.extend(
            [
                "--ecdv_beta_max",
                str(args.ecdv_beta_max),
                "--ecdv_eval_beta",
                str(args.ecdv_eval_beta if args.ecdv_eval_beta is not None else args.ecdv_beta_max),
                "--ecdv_hidden_channels",
                str(args.ecdv_hidden_channels),
                "--ecdv_evidence_threshold",
                str(args.ecdv_evidence_threshold),
                "--ecdv_contrast_kernel",
                str(args.ecdv_contrast_kernel),
                "--ecdv_highpass_kernel",
                str(args.ecdv_highpass_kernel),
            ]
        )
    if args.model_name == "MSCVMSHNet":
        cmd.extend(
            [
                "--mscv_beta_max",
                str(args.mscv_beta_max),
                "--mscv_eval_beta",
                str(args.mscv_eval_beta if args.mscv_eval_beta is not None else args.mscv_beta_max),
                "--mscv_hidden_channels",
                str(args.mscv_hidden_channels),
                "--mscv_evidence_threshold",
                str(args.mscv_evidence_threshold),
                "--mscv_contrast_kernel",
                str(args.mscv_contrast_kernel),
                "--mscv_far_radius",
                str(args.mscv_far_radius),
                "--mscv_candidate_prob_thr",
                str(args.mscv_candidate_prob_thr),
                "--mscv_candidate_std_thr",
                str(args.mscv_candidate_std_thr),
                "--mscv_nonflat_thr",
                str(args.mscv_nonflat_thr),
            ]
        )
    if args.image_list:
        cmd.extend(["--image_list", args.image_list])
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Export MSHNet-family predictions and per-image metrics.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--model_name", default="MSHNet")
    parser.add_argument("--mshnet_export_head", default="auto", choices=["auto", "output0", "final"])
    parser.add_argument("--allow_legacy_mshnet_output0", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ecdv_beta_max", type=float, default=0.1)
    parser.add_argument("--ecdv_eval_beta", type=float, default=None)
    parser.add_argument("--ecdv_hidden_channels", type=int, default=32)
    parser.add_argument("--ecdv_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--ecdv_detach_verifier_input", action="store_true", default=True)
    parser.add_argument("--ecdv_no_detach_verifier_input", dest="ecdv_detach_verifier_input", action="store_false")
    parser.add_argument("--ecdv_contrast_kernel", type=int, default=9)
    parser.add_argument("--ecdv_highpass_kernel", type=int, default=9)
    parser.add_argument("--mscv_beta_max", type=float, default=0.1)
    parser.add_argument("--mscv_eval_beta", type=float, default=None)
    parser.add_argument("--mscv_hidden_channels", type=int, default=32)
    parser.add_argument("--mscv_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--mscv_detach_verifier_input", action="store_true", default=True)
    parser.add_argument("--mscv_no_detach_verifier_input", dest="mscv_detach_verifier_input", action="store_false")
    parser.add_argument("--mscv_contrast_kernel", type=int, default=9)
    parser.add_argument("--mscv_far_radius", type=int, default=7)
    parser.add_argument("--mscv_candidate_prob_thr", type=float, default=0.2)
    parser.add_argument("--mscv_candidate_std_thr", type=float, default=0.05)
    parser.add_argument("--mscv_nonflat_thr", type=float, default=0.05)
    parser.add_argument("--ohcm_warm_epoch", type=int, default=60)
    parser.add_argument("--ohcm_gamma_max", type=float, default=0.3)
    parser.add_argument("--ohcm_gamma_ramp_epochs", type=int, default=60)
    parser.add_argument("--ohcm_inhibition_start_epoch", type=int, default=None)
    parser.add_argument("--ohcm_tau", type=float, default=0.5)
    parser.add_argument("--ohcm_dilate_radius", type=int, default=5)
    parser.add_argument("--ohcm_topk", type=int, default=3)
    parser.add_argument("--ohcm_margin_m", type=float, default=0.1)
    parser.add_argument("--ohcm_margin_delta", type=float, default=0.5)
    parser.add_argument("--ohcm_gt_area_median", type=float, default=20.0)
    parser.add_argument("--ohcm_mining_mode", default="cc_area_lc_ms")
    parser.add_argument("--ohcm_force_no_proto", action="store_true")
    parser.add_argument("--ohcm_proto_start_epoch", type=int, default=80)
    parser.add_argument("--ohcm_proto_momentum", type=float, default=0.9)
    parser.add_argument("--ohcm_proto_temperature", type=float, default=0.1)
    parser.add_argument("--lambda_clu", type=float, default=0.2)
    parser.add_argument("--lambda_sup", type=float, default=0.5)
    parser.add_argument("--lambda_margin", type=float, default=0.1)
    parser.add_argument("--lambda_proto", type=float, default=0.05)
    parser.add_argument("--skip_direct_export_parity_gate", action="store_true")
    parser.add_argument("--parity_max_prob_diff", type=float, default=1e-6)
    parser.add_argument("--parity_max_miou_diff", type=float, default=1e-6)
    parser.add_argument("--parity_max_pd_diff", type=float, default=1e-6)
    parser.add_argument("--parity_max_fa_ppm_diff", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for subdir in [
        "masks",
        "probs",
        "logits",
        "target_logits",
        "clutter_logits",
        "features",
        "vis",
        "evidence_logits",
        "risk_logits",
        "risk_probs",
        "suppression_maps",
        "final_logits",
        "decoy_debug",
        "validity_logits",
        "validity_probs",
        "p_mean",
        "p_std",
        "p_min",
        "p_max",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    if args.image_list:
        test_set.test_list = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=args.batch_size, shuffle=False)

    net = Net(
        model_name=args.model_name,
        mode="test",
        loss_cfg=vars(args),
    ).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)
    net.eval()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    threshold_stats = {
        threshold: {
            "inter": 0.0,
            "union": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pixels": 0.0,
            "niou_sum": 0.0,
            "count": 0,
            "matched_targets": 0.0,
            "target_components": 0.0,
            "fp_components": 0.0,
        }
        for threshold in thresholds
    }
    per_image_rows = []

    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(test_loader):
            img = Variable(img).to(device)
            gt_mask = gt_mask.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            export = export_forward(net, img, args)
            logit_tensor = export["logit"]
            target_logit_tensor = export["target_logit"]
            clutter_logit_tensor = export["clutter_logit"]
            feature_tensor = export["feature"]
            evidence_logit_tensor = export.get("evidence_logit", target_logit_tensor)
            risk_logit_tensor = export.get("risk_logit")
            risk_prob_tensor = export.get("risk_prob")
            validity_logit_tensor = export.get("validity_logit")
            validity_prob_tensor = export.get("validity_prob")
            suppression_tensor = export.get("suppression_map")
            final_logit_tensor = export.get("final_logit", logit_tensor)
            p_mean_tensor = export.get("p_mean")
            p_std_tensor = export.get("p_std")
            p_min_tensor = export.get("p_min")
            p_max_tensor = export.get("p_max")

            logit = logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            target_logit = target_logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            clutter_logit = clutter_logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            evidence_logit = evidence_logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            final_logit = final_logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            prob = foreground_probability(logit_tensor)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            gt = gt_mask[0, 0, :h, :w].detach().cpu().numpy() > 0
            pred_mask = prob > args.threshold
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)

            np.save(output_dir / "probs" / f"{name}.npy", prob)
            np.save(output_dir / "logits" / f"{name}.npy", logit)
            np.save(output_dir / "target_logits" / f"{name}.npy", target_logit)
            np.save(output_dir / "clutter_logits" / f"{name}.npy", clutter_logit)
            np.save(output_dir / "evidence_logits" / f"{name}.npy", evidence_logit)
            np.save(output_dir / "final_logits" / f"{name}.npy", final_logit)
            decoy_debug = {
                "evidence_logit": evidence_logit,
                "final_logit": final_logit,
            }
            if risk_logit_tensor is not None:
                risk_logit = risk_logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                np.save(output_dir / "risk_logits" / f"{name}.npy", risk_logit)
                decoy_debug["risk_logit"] = risk_logit
            if risk_prob_tensor is not None:
                risk_prob = risk_prob_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                np.save(output_dir / "risk_probs" / f"{name}.npy", risk_prob)
                decoy_debug["risk_prob"] = risk_prob
            if validity_logit_tensor is not None:
                validity_logit = validity_logit_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                np.save(output_dir / "validity_logits" / f"{name}.npy", validity_logit)
                decoy_debug["validity_logit"] = validity_logit
            if validity_prob_tensor is not None:
                validity_prob = validity_prob_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                np.save(output_dir / "validity_probs" / f"{name}.npy", validity_prob)
                decoy_debug["validity_prob"] = validity_prob
            if suppression_tensor is not None:
                suppression = suppression_tensor[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                np.save(output_dir / "suppression_maps" / f"{name}.npy", suppression)
                decoy_debug["suppression_map"] = suppression
            for tensor_name, tensor_value, subdir in [
                ("p_mean", p_mean_tensor, "p_mean"),
                ("p_std", p_std_tensor, "p_std"),
                ("p_min", p_min_tensor, "p_min"),
                ("p_max", p_max_tensor, "p_max"),
            ]:
                if tensor_value is not None:
                    array = tensor_value[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
                    np.save(output_dir / subdir / f"{name}.npy", array)
                    decoy_debug[tensor_name] = array
            np.savez_compressed(output_dir / "decoy_debug" / f"{name}.npz", **decoy_debug)
            feature = feature_tensor[0, :, :h, :w].detach().cpu().numpy().astype(np.float16)
            np.savez_compressed(output_dir / "features" / f"{name}.npz", decoder_feature=feature)

            Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(output_dir / "masks" / f"{name}.png")
            Image.fromarray((prob * 255.0).clip(0, 255).astype(np.uint8)).save(output_dir / "probs" / f"{name}.png")

            raw = Denormalization(img[0, 0, :h, :w].detach().cpu().numpy(), img_norm_cfg)
            raw = np.clip(raw, 0, 255).astype(np.uint8)
            panels = [
                np.stack([raw, raw, raw], axis=-1),
                overlay_mask(raw, gt, (0, 255, 0)),
                prob_to_rgb(prob),
                overlay_mask(raw, pred_mask, (255, 0, 0)),
            ]
            vis = np.concatenate(panels, axis=1)
            Image.fromarray(vis).save(output_dir / "vis" / f"{name}.png")

            per_image_rows.append(image_metrics(name, prob, logit, pred_mask, gt, args.threshold))
            for threshold in thresholds:
                threshold_update(threshold_stats, prob, gt, threshold)

            if (idx + 1) % 100 == 0:
                print(f"Exported [{idx + 1}/{len(test_loader)}]", flush=True)

    per_image_fields = [
        "image_name",
        "threshold",
        "IoU",
        "nIoU",
        "Pd",
        "FA",
        "FA_ppm",
        "Precision",
        "Recall",
        "F1",
        "FP_components",
        "target_components",
        "target_area",
        "mean_prob_target",
        "mean_prob_bg",
        "mean_logit_target",
        "mean_logit_bg",
    ]
    write_csv(output_dir / "metrics_per_image.csv", per_image_rows, per_image_fields)

    rows = threshold_rows(threshold_stats)
    threshold_fields = [
        "threshold",
        "mIoU",
        "nIoU",
        "Pd",
        "FA",
        "FA_ppm",
        "Precision",
        "Recall",
        "F1",
        "FP_components",
    ]
    write_csv(output_dir / "threshold_curve.csv", rows, threshold_fields)
    summary = {
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "model": args.model_name,
        "mshnet_export_head": resolve_mshnet_head(
            args.model_name,
            args.mshnet_export_head,
            args.allow_legacy_mshnet_output0,
        ),
        "seed": args.seed,
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "threshold": args.threshold,
        "metrics_at_threshold": next(row for row in rows if abs(row["threshold"] - args.threshold) < 1e-9),
        "num_images": len(per_image_rows),
        "outputs": {
            "pred_mask": str(output_dir / "masks"),
            "prob": str(output_dir / "probs"),
            "logit": str(output_dir / "logits"),
            "target_logit": str(output_dir / "target_logits"),
            "clutter_logit": str(output_dir / "clutter_logits"),
            "evidence_logit": str(output_dir / "evidence_logits"),
            "risk_logit": str(output_dir / "risk_logits"),
            "risk_prob": str(output_dir / "risk_probs"),
            "validity_logit": str(output_dir / "validity_logits"),
            "validity_prob": str(output_dir / "validity_probs"),
            "suppression_map": str(output_dir / "suppression_maps"),
            "final_logit": str(output_dir / "final_logits"),
            "p_mean": str(output_dir / "p_mean"),
            "p_std": str(output_dir / "p_std"),
            "p_min": str(output_dir / "p_min"),
            "p_max": str(output_dir / "p_max"),
            "decoy_debug": str(output_dir / "decoy_debug"),
            "decoder_feature": str(output_dir / "features"),
            "vis": str(output_dir / "vis"),
            "per_image_metrics": str(output_dir / "metrics_per_image.csv"),
            "threshold_curve": str(output_dir / "threshold_curve.csv"),
        },
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics_at_threshold"], indent=2), flush=True)
    if not args.skip_direct_export_parity_gate:
        run_direct_export_parity_gate(args, output_dir)


if __name__ == "__main__":
    main()
