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
from skimage import measure, morphology
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
CANDIDATE_TOPK_METRICS = (
    "confidence",
    "instability",
    "sps_score",
    "target_margin_instability",
    "target_margin_sps_score",
    "target_contrast_instability",
    "target_contrast_sps_score",
)
CANDIDATE_POOL_METRICS = CANDIDATE_TOPK_METRICS + ("confidence_and_target_contrast",)
CANDIDATE_POOL_METRICS = CANDIDATE_POOL_METRICS + ("target_contrast", "confidence_x_target_contrast")
RERANK_SIGNAL_METRICS = ("instability", "target_margin", "target_contrast", "none")
RERANK_BASE_METRICS = ("weak_neg_loss", "mean_neg_loss", "confidence", "none")


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")


def load_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def load_checkpoint(net: Net, checkpoint_path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint if isinstance(checkpoint, dict) else {}


def forward_prob(net: Net, img: torch.Tensor, h: int, w: int) -> np.ndarray:
    with torch.no_grad():
        logit = net.export_logits_features(img)["logit"]
        prob = foreground_probability(logit)[0, 0, :h, :w]
    return prob.detach().cpu().numpy().astype(np.float32)


def perturb_image(img: torch.Tensor, perturbation: str, gain: float, offset: float, noise_std: float):
    if perturbation == "hflip":
        return torch.flip(img, dims=[-1]), "hflip"
    if perturbation == "vflip":
        return torch.flip(img, dims=[-2]), "vflip"
    if perturbation == "hvflip":
        return torch.flip(img, dims=[-2, -1]), "hvflip"
    if perturbation == "transpose":
        return img.transpose(-1, -2).contiguous(), "transpose"
    if perturbation == "gain_offset":
        return img * gain + offset, "identity"
    if perturbation == "gaussian_noise":
        return img + torch.randn_like(img) * noise_std, "identity"
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def parse_perturbation_spec(spec: str) -> tuple[str, dict]:
    parts = [item.strip() for item in spec.split(":") if item.strip()]
    name = parts[0]
    params = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip()] = float(value)
    return name, params


def perturb_from_spec(img: torch.Tensor, spec: str, args):
    name, params = parse_perturbation_spec(spec)
    gain = float(params.get("gain", args.gain))
    offset = float(params.get("offset", args.offset))
    noise_std = float(params.get("noise_std", args.noise_std))
    return perturb_image(img, name, gain, offset, noise_std)


def align_back(array: np.ndarray, op: str) -> np.ndarray:
    if op == "hflip":
        return np.flip(array, axis=1).copy()
    if op == "vflip":
        return np.flip(array, axis=0).copy()
    if op == "hvflip":
        return np.flip(array, axis=(0, 1)).copy()
    if op == "transpose":
        return array.T.copy()
    if op == "identity":
        return array
    raise ValueError(f"Unsupported inverse op: {op}")


def adaptive_far_mask(gt: np.ndarray, kappa: float, r0: float, rmin: int, rmax: int) -> np.ndarray:
    if not gt.any():
        return np.ones_like(gt, dtype=bool)
    blocked = np.zeros_like(gt, dtype=bool)
    labels = measure.label(gt.astype(np.uint8), connectivity=2)
    for region in measure.regionprops(labels):
        radius = int(math.ceil(kappa * math.sqrt(float(region.area) / math.pi)) + r0)
        radius = max(rmin, min(rmax, radius))
        component = labels == region.label
        blocked |= morphology.binary_dilation(component, morphology.disk(radius))
    return ~blocked


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def classify_pixels(prob400: np.ndarray, ensemble_prob: np.ndarray, gt: np.ndarray, threshold: float, far_mask: np.ndarray):
    pred = prob400 > threshold
    ens_pred = ensemble_prob > threshold
    boundary = np.zeros_like(gt, dtype=bool)
    detached_near = np.zeros_like(gt, dtype=bool)
    detached_far = np.zeros_like(gt, dtype=bool)
    tce_removed_far = np.zeros_like(gt, dtype=bool)

    gt_d1 = morphology.binary_dilation(gt, morphology.disk(1)) if gt.any() else gt
    gt_d10 = morphology.binary_dilation(gt, morphology.disk(10)) if gt.any() else gt
    for region in connected_regions(pred):
        coords = region.coords
        comp = np.zeros_like(gt, dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True
        if np.logical_and(comp, gt).any():
            boundary |= np.logical_and(comp, ~gt)
            continue
        near_part = np.logical_and(comp, gt_d10)
        near_part = np.logical_and(near_part, ~gt_d1)
        far_part = np.logical_and(comp, far_mask)
        detached_near |= near_part
        detached_far |= far_part
        tce_removed_far |= np.logical_and(far_part, ~ens_pred)

    easy_bg = np.logical_and(~gt, ~pred)
    easy_bg = np.logical_and(easy_bg, far_mask)
    return {
        "target": gt,
        "boundary_excess": boundary,
        "detached_near_fp": detached_near,
        "detached_far_fp": detached_far,
        "easy_background": easy_bg,
        "tce_removed_far_fp": tce_removed_far,
    }


def select_instability_map(
    net: Net,
    img: torch.Tensor,
    h: int,
    w: int,
    prob400: np.ndarray,
    gt: np.ndarray,
    far_mask: np.ndarray,
    args,
) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    best = None
    specs = [item.strip() for item in args.perturbation_pool.split(",") if item.strip()]
    for spec in specs:
        p_img, inverse_op = perturb_from_spec(img, spec, args)
        ph, pw = p_img.shape[-2:]
        prob_perturb = align_back(forward_prob(net, p_img, ph, pw), inverse_op)[:h, :w]
        u_sps = np.abs(prob400 - prob_perturb).astype(np.float32)
        confidence = np.maximum(prob400, prob_perturb)
        far_score = float((confidence[far_mask] * u_sps[far_mask]).mean()) if far_mask.any() else 0.0
        target_score = float(u_sps[gt].mean()) if gt.any() else 0.0
        score = far_score - args.selection_beta * target_score
        if best is None or score > best[0]:
            best = (score, u_sps, prob_perturb, spec, far_score, target_score)
    if best is None:
        raise ValueError("--perturbation_pool did not contain any valid perturbation.")
    return best[1], best[2], best[3], best[4], best[5]


def sps_candidate_and_selected(
    prob_w: np.ndarray,
    prob_p: np.ndarray,
    instability: np.ndarray,
    gt: np.ndarray,
    far_mask: np.ndarray,
    candidate_tau: float,
    candidate_topk_ratio: float,
    candidate_topk_metric: str,
    candidate_min_metric: float | None,
    candidate_min_confidence: float,
    candidate_fallback_topk_ratio: float,
    candidate_expand_radius: int,
    candidate_expand_min_confidence: float,
    target_margin_quantile: float,
    target_margin_temp: float,
    target_margin_min: float,
    budget_q: float,
    kmax: int,
    eta: float,
    candidate_pool_metric: str | None = None,
    rerank_signal_metric: str | None = None,
    rerank_base_metric: str | None = None,
    fixed_budget_pixels: int | None = None,
    return_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    confidence = np.maximum(prob_w, prob_p)
    weak_neg_loss = -np.log1p(-np.clip(prob_w, 0.0, 1.0 - 1e-6))
    perturb_neg_loss = -np.log1p(-np.clip(prob_p, 0.0, 1.0 - 1e-6))
    hardness = 0.5 * (weak_neg_loss + perturb_neg_loss)
    margin_signal = None
    contrast_signal = None

    def get_target_margin_signal() -> np.ndarray:
        nonlocal margin_signal
        if margin_signal is not None:
            return margin_signal
        if gt.any():
            q = min(1.0, max(0.0, float(target_margin_quantile)))
            ref = float(np.quantile(instability[gt], q)) + max(0.0, float(target_margin_min))
        else:
            ref = max(0.0, float(target_margin_min))
        margin_signal = np.maximum(instability - ref, 0.0)
        return margin_signal

    def get_target_contrast_signal() -> np.ndarray:
        nonlocal contrast_signal
        if contrast_signal is not None:
            return contrast_signal
        if gt.any():
            q = min(1.0, max(0.0, float(target_margin_quantile)))
            ref = float(np.quantile(instability[gt], q)) + max(0.0, float(target_margin_min))
        else:
            ref = max(0.0, float(target_margin_min))
        temp = max(1e-6, float(target_margin_temp))
        raw = np.clip((instability - ref) / temp, -60.0, 60.0)
        contrast_signal = 1.0 / (1.0 + np.exp(-raw))
        return contrast_signal

    def metric_by_name(name: str) -> np.ndarray:
        if name == "confidence":
            return confidence
        if name == "instability":
            return instability
        if name == "sps_score":
            return hardness * np.power(instability + 1e-6, eta)
        if name == "target_margin_instability":
            return get_target_margin_signal()
        if name == "target_margin_sps_score":
            return hardness * np.power(get_target_margin_signal(), eta)
        if name == "target_contrast_instability":
            return get_target_contrast_signal()
        if name == "target_contrast":
            return get_target_contrast_signal()
        if name == "target_contrast_sps_score":
            return hardness * np.power(get_target_contrast_signal(), eta)
        if name == "confidence_and_target_contrast":
            return confidence * get_target_contrast_signal()
        if name == "confidence_x_target_contrast":
            return confidence * get_target_contrast_signal()
        raise ValueError(f"Unsupported metric: {name}")

    def rerank_score() -> np.ndarray:
        if candidate_pool_metric is None and rerank_signal_metric is None and rerank_base_metric is None:
            if candidate_topk_metric.startswith("target_margin_"):
                return hardness * np.power(get_target_margin_signal(), eta)
            if candidate_topk_metric.startswith("target_contrast_"):
                return hardness * np.power(get_target_contrast_signal(), eta)
            return hardness * np.power(instability + 1e-6, eta)

        base_name = rerank_base_metric or "weak_neg_loss"
        signal_name = rerank_signal_metric or "target_contrast"
        if base_name == "weak_neg_loss":
            base = weak_neg_loss
        elif base_name == "mean_neg_loss":
            base = hardness
        elif base_name == "confidence":
            base = confidence
        elif base_name == "none":
            base = np.ones_like(instability, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported rerank_base_metric: {base_name}")

        if signal_name == "instability":
            signal = instability + 1e-6
        elif signal_name == "target_margin":
            signal = get_target_margin_signal()
        elif signal_name == "target_contrast":
            signal = get_target_contrast_signal()
        elif signal_name == "none":
            signal = np.ones_like(instability, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported rerank_signal_metric: {signal_name}")
        return base * np.power(np.maximum(signal, 0.0), eta)

    pool_metric_name = candidate_pool_metric or candidate_topk_metric
    metric = metric_by_name(pool_metric_name)

    def fill_topk(base: np.ndarray, ratio: float) -> np.ndarray:
        if ratio <= 0:
            return base
        valid = far_mask.reshape(-1).copy()
        if candidate_min_metric is not None:
            valid &= metric.reshape(-1) > float(candidate_min_metric)
        if candidate_min_confidence > 0:
            valid &= confidence.reshape(-1) >= float(candidate_min_confidence)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size < 1:
            return base
        valid_metric = metric.reshape(-1)[valid_idx]
        k_cand = max(1, int(math.floor(ratio * valid_idx.size)))
        k_cand = min(k_cand, valid_idx.size)
        top_local = np.argpartition(valid_metric, -k_cand)[-k_cand:]
        base.reshape(-1)[valid_idx[top_local]] = True
        return base

    fallback_used = False
    if candidate_topk_ratio > 0:
        candidate = fill_topk(np.zeros_like(far_mask, dtype=bool), candidate_topk_ratio)
    else:
        candidate = np.logical_and(far_mask, confidence > candidate_tau)
        if candidate_fallback_topk_ratio > 0 and not candidate.any():
            fallback_used = True
            candidate = fill_topk(candidate, candidate_fallback_topk_ratio)
    if candidate_expand_radius > 0 and candidate.any():
        footprint = np.ones((2 * int(candidate_expand_radius) + 1, 2 * int(candidate_expand_radius) + 1), dtype=bool)
        candidate = np.logical_and(morphology.binary_dilation(candidate, footprint=footprint), far_mask)
        if candidate_expand_min_confidence > 0:
            candidate = np.logical_and(candidate, confidence >= float(candidate_expand_min_confidence))
    selected = np.zeros_like(candidate, dtype=bool)
    diagnostics = {
        "candidate_pool_metric": pool_metric_name,
        "rerank_signal_metric": rerank_signal_metric,
        "rerank_base_metric": rerank_base_metric,
        "candidate_pixels": int(candidate.sum()),
        "budget_pixels": int(max(0, fixed_budget_pixels or 0)),
        "candidate_to_budget_ratio": 0.0,
        "candidate_under_budget": False,
        "selected_pixels": 0,
        "fallback_used": bool(fallback_used),
    }
    if not candidate.any():
        diagnostics["candidate_under_budget"] = bool(fixed_budget_pixels and fixed_budget_pixels > 0)
        if return_stats:
            return candidate, selected, diagnostics
        return candidate, selected

    score = rerank_score()
    valid_scores = score[candidate]
    if fixed_budget_pixels is None:
        budget_pixels = max(1, int(math.floor(budget_q * int(candidate.sum()))))
        budget_pixels = min(budget_pixels, max(1, int(kmax)))
    else:
        budget_pixels = max(0, int(fixed_budget_pixels))
    k = min(budget_pixels, valid_scores.size)
    diagnostics["candidate_under_budget"] = bool(valid_scores.size < budget_pixels)
    if k <= 0:
        diagnostics.update({
            "budget_pixels": int(budget_pixels),
            "candidate_to_budget_ratio": float(int(candidate.sum()) / max(1, int(budget_pixels))),
            "selected_pixels": 0,
        })
        if return_stats:
            return candidate, selected, diagnostics
        return candidate, selected
    valid_indices = np.flatnonzero(candidate.reshape(-1))
    top_local = np.argpartition(valid_scores, -k)[-k:]
    flat = selected.reshape(-1)
    flat[valid_indices[top_local]] = True
    diagnostics.update({
        "budget_pixels": int(budget_pixels),
        "candidate_to_budget_ratio": float(int(candidate.sum()) / max(1, int(budget_pixels))),
        "selected_pixels": int(selected.sum()),
    })
    if return_stats:
        return candidate, selected, diagnostics
    return candidate, selected


def target_contrast_signal_np(
    instability: np.ndarray,
    gt: np.ndarray,
    target_margin_quantile: float,
    target_margin_temp: float,
    target_margin_min: float,
) -> np.ndarray:
    if gt.any():
        q = min(1.0, max(0.0, float(target_margin_quantile)))
        ref = float(np.quantile(instability[gt], q)) + max(0.0, float(target_margin_min))
    else:
        ref = max(0.0, float(target_margin_min))
    temp = max(1e-6, float(target_margin_temp))
    raw = np.clip((instability - ref) / temp, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)


def region_candidate_and_selected(
    prob_w: np.ndarray,
    prob_p: np.ndarray,
    instability: np.ndarray,
    gt: np.ndarray,
    far_mask: np.ndarray,
    mode: str,
    budget_pixels: int,
    target_margin_quantile: float,
    target_margin_temp: float,
    target_margin_min: float,
    region_min_area: int,
    region_max_area: int,
    region_conf_min: float,
    region_signal_min: float,
    region_pool_topq: float,
    peak_topk_ratio: float,
    peak_nms_radius: int,
    peak_window_radius: int,
    peak_min_conf: float,
    peak_min_signal: float,
) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    confidence = np.maximum(prob_w, prob_p)
    weak_neg_loss = -np.log1p(-np.clip(prob_w, 0.0, 1.0 - 1e-6))
    perturb_neg_loss = -np.log1p(-np.clip(prob_p, 0.0, 1.0 - 1e-6))
    hardness = 0.5 * (weak_neg_loss + perturb_neg_loss)
    target_contrast = target_contrast_signal_np(
        instability,
        gt,
        target_margin_quantile,
        target_margin_temp,
        target_margin_min,
    )
    local_select_score = hardness * target_contrast

    candidate = np.zeros_like(far_mask, dtype=bool)
    selected = np.zeros_like(far_mask, dtype=bool)
    region_records = []
    proposals = []

    def add_region(region_mask: np.ndarray, region_id: int, peak_y: int | None = None, peak_x: int | None = None):
        region_mask = np.logical_and(region_mask, far_mask)
        area = int(region_mask.sum())
        touches_blocked = bool(np.logical_and(region_mask, ~far_mask).any() or np.logical_and(region_mask, gt).any())
        if area < int(region_min_area) or area > int(region_max_area):
            return
        if touches_blocked:
            return
        if area <= 0:
            return
        mean_conf = float(confidence[region_mask].mean())
        mean_signal = float(target_contrast[region_mask].mean())
        if mean_conf < float(region_conf_min):
            return
        if mean_signal < float(region_signal_min):
            return
        peak_conf = float(confidence[region_mask].max())
        mean_hardness = float(hardness[region_mask].mean())
        mean_instability = float(instability[region_mask].mean())
        score = mean_hardness * mean_signal * peak_conf
        proposals.append({
            "id": int(region_id),
            "mask": region_mask,
            "area": area,
            "mean_confidence": mean_conf,
            "peak_confidence": peak_conf,
            "mean_hardness": mean_hardness,
            "mean_instability": mean_instability,
            "mean_target_contrast": mean_signal,
            "region_score": float(score),
            "touches_dilated_target": int(touches_blocked),
            "peak_y": None if peak_y is None else int(peak_y),
            "peak_x": None if peak_x is None else int(peak_x),
        })

    if mode == "region_component":
        valid = np.logical_and(far_mask, confidence >= float(region_conf_min))
        proposal = np.logical_and(valid, target_contrast >= float(region_signal_min))
        if region_pool_topq > 0 and valid.any():
            cutoff = float(np.quantile(target_contrast[valid], 1.0 - min(1.0, max(0.0, float(region_pool_topq)))))
            proposal = np.logical_or(proposal, np.logical_and(valid, target_contrast >= cutoff))
        labels = measure.label(proposal.astype(np.uint8), connectivity=2)
        for region in measure.regionprops(labels):
            region_mask = labels == region.label
            add_region(region_mask, int(region.label))
    elif mode == "peak_region":
        valid = far_mask.copy()
        valid &= confidence >= float(peak_min_conf)
        valid &= target_contrast >= float(peak_min_signal)
        peak_score = confidence * target_contrast
        valid_idx = np.flatnonzero(valid.reshape(-1))
        if valid_idx.size:
            k_peaks = max(1, int(math.floor(float(peak_topk_ratio) * valid_idx.size)))
            k_peaks = min(k_peaks, valid_idx.size)
            valid_scores = peak_score.reshape(-1)[valid_idx]
            top_local = np.argpartition(valid_scores, -k_peaks)[-k_peaks:]
            peak_indices = valid_idx[top_local]
            peak_indices = peak_indices[np.argsort(-peak_score.reshape(-1)[peak_indices], kind="mergesort")]
            accepted: list[tuple[int, int]] = []
            radius_sq = int(peak_nms_radius) ** 2
            h, w = far_mask.shape
            for flat_idx in peak_indices:
                y, x = np.unravel_index(int(flat_idx), far_mask.shape)
                if any((int(y) - ay) ** 2 + (int(x) - ax) ** 2 <= radius_sq for ay, ax in accepted):
                    continue
                accepted.append((int(y), int(x)))
                y0 = max(0, int(y) - int(peak_window_radius))
                y1 = min(h, int(y) + int(peak_window_radius) + 1)
                x0 = max(0, int(x) - int(peak_window_radius))
                x1 = min(w, int(x) + int(peak_window_radius) + 1)
                region_mask = np.zeros_like(far_mask, dtype=bool)
                region_mask[y0:y1, x0:x1] = True
                add_region(region_mask, len(accepted), int(y), int(x))
    else:
        raise ValueError(f"Unsupported candidate mode: {mode}")

    for proposal in proposals:
        candidate |= proposal["mask"]
    candidate_pixels = int(candidate.sum())
    budget_pixels = max(0, int(budget_pixels))
    selected_region_ids = set()
    selected_pixels = 0
    for proposal in sorted(proposals, key=lambda item: item["region_score"], reverse=True):
        if selected_pixels >= budget_pixels:
            break
        available = np.logical_and(proposal["mask"], ~selected)
        available_count = int(available.sum())
        if available_count <= 0:
            continue
        remaining = budget_pixels - selected_pixels
        if available_count <= remaining:
            chosen = available
        else:
            scores = local_select_score[available]
            chosen = np.zeros_like(available, dtype=bool)
            available_idx = np.flatnonzero(available.reshape(-1))
            top_local = np.argpartition(scores, -remaining)[-remaining:]
            chosen.reshape(-1)[available_idx[top_local]] = True
        if chosen.any():
            selected |= chosen
            selected_pixels = int(selected.sum())
            selected_region_ids.add(int(proposal["id"]))

    for proposal in proposals:
        region_records.append({
            "region_id": int(proposal["id"]),
            "_mask": proposal["mask"],
            "area": int(proposal["area"]),
            "mean_confidence": float(proposal["mean_confidence"]),
            "peak_confidence": float(proposal["peak_confidence"]),
            "mean_hardness": float(proposal["mean_hardness"]),
            "mean_instability": float(proposal["mean_instability"]),
            "mean_target_contrast": float(proposal["mean_target_contrast"]),
            "region_score": float(proposal["region_score"]),
            "touches_dilated_target": int(proposal["touches_dilated_target"]),
            "selected": int(int(proposal["id"]) in selected_region_ids),
            "peak_y": proposal["peak_y"],
            "peak_x": proposal["peak_x"],
        })

    diagnostics = {
        "candidate_pixels": candidate_pixels,
        "budget_pixels": budget_pixels,
        "candidate_to_budget_ratio": float(candidate_pixels / max(1, budget_pixels)),
        "candidate_under_budget": bool(candidate_pixels < budget_pixels),
        "candidate_region_count": int(len(proposals)),
        "selected_region_count": int(len(selected_region_ids)),
        "selected_pixels": int(selected.sum()),
        "fallback_used": False,
    }
    return candidate, selected, diagnostics, region_records


def topk_mask(score: np.ndarray, valid: np.ndarray, ratio: float) -> np.ndarray:
    selected = np.zeros_like(valid, dtype=bool)
    if not valid.any():
        return selected
    valid_scores = score[valid]
    k = max(1, int(math.floor(ratio * valid_scores.size)))
    k = min(k, valid_scores.size)
    valid_indices = np.flatnonzero(valid.reshape(-1))
    top_local = np.argpartition(valid_scores, -k)[-k:]
    flat = selected.reshape(-1)
    flat[valid_indices[top_local]] = True
    return selected


def selected_component_coverage(selected: np.ndarray, component_mask: np.ndarray) -> tuple[int, int]:
    labels = measure.label(component_mask.astype(np.uint8), connectivity=2)
    num_components = int(labels.max())
    if num_components == 0:
        return 0, 0
    touched = np.unique(labels[np.logical_and(selected, component_mask)])
    touched = touched[touched > 0]
    return int(touched.size), num_components


def stats(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "q75": 0.0, "q90": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
    }


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled = math.sqrt(0.5 * (float(a.var()) + float(b.var())) + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    n = min(x.size, y.size)
    rx = rankdata(x[:n])
    ry = rankdata(y[:n])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return 0.0
    ranks = rankdata(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(sorted_labels.size) + 1.0)
    return float((precision * sorted_labels).sum() / pos)


def binary_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict:
    if pos_scores.size == 0 or neg_scores.size == 0:
        return {"auroc": 0.0, "auprc": 0.0, "effect_size": 0.0}
    y = np.concatenate([np.ones(pos_scores.size), np.zeros(neg_scores.size)])
    scores = np.concatenate([pos_scores, neg_scores])
    if np.unique(y).size < 2:
        return {"auroc": 0.0, "auprc": 0.0, "effect_size": effect_size(pos_scores, neg_scores)}
    return {
        "auroc": roc_auc_binary(y, scores),
        "auprc": average_precision_binary(y, scores),
        "effect_size": effect_size(pos_scores, neg_scores),
    }


def sample_for_corr(a: list[np.ndarray], b: list[np.ndarray], max_points: int, rng: np.random.Generator):
    x = np.concatenate([item.reshape(-1) for item in a if item.size])
    y = np.concatenate([item.reshape(-1) for item in b if item.size])
    n = min(x.size, y.size)
    if n <= max_points:
        return x[:n], y[:n]
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Census whether SPS perturbation instability predicts TCE-removed far-FP.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--checkpoint_e400", required=True)
    parser.add_argument("--tce_checkpoints", required=True, help="Comma-separated checkpoints, e.g. e250,e300,e350,e400.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--perturbation", default="gain_offset",
                        choices=["hflip", "vflip", "hvflip", "transpose", "gain_offset", "gaussian_noise"])
    parser.add_argument("--select_target_preserving", action="store_true")
    parser.add_argument("--perturbation_pool", default="hflip,vflip,gain_offset:gain=1.02:offset=0,gaussian_noise:noise_std=0.005")
    parser.add_argument("--selection_beta", type=float, default=2.0)
    parser.add_argument("--gain", type=float, default=1.08)
    parser.add_argument("--offset", type=float, default=0.02)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--radius_kappa", type=float, default=1.0)
    parser.add_argument("--radius_r0", type=float, default=2.0)
    parser.add_argument("--radius_min", type=int, default=3)
    parser.add_argument("--radius_max", type=int, default=9)
    parser.add_argument("--top_quantile", type=float, default=0.20)
    parser.add_argument("--candidate_mode", default="pixel",
                        choices=["pixel", "region_component", "peak_region"])
    parser.add_argument("--candidate_tau", type=float, default=0.3)
    parser.add_argument("--candidate_topk_ratio", type=float, default=0.0)
    parser.add_argument("--candidate_topk_metric", default="confidence",
                        choices=CANDIDATE_TOPK_METRICS)
    parser.add_argument("--candidate_pool_metric", default=None,
                        choices=CANDIDATE_POOL_METRICS)
    parser.add_argument("--rerank_signal_metric", default=None,
                        choices=RERANK_SIGNAL_METRICS)
    parser.add_argument("--rerank_base_metric", default=None,
                        choices=RERANK_BASE_METRICS)
    parser.add_argument("--min_candidate_to_budget_ratio", type=float, default=0.0)
    parser.add_argument("--diagnostic_max_ohem_overlap", type=float, default=0.70)
    parser.add_argument("--candidate_min_metric", type=float, default=None)
    parser.add_argument("--candidate_min_confidence", type=float, default=0.0)
    parser.add_argument("--candidate_fallback_topk_ratio", type=float, default=0.0)
    parser.add_argument("--candidate_expand_radius", type=int, default=0)
    parser.add_argument("--candidate_expand_min_confidence", type=float, default=0.0)
    parser.add_argument("--target_margin_quantile", type=float, default=0.85)
    parser.add_argument("--target_margin_temp", type=float, default=0.01)
    parser.add_argument("--target_margin_min", type=float, default=0.0)
    parser.add_argument("--flat_conf_tau", type=float, default=0.5)
    parser.add_argument("--budget_q", type=float, default=0.1)
    parser.add_argument("--kmax", type=int, default=256)
    parser.add_argument("--region_min_area", type=int, default=1)
    parser.add_argument("--region_max_area", type=int, default=128)
    parser.add_argument("--region_conf_min", type=float, default=0.10)
    parser.add_argument("--region_signal_min", type=float, default=0.50)
    parser.add_argument("--region_pool_topq", type=float, default=0.05)
    parser.add_argument("--region_score_metric", default="mean_hardness_x_signal_x_peak_conf")
    parser.add_argument("--region_budget_fill_policy", default="region_then_local_top",
                        choices=["region_then_local_top"])
    parser.add_argument("--peak_topk_ratio", type=float, default=0.005)
    parser.add_argument("--peak_nms_radius", type=int, default=3)
    parser.add_argument("--peak_window_radius", type=int, default=4)
    parser.add_argument("--peak_min_conf", type=float, default=0.10)
    parser.add_argument("--peak_min_signal", type=float, default=0.50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ohem_topk_ratio", type=float, default=0.01)
    parser.add_argument("--corr_max_points", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset_name = args.train_dataset_name or args.dataset_name
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    image_filter = None
    if args.image_list:
        image_filter = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    test_set = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    if image_filter is not None:
        test_set.test_list = image_filter
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    loss_cfg = vars(args)
    base_net = Net(model_name=args.model_name, mode="test", loss_cfg=loss_cfg).to(device)
    load_checkpoint(base_net, Path(args.checkpoint_e400), device)
    tce_nets = []
    for path in [Path(item) for item in args.tce_checkpoints.split(",") if item.strip()]:
        net = Net(model_name=args.model_name, mode="test", loss_cfg=loss_cfg).to(device)
        load_checkpoint(net, path, device)
        tce_nets.append(net)
    if len(tce_nets) < 2:
        raise ValueError("--tce_checkpoints must contain at least two checkpoints.")

    region_values = {
        "target": [],
        "boundary_excess": [],
        "detached_near_fp": [],
        "detached_far_fp": [],
        "easy_background": [],
        "tce_removed_far_fp": [],
    }
    corr_sps = []
    corr_tce = []
    per_image_top20_target_rates = []
    per_image_top20_removed_recalls = []
    candidate_flat = 0
    candidate_total = 0
    selected_flat = 0
    selected_total = 0
    candidate_instability_values = []
    candidate_hardness_values = []
    candidate_sps_score_values = []
    target_hardness_values = []
    target_sps_score_values = []
    removed_hardness_values = []
    removed_sps_score_values = []
    selected_removed_total = 0
    selected_detached_far_total = 0
    selected_persistent_far_total = 0
    ohem_overlap_selected_total = 0
    candidate_ohem_overlap_total = 0
    selected_target_leakage_total = 0
    far_fp_components_touched_total = 0
    far_fp_components_total = 0
    num_images_with_candidate_empty = 0
    num_images_with_fallback = 0
    num_images_with_candidate_under_budget = 0
    candidate_to_budget_ratios = []
    budget_pixel_values = []
    candidate_region_counts = []
    selected_region_counts = []
    region_candidate_pixel_values = []
    ohem_removed_total = 0
    per_image_rows = []
    per_region_rows = []

    for idx, (img, gt_mask, size, image_name) in enumerate(test_loader):
        name = image_name[0] if isinstance(image_name, (tuple, list)) else str(image_name)
        img = img.to(device)
        h, w = size_to_int(size[0]), size_to_int(size[1])
        gt = load_mask(find_file(dataset_dir / "masks", name))[:h, :w]

        prob400 = forward_prob(base_net, img, h, w)
        far_mask = adaptive_far_mask(gt, args.radius_kappa, args.radius_r0, args.radius_min, args.radius_max)
        selected_perturbation = args.perturbation
        selected_far_score = 0.0
        selected_target_score = 0.0
        if args.select_target_preserving:
            u_sps, prob_perturb, selected_perturbation, selected_far_score, selected_target_score = select_instability_map(
                base_net, img, h, w, prob400, gt, far_mask, args
            )
        else:
            p_img, inverse_op = perturb_image(img, args.perturbation, args.gain, args.offset, args.noise_std)
            ph, pw = p_img.shape[-2:]
            prob_perturb = align_back(forward_prob(base_net, p_img, ph, pw), inverse_op)[:h, :w]
            u_sps = np.abs(prob400 - prob_perturb).astype(np.float32)

        tce_probs = [forward_prob(net, img, h, w) for net in tce_nets]
        tce_stack = np.stack(tce_probs, axis=0)
        ensemble_prob = tce_stack.mean(axis=0).astype(np.float32)
        u_tce = tce_stack.std(axis=0).astype(np.float32)

        masks = classify_pixels(prob400, ensemble_prob, gt, args.threshold, far_mask)
        hardness = -0.5 * (
            np.log1p(-np.clip(prob400, 0.0, 1.0 - 1e-6))
            + np.log1p(-np.clip(prob_perturb, 0.0, 1.0 - 1e-6))
        )
        ohem_mask = topk_mask(hardness, ~gt, args.ohem_topk_ratio)
        fixed_budget_pixels = int(ohem_mask.sum())
        if args.candidate_mode == "pixel":
            candidate_mask, selected_mask, selection_stats = sps_candidate_and_selected(
                prob400,
                prob_perturb,
                u_sps,
                gt,
                far_mask,
                args.candidate_tau,
                args.candidate_topk_ratio,
                args.candidate_topk_metric,
                args.candidate_min_metric,
                args.candidate_min_confidence,
                args.candidate_fallback_topk_ratio,
                args.candidate_expand_radius,
                args.candidate_expand_min_confidence,
                args.target_margin_quantile,
                args.target_margin_temp,
                args.target_margin_min,
                args.budget_q,
                args.kmax,
                args.eta,
                candidate_pool_metric=args.candidate_pool_metric,
                rerank_signal_metric=args.rerank_signal_metric,
                rerank_base_metric=args.rerank_base_metric,
                fixed_budget_pixels=fixed_budget_pixels,
                return_stats=True,
            )
            image_region_rows = []
        else:
            candidate_mask, selected_mask, selection_stats, image_region_rows = region_candidate_and_selected(
                prob400,
                prob_perturb,
                u_sps,
                gt,
                far_mask,
                mode=args.candidate_mode,
                budget_pixels=fixed_budget_pixels,
                target_margin_quantile=args.target_margin_quantile,
                target_margin_temp=args.target_margin_temp,
                target_margin_min=args.target_margin_min,
                region_min_area=args.region_min_area,
                region_max_area=args.region_max_area,
                region_conf_min=args.region_conf_min,
                region_signal_min=args.region_signal_min,
                region_pool_topq=args.region_pool_topq,
                peak_topk_ratio=args.peak_topk_ratio,
                peak_nms_radius=args.peak_nms_radius,
                peak_window_radius=args.peak_window_radius,
                peak_min_conf=args.peak_min_conf,
                peak_min_signal=args.peak_min_signal,
            )
        sps_score = hardness * np.power(u_sps + 1e-6, args.eta)
        easy_bg = masks["easy_background"]
        if easy_bg.any():
            easy_tce_q90 = float(np.quantile(u_tce[easy_bg], 0.90))
        else:
            easy_tce_q90 = 0.0
        confidence = np.maximum(prob400, prob_perturb)
        flat_like_bg = easy_bg & (confidence < args.flat_conf_tau) & (u_tce <= easy_tce_q90)
        candidate_flat += int(np.logical_and(candidate_mask, flat_like_bg).sum())
        candidate_total += int(candidate_mask.sum())
        selected_flat += int(np.logical_and(selected_mask, flat_like_bg).sum())
        selected_total += int(selected_mask.sum())
        if candidate_mask.any():
            candidate_instability_values.append(u_sps[candidate_mask])
            candidate_hardness_values.append(hardness[candidate_mask])
            candidate_sps_score_values.append(sps_score[candidate_mask])
        removed = masks["tce_removed_far_fp"]
        if gt.any():
            target_hardness_values.append(hardness[gt])
            target_sps_score_values.append(sps_score[gt])
        if removed.any():
            removed_hardness_values.append(hardness[removed])
            removed_sps_score_values.append(sps_score[removed])
        selected_removed_pixels = int(np.logical_and(selected_mask, removed).sum())
        selected_detached_far_pixels = int(np.logical_and(selected_mask, masks["detached_far_fp"]).sum())
        selected_tce_removed_far_fp_pixels = selected_removed_pixels
        selected_ohem_overlap_pixels = int(np.logical_and(selected_mask, ohem_mask).sum())
        candidate_ohem_overlap_pixels = int(np.logical_and(candidate_mask, ohem_mask).sum())
        selected_target_leakage_pixels = int(np.logical_and(selected_mask, gt).sum())
        selected_flat_background_ratio = float(np.logical_and(selected_mask, flat_like_bg).sum() / max(1, selected_mask.sum()))
        candidate_to_budget_ratio = float(selection_stats.get("candidate_to_budget_ratio", 0.0))
        fallback_used = bool(selection_stats.get("fallback_used", False))
        candidate_under_budget = bool(selection_stats.get("candidate_under_budget", False))
        candidate_empty = int(candidate_mask.sum()) == 0
        if candidate_empty:
            num_images_with_candidate_empty += 1
        if fallback_used:
            num_images_with_fallback += 1
        if candidate_under_budget:
            num_images_with_candidate_under_budget += 1
        candidate_to_budget_ratios.append(candidate_to_budget_ratio)
        budget_pixel_values.append(int(selection_stats.get("budget_pixels", fixed_budget_pixels)))
        candidate_region_counts.append(int(selection_stats.get("candidate_region_count", 0)))
        selected_region_counts.append(int(selection_stats.get("selected_region_count", 0)))
        region_candidate_pixel_values.append(int(candidate_mask.sum()))
        touched_components, total_components = selected_component_coverage(selected_mask, masks["detached_far_fp"])
        far_fp_components_touched_total += touched_components
        far_fp_components_total += total_components

        selected_removed_total += selected_removed_pixels
        selected_detached_far_total += selected_detached_far_pixels
        selected_persistent_far_total += int(np.logical_and(selected_mask, np.logical_and(masks["detached_far_fp"], ~removed)).sum())
        ohem_overlap_selected_total += selected_ohem_overlap_pixels
        candidate_ohem_overlap_total += candidate_ohem_overlap_pixels
        selected_target_leakage_total += selected_target_leakage_pixels
        ohem_removed_total += int(np.logical_and(ohem_mask, removed).sum())
        for key, mask in masks.items():
            if mask.any():
                region_values[key].append(u_sps[mask])

        valid_corr = far_mask | gt
        corr_sps.append(u_sps[valid_corr])
        corr_tce.append(u_tce[valid_corr])

        cutoff = float(np.quantile(u_sps[valid_corr], 1.0 - args.top_quantile)) if valid_corr.any() else 1.0
        top = u_sps >= cutoff
        target_rate = float(np.logical_and(top, gt).sum() / max(1, gt.sum()))
        removed_recall = float(np.logical_and(top, removed).sum() / max(1, removed.sum())) if removed.any() else 0.0
        per_image_top20_target_rates.append(target_rate)
        per_image_top20_removed_recalls.append(removed_recall)
        for region_row in image_region_rows:
            region_mask = region_row.pop("_mask")
            region_selected = int(np.logical_and(region_mask, selected_mask).any())
            ohem_overlap_pixels = int(np.logical_and(region_mask, ohem_mask).sum())
            per_region_rows.append({
                "image_name": name,
                "candidate_mode": args.candidate_mode,
                **region_row,
                "ohem_overlap_pixels": ohem_overlap_pixels,
                "ohem_overlap_fraction": float(ohem_overlap_pixels / max(1, int(region_mask.sum()))),
                "tce_removed_far_fp_pixels": int(np.logical_and(region_mask, removed).sum()),
                "detached_far_fp_pixels": int(np.logical_and(region_mask, masks["detached_far_fp"]).sum()),
                "flat_background_pixels": int(np.logical_and(region_mask, flat_like_bg).sum()),
                "selected": region_selected,
            })
        selected_pixels = int(selected_mask.sum())
        candidate_pixels = int(candidate_mask.sum())
        budget_pixels = int(selection_stats.get("budget_pixels", selected_pixels))
        candidate_ohem_overlap_fraction = float(candidate_ohem_overlap_pixels / max(1, candidate_pixels))
        selected_ohem_overlap_fraction = float(selected_ohem_overlap_pixels / max(1, selected_pixels))
        fail_reasons = []
        if candidate_empty:
            fail_reasons.append("candidate_empty")
        if fallback_used:
            fail_reasons.append("fallback_used")
        if (
            args.min_candidate_to_budget_ratio > 0
            and candidate_to_budget_ratio < float(args.min_candidate_to_budget_ratio)
        ):
            fail_reasons.append("candidate_to_budget_low")
        if candidate_under_budget:
            fail_reasons.append("candidate_under_budget")
        if selected_ohem_overlap_fraction > float(args.diagnostic_max_ohem_overlap):
            fail_reasons.append("selected_ohem_overlap_high")
        if selected_target_leakage_pixels > 0:
            fail_reasons.append("target_leakage")
        if selected_flat_background_ratio > 0.20:
            fail_reasons.append("flat_background_high")
        per_image_rows.append({
            "image_name": name,
            "target_pixels": int(gt.sum()),
            "detached_far_fp_pixels": int(masks["detached_far_fp"].sum()),
            "tce_removed_far_fp_pixels": int(removed.sum()),
            "target_u_mean": float(u_sps[gt].mean()) if gt.any() else 0.0,
            "far_fp_u_mean": float(u_sps[masks["detached_far_fp"]].mean()) if masks["detached_far_fp"].any() else 0.0,
            "top20_target_rate": target_rate,
            "top20_removed_far_fp_recall": removed_recall,
            "selected_perturbation": selected_perturbation,
            "selected_far_score": selected_far_score,
            "selected_target_score": selected_target_score,
            "budget_pixels": budget_pixels,
            "candidate_pixels": candidate_pixels,
            "candidate_to_budget_ratio": candidate_to_budget_ratio,
            "candidate_mode": args.candidate_mode,
            "candidate_region_count": int(selection_stats.get("candidate_region_count", 0)),
            "selected_region_count": int(selection_stats.get("selected_region_count", 0)),
            "candidate_under_budget": int(candidate_under_budget),
            "candidate_ohem_overlap_pixels": candidate_ohem_overlap_pixels,
            "candidate_ohem_overlap_fraction": candidate_ohem_overlap_fraction,
            "candidate_flat_background_ratio": float(np.logical_and(candidate_mask, flat_like_bg).sum() / max(1, candidate_mask.sum())),
            "selected_pixels": selected_pixels,
            "selected_ohem_overlap_pixels": selected_ohem_overlap_pixels,
            "selected_ohem_overlap_fraction": selected_ohem_overlap_fraction,
            "selected_target_leakage_pixels": selected_target_leakage_pixels,
            "selected_detached_far_fp_pixels": selected_detached_far_pixels,
            "selected_tce_removed_far_fp_pixels": selected_tce_removed_far_fp_pixels,
            "selected_flat_background_ratio": selected_flat_background_ratio,
            "fallback_used": int(fallback_used),
            "fail_reason": ";".join(fail_reasons),
            "easy_tce_q90": easy_tce_q90,
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_loader)}", flush=True)

    concat = {
        key: np.concatenate(values) if values else np.asarray([], dtype=np.float32)
        for key, values in region_values.items()
    }
    sx, sy = sample_for_corr(corr_sps, corr_tce, args.corr_max_points, rng)
    spearman = spearman_corr(sx, sy) if sx.size > 1 else 0.0
    if not np.isfinite(spearman):
        spearman = 0.0

    region_stats = {key: stats(values) for key, values in concat.items()}
    far_vs_target = binary_metrics(concat["detached_far_fp"], concat["target"])
    near_vs_target = binary_metrics(concat["detached_near_fp"], concat["target"])
    removed_vs_target = binary_metrics(concat["tce_removed_far_fp"], concat["target"])
    candidate_values = (
        np.concatenate(candidate_instability_values)
        if candidate_instability_values
        else np.asarray([], dtype=np.float32)
    )
    candidate_hardness = (
        np.concatenate(candidate_hardness_values)
        if candidate_hardness_values
        else np.asarray([], dtype=np.float32)
    )
    candidate_sps_score = (
        np.concatenate(candidate_sps_score_values)
        if candidate_sps_score_values
        else np.asarray([], dtype=np.float32)
    )
    target_hardness = (
        np.concatenate(target_hardness_values)
        if target_hardness_values
        else np.asarray([], dtype=np.float32)
    )
    target_sps_score = (
        np.concatenate(target_sps_score_values)
        if target_sps_score_values
        else np.asarray([], dtype=np.float32)
    )
    removed_hardness = (
        np.concatenate(removed_hardness_values)
        if removed_hardness_values
        else np.asarray([], dtype=np.float32)
    )
    removed_sps_score = (
        np.concatenate(removed_sps_score_values)
        if removed_sps_score_values
        else np.asarray([], dtype=np.float32)
    )
    if candidate_values.size:
        candidate_top_cutoff = float(np.quantile(candidate_values, 1.0 - args.top_quantile))
        candidate_top20_target_rate = float(
            (concat["target"] >= candidate_top_cutoff).sum() / max(1, concat["target"].size)
        )
        candidate_top20_removed_recall = float(
            (concat["tce_removed_far_fp"] >= candidate_top_cutoff).sum()
            / max(1, concat["tce_removed_far_fp"].size)
        )
    else:
        candidate_top_cutoff = 0.0
        candidate_top20_target_rate = 0.0
        candidate_top20_removed_recall = 0.0
    if candidate_hardness.size:
        hardness_top_cutoff = float(np.quantile(candidate_hardness, 1.0 - args.top_quantile))
        hardness_top20_target_rate = float((target_hardness >= hardness_top_cutoff).sum() / max(1, target_hardness.size))
        hardness_top20_removed_recall = float(
            (removed_hardness >= hardness_top_cutoff).sum() / max(1, removed_hardness.size)
        )
    else:
        hardness_top_cutoff = 0.0
        hardness_top20_target_rate = 0.0
        hardness_top20_removed_recall = 0.0
    if candidate_sps_score.size:
        score_top_cutoff = float(np.quantile(candidate_sps_score, 1.0 - args.top_quantile))
        score_top20_target_rate = float((target_sps_score >= score_top_cutoff).sum() / max(1, target_sps_score.size))
        score_top20_removed_recall = float(
            (removed_sps_score >= score_top_cutoff).sum() / max(1, removed_sps_score.size)
        )
    else:
        score_top_cutoff = 0.0
        score_top20_target_rate = 0.0
        score_top20_removed_recall = 0.0

    if concat["target"].size and concat["tce_removed_far_fp"].size:
        discrim_pool = np.concatenate([concat["target"], concat["tce_removed_far_fp"]])
        top_cutoff = float(np.quantile(discrim_pool, 1.0 - args.top_quantile))
        global_top20_target_rate = float((concat["target"] >= top_cutoff).sum() / max(1, concat["target"].size))
        global_top20_removed_recall = float(
            (concat["tce_removed_far_fp"] >= top_cutoff).sum() / max(1, concat["tce_removed_far_fp"].size)
        )
    else:
        top_cutoff = 0.0
        global_top20_target_rate = 0.0
        global_top20_removed_recall = 0.0
    candidate_flat_ratio = float(candidate_flat / max(1, candidate_total))
    selected_flat_ratio = float(selected_flat / max(1, selected_total))
    candidate_budget_ratio_values = np.asarray(candidate_to_budget_ratios, dtype=np.float32)
    budget_pixel_array = np.asarray(budget_pixel_values, dtype=np.float32)
    candidate_region_count_array = np.asarray(candidate_region_counts, dtype=np.float32)
    selected_region_count_array = np.asarray(selected_region_counts, dtype=np.float32)
    region_candidate_pixel_array = np.asarray(region_candidate_pixel_values, dtype=np.float32)
    budget_pixels_mean = float(budget_pixel_array.mean()) if budget_pixel_array.size else 0.0
    candidate_region_count_mean = (
        float(candidate_region_count_array.mean()) if candidate_region_count_array.size else 0.0
    )
    selected_region_count_mean = (
        float(selected_region_count_array.mean()) if selected_region_count_array.size else 0.0
    )
    region_candidate_pixels_mean = (
        float(region_candidate_pixel_array.mean()) if region_candidate_pixel_array.size else 0.0
    )
    candidate_to_budget_ratio_mean = (
        float(candidate_budget_ratio_values.mean()) if candidate_budget_ratio_values.size else 0.0
    )
    candidate_to_budget_ratio_min = (
        float(candidate_budget_ratio_values.min()) if candidate_budget_ratio_values.size else 0.0
    )
    candidate_ohem_overlap_fraction = float(candidate_ohem_overlap_total / max(1, candidate_total))
    selected_ohem_overlap_fraction = float(ohem_overlap_selected_total / max(1, selected_total))
    selected_tce_removed_far_fp_recall = float(selected_removed_total / max(1, concat["tce_removed_far_fp"].size))
    selected_detached_far_fp_recall = float(selected_detached_far_total / max(1, concat["detached_far_fp"].size))
    selected_far_fp_component_coverage = float(far_fp_components_touched_total / max(1, far_fp_components_total))
    ohem_removed_far_fp_recall = float(ohem_removed_total / max(1, concat["tce_removed_far_fp"].size))
    is_region_gate = args.candidate_mode in {"region_component", "peak_region"}
    min_candidate_ratio = float(args.min_candidate_to_budget_ratio)
    min_candidate_ratio_floor = 1.0 if is_region_gate else min(1.5, min_candidate_ratio)
    min_selected_removed_recall = ohem_removed_far_fp_recall * (0.5 if is_region_gate else 1.0)

    failed_items = []
    if global_top20_target_rate > 0.15:
        failed_items.append("target_top20_gt_0p15")
    if selected_ohem_overlap_fraction > float(args.diagnostic_max_ohem_overlap):
        failed_items.append("selected_ohem_overlap_gt_limit")
    if num_images_with_candidate_empty > 0:
        failed_items.append("candidate_empty")
    if num_images_with_fallback > 0:
        failed_items.append("fallback_used")
    if num_images_with_candidate_under_budget > 0:
        failed_items.append("candidate_under_budget")
    if (
        args.min_candidate_to_budget_ratio > 0
        and candidate_to_budget_ratio_mean < min_candidate_ratio
    ):
        failed_items.append("candidate_to_budget_mean_lt_min")
    if (
        args.min_candidate_to_budget_ratio > 0
        and candidate_to_budget_ratio_min < min_candidate_ratio_floor
    ):
        failed_items.append("candidate_to_budget_min_lt_floor")
    if selected_flat_ratio > 0.20:
        failed_items.append("selected_flat_background_gt_0p20")
    if selected_tce_removed_far_fp_recall < min_selected_removed_recall:
        failed_items.append("selected_tce_removed_far_fp_recall_lt_required_floor")
    if selected_target_leakage_total > 0:
        failed_items.append("target_leakage")
    if is_region_gate and selected_detached_far_fp_recall <= 0:
        failed_items.append("selected_detached_far_fp_recall_eq_0")
    if is_region_gate and selected_region_count_mean <= 0:
        failed_items.append("selected_region_count_eq_0")

    gate = {
        "far_fp_instability_gt_target": region_stats["detached_far_fp"]["mean"] > region_stats["target"]["mean"],
        "far_fp_vs_target_auroc_ge_0p65": far_vs_target["auroc"] >= 0.65,
        "sps_tce_spearman_ge_0p25": float(spearman) >= 0.25,
        "target_top20_rate_le_0p15": global_top20_target_rate <= 0.15,
        "selected_ohem_overlap_le_limit": selected_ohem_overlap_fraction <= float(args.diagnostic_max_ohem_overlap),
        "selected_target_leakage_eq_0": selected_target_leakage_total == 0,
        "fallback_images_eq_0": num_images_with_fallback == 0,
        "candidate_empty_images_eq_0": num_images_with_candidate_empty == 0,
        "candidate_under_budget_images_eq_0": num_images_with_candidate_under_budget == 0,
        "candidate_to_budget_ratio_mean_ge_min": (
            args.min_candidate_to_budget_ratio <= 0
            or candidate_to_budget_ratio_mean >= min_candidate_ratio
        ),
        "candidate_to_budget_ratio_min_ge_floor": (
            args.min_candidate_to_budget_ratio <= 0
            or candidate_to_budget_ratio_min >= min_candidate_ratio_floor
        ),
        "selected_tce_removed_far_fp_recall_ge_floor": (
            selected_tce_removed_far_fp_recall >= min_selected_removed_recall
        ),
        "selected_detached_far_fp_recall_gt_0": (not is_region_gate) or selected_detached_far_fp_recall > 0,
        "selected_region_count_mean_gt_0": (not is_region_gate) or selected_region_count_mean > 0,
        "candidate_top20_removed_far_fp_recall_ge_0p40": candidate_top20_removed_recall >= 0.40,
        "candidate_top20_target_rate_le_0p15": candidate_top20_target_rate <= 0.15,
        "candidate_flat_background_ratio_le_0p20": candidate_flat_ratio <= 0.20,
        "selected_flat_background_ratio_le_0p20": selected_flat_ratio <= 0.20,
    }
    gate_failed_names = [key for key, value in gate.items() if not bool(value)]
    for key in gate_failed_names:
        if key not in failed_items:
            failed_items.append(key)
    ideal_gate = {
        "far_fp_instability_gt_target": gate["far_fp_instability_gt_target"],
        "far_fp_vs_target_auroc_ge_0p70": far_vs_target["auroc"] >= 0.70,
        "sps_tce_spearman_ge_0p30": float(spearman) >= 0.30,
        "candidate_top20_removed_far_fp_recall_ge_0p50": candidate_top20_removed_recall >= 0.50,
        "candidate_top20_target_rate_le_0p15": candidate_top20_target_rate <= 0.15,
        "candidate_flat_background_ratio_le_0p20": candidate_flat_ratio <= 0.20,
        "selected_flat_background_ratio_le_0p20": selected_flat_ratio <= 0.20,
    }
    gate["pass"] = all(gate.values())
    gate0_pass = bool(gate["pass"])
    ideal_gate["pass"] = all(ideal_gate.values())
    summary = {
        "current_stop_stage": None if gate0_pass else "Step3/Gate0",
        "stop_reason": "Gate0 passed" if gate0_pass else f"Gate0 failed: {', '.join(failed_items)}",
        "gate0_pass": gate0_pass,
        "failed_items": failed_items,
        "dataset": args.dataset_name,
        "train_dataset": train_dataset_name,
        "num_images": len(test_set),
        "image_list": str(Path(args.image_list).resolve()) if args.image_list else None,
        "model_name": args.model_name,
        "checkpoint_e400": str(Path(args.checkpoint_e400).resolve()),
        "tce_checkpoints": [str(Path(item).resolve()) for item in args.tce_checkpoints.split(",") if item.strip()],
        "threshold": args.threshold,
        "perturbation": args.perturbation,
        "select_target_preserving": args.select_target_preserving,
        "perturbation_pool": args.perturbation_pool if args.select_target_preserving else None,
        "selection_beta": args.selection_beta if args.select_target_preserving else None,
        "region_stats": region_stats,
        "far_fp_vs_target": far_vs_target,
        "near_fp_vs_target": near_vs_target,
        "tce_removed_far_fp_vs_target": removed_vs_target,
        "sps_tce_spearman": float(spearman),
        "top20_cutoff_target_removed_pool": top_cutoff,
        "top20_removed_far_fp_recall_mean": global_top20_removed_recall,
        "top20_target_rate_mean": global_top20_target_rate,
        "candidate_top20_cutoff": candidate_top_cutoff,
        "candidate_top20_removed_far_fp_recall": candidate_top20_removed_recall,
        "candidate_top20_target_rate": candidate_top20_target_rate,
        "candidate_instability_stats": stats(candidate_values),
        "candidate_hardness_top20_cutoff": hardness_top_cutoff,
        "candidate_hardness_top20_removed_far_fp_recall": hardness_top20_removed_recall,
        "candidate_hardness_top20_target_rate": hardness_top20_target_rate,
        "candidate_hardness_stats": stats(candidate_hardness),
        "candidate_sps_score_top20_cutoff": score_top_cutoff,
        "candidate_sps_score_top20_removed_far_fp_recall": score_top20_removed_recall,
        "candidate_sps_score_top20_target_rate": score_top20_target_rate,
        "candidate_sps_score_stats": stats(candidate_sps_score),
        "far_background_candidate_flat_ratio": candidate_flat_ratio,
        "sps_selected_flat_background_ratio": selected_flat_ratio,
        "budget_pixels_mean": budget_pixels_mean,
        "candidate_region_count_mean": candidate_region_count_mean,
        "selected_region_count_mean": selected_region_count_mean,
        "region_candidate_pixels_mean": region_candidate_pixels_mean,
        "num_images_with_candidate_empty": num_images_with_candidate_empty,
        "num_images_with_fallback": num_images_with_fallback,
        "num_images_with_candidate_under_budget": num_images_with_candidate_under_budget,
        "candidate_to_budget_ratio_mean": candidate_to_budget_ratio_mean,
        "candidate_to_budget_ratio_min": candidate_to_budget_ratio_min,
        "candidate_ohem_overlap_fraction": candidate_ohem_overlap_fraction,
        "selected_ohem_overlap_fraction": selected_ohem_overlap_fraction,
        "selected_target_leakage_pixels": selected_target_leakage_total,
        "selected_far_fp_component_coverage": selected_far_fp_component_coverage,
        "selected_tce_removed_far_fp_recall": selected_tce_removed_far_fp_recall,
        "selected_detached_far_fp_recall": selected_detached_far_fp_recall,
        "selected_flat_background_ratio": selected_flat_ratio,
        "sps_selected_removed_far_fp_recall": selected_tce_removed_far_fp_recall,
        "sps_selected_detached_far_fp_recall": selected_detached_far_fp_recall,
        "sps_selected_removed_fraction": float(selected_removed_total / max(1, selected_total)),
        "sps_selected_persistent_far_fp_fraction": float(selected_persistent_far_total / max(1, selected_total)),
        "sps_selected_ohem_overlap_fraction": selected_ohem_overlap_fraction,
        "ohem_top1_removed_far_fp_recall": ohem_removed_far_fp_recall,
        "ohem_topk_ratio": args.ohem_topk_ratio,
        "candidate_mode": args.candidate_mode,
        "candidate_tau": args.candidate_tau,
        "candidate_topk_ratio": args.candidate_topk_ratio,
        "candidate_topk_metric": args.candidate_topk_metric,
        "candidate_pool_metric": args.candidate_pool_metric,
        "rerank_signal_metric": args.rerank_signal_metric,
        "rerank_base_metric": args.rerank_base_metric,
        "min_candidate_to_budget_ratio": args.min_candidate_to_budget_ratio,
        "diagnostic_max_ohem_overlap": args.diagnostic_max_ohem_overlap,
        "candidate_min_metric": args.candidate_min_metric,
        "candidate_min_confidence": args.candidate_min_confidence,
        "candidate_fallback_topk_ratio": args.candidate_fallback_topk_ratio,
        "candidate_expand_radius": args.candidate_expand_radius,
        "candidate_expand_min_confidence": args.candidate_expand_min_confidence,
        "region_min_area": args.region_min_area,
        "region_max_area": args.region_max_area,
        "region_conf_min": args.region_conf_min,
        "region_signal_min": args.region_signal_min,
        "region_pool_topq": args.region_pool_topq,
        "region_score_metric": args.region_score_metric,
        "region_budget_fill_policy": args.region_budget_fill_policy,
        "peak_topk_ratio": args.peak_topk_ratio,
        "peak_nms_radius": args.peak_nms_radius,
        "peak_window_radius": args.peak_window_radius,
        "peak_min_conf": args.peak_min_conf,
        "peak_min_signal": args.peak_min_signal,
        "target_margin_quantile": args.target_margin_quantile,
        "target_margin_temp": args.target_margin_temp,
        "target_margin_min": args.target_margin_min,
        "flat_conf_tau": args.flat_conf_tau,
        "budget_q": args.budget_q,
        "kmax": args.kmax,
        "eta": args.eta,
        "per_image_top20_removed_far_fp_recall_mean": (
            float(np.mean(per_image_top20_removed_recalls)) if per_image_top20_removed_recalls else 0.0
        ),
        "per_image_top20_target_rate_mean": (
            float(np.mean(per_image_top20_target_rates)) if per_image_top20_target_rates else 0.0
        ),
        "gate": gate,
        "ideal_gate": ideal_gate,
    }
    (output_dir / "sps_perturbation_census_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "sps_perturbation_census_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_image_rows[0].keys()) if per_image_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)
    if per_region_rows:
        with (output_dir / "sps_perturbation_census_per_region.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(per_region_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_region_rows)
    print(json.dumps(summary["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
