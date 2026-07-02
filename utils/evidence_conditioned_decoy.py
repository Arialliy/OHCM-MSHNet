from __future__ import annotations

import torch
import torch.nn.functional as F

from probability import foreground_probability
from utils.pseudo_fp_generator import dilate_mask, sample_far_locations, validate_no_target_overlap


def build_blob_support(height, width, center, patch_radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cy, cx = center
    dist2 = (yy - int(cy)).pow(2) + (xx - int(cx)).pow(2)
    return (dist2 <= int(patch_radius) ** 2).view(1, 1, height, width)


def total_variation(x):
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def topk_mean(values, k):
    values = values.reshape(-1)
    if values.numel() == 0:
        return values.sum() * 0.0
    k = max(1, min(int(k), values.numel()))
    return torch.topk(values, k=k, largest=True).values.mean()


def evidence_logit(evidence_model, image):
    output = evidence_model.export_logits_features(image) if hasattr(evidence_model, "export_logits_features") else evidence_model(image)
    if isinstance(output, dict):
        return output.get("evidence_logit", output["logit"])
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def local_contrast_z(image, support):
    ring = dilate_mask(support, 3).float() - support.float()
    ring = (ring > 0).float()
    inside_pixels = support.float().sum()
    ring_pixels = ring.sum()
    if inside_pixels <= 0 or ring_pixels <= 0:
        return 0.0
    inside_mean = (image * support).sum() / (inside_pixels + 1e-6)
    ring_mean = (image * ring).sum() / (ring_pixels + 1e-6)
    ring_var = (((image - ring_mean) * ring).pow(2).sum() / (ring_pixels + 1e-6)).clamp_min(0.0)
    return float(((inside_mean - ring_mean) / (ring_var.sqrt() + 1e-6)).detach().cpu())


def generate_evidence_conditioned_decoy(
    image,
    gt_mask,
    evidence_model,
    *,
    center,
    patch_radius=5,
    steps=15,
    lr=0.1,
    target_dilate_radius=9,
    tv_weight=0.01,
    l2_weight=0.001,
    max_delta=0.5,
    response_threshold=0.5,
    min_gain=0.20,
    topk=8,
    clamp_range=None,
):
    if image.dim() != 4 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError("image must have shape 1x1xHxW")
    if gt_mask.shape[-2:] != image.shape[-2:]:
        gt_mask = F.interpolate(gt_mask.float(), size=image.shape[-2:], mode="nearest")

    _, _, height, width = image.shape
    device = image.device
    support = build_blob_support(height, width, center, patch_radius, device=device).to(image.dtype)
    safe, overlap = validate_no_target_overlap(support, gt_mask, target_dilate_radius)
    if not safe:
        return image.detach().clone(), support.detach(), {
            "accepted": False,
            "reject_reason": "target_dilate_overlap",
            "target_dilate_overlap_pixels": overlap,
            "prob_before_max": 0.0,
            "prob_after_max": 0.0,
            "prob_gain": 0.0,
            "area": float(support.sum().detach().cpu()),
            "contrast_z": 0.0,
            "residual": torch.zeros_like(image),
        }

    was_training = evidence_model.training
    evidence_model.eval()
    for param in evidence_model.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        before_prob = foreground_probability(evidence_logit(evidence_model, image))
        before_max = float((before_prob * support).amax().detach().cpu())

    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))

    for _ in range(int(steps)):
        residual = support * delta
        image_aug = image + residual
        if clamp_range is not None:
            lo, hi = clamp_range
            image_aug = torch.clamp(image_aug, float(lo), float(hi))
        z_e = evidence_logit(evidence_model, image_aug)
        support_eval = support if z_e.shape[-2:] == support.shape[-2:] else F.interpolate(support, size=z_e.shape[-2:], mode="nearest")
        response = topk_mean(z_e[support_eval.bool()], topk)
        reg_l2 = residual.pow(2).sum() / (support.sum() + 1e-6)
        loss = -response + float(tv_weight) * total_variation(residual) + float(l2_weight) * reg_l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-float(max_delta), float(max_delta))

    with torch.no_grad():
        residual = support * delta
        image_aug = image + residual
        if clamp_range is not None:
            lo, hi = clamp_range
            image_aug = torch.clamp(image_aug, float(lo), float(hi))
        after_prob = foreground_probability(evidence_logit(evidence_model, image_aug))
        support_eval = support if after_prob.shape[-2:] == support.shape[-2:] else F.interpolate(support, size=after_prob.shape[-2:], mode="nearest")
        after_values = after_prob[support_eval.bool()]
        after_max = float(after_values.max().detach().cpu()) if after_values.numel() else 0.0
        after_topk = float(topk_mean(after_values, topk).detach().cpu()) if after_values.numel() else 0.0
        gain = after_max - before_max
        contrast_z = local_contrast_z(image_aug, support)
        safe, overlap = validate_no_target_overlap(support, gt_mask, target_dilate_radius)
        accepted = bool(safe and (after_max >= response_threshold or after_topk >= response_threshold) and gain >= min_gain)

    if was_training:
        evidence_model.train()

    return image_aug.detach(), support.detach(), {
        "accepted": accepted,
        "reject_reason": "" if accepted else "response_or_gain",
        "target_dilate_overlap_pixels": int(overlap),
        "prob_before_max": before_max,
        "prob_after_max": after_max,
        "prob_after_topk": after_topk,
        "prob_gain": float(gain),
        "area": float(support.sum().detach().cpu()),
        "contrast_z": contrast_z,
        "residual": residual.detach(),
    }


def sample_safe_centers(gt_mask, num_centers, patch_radius=5, target_dilate_radius=9, generator=None):
    return sample_far_locations(
        gt_mask,
        num_centers,
        radius=target_dilate_radius,
        generator=generator,
        blob_radius=patch_radius,
    )
