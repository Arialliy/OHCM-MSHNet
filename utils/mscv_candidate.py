from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_mask(mask: torch.Tensor, size) -> torch.Tensor:
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="nearest")
    return mask


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = mask.float()
    if int(radius) <= 0:
        return (mask > 0).float()
    kernel = 2 * int(radius) + 1
    return (F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=int(radius)) > 0).float()


def build_mscv_candidate_mask(
    p_max: torch.Tensor,
    p_std: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    far_radius: int = 7,
    candidate_prob_thr: float = 0.2,
    candidate_std_thr: float = 0.05,
    local_contrast: torch.Tensor | None = None,
    nonflat_thr: float = 0.05,
):
    target = resize_mask(gt_mask, p_max.shape[-2:])
    target_near = dilate_mask(target, far_radius)
    far_mask = (target_near <= 0).float()
    base_candidate = (
        (far_mask > 0.5)
        & (p_max.detach() > float(candidate_prob_thr))
        & (p_std.detach() > float(candidate_std_thr))
    )
    if local_contrast is None:
        nonflat_mask = torch.ones_like(base_candidate, dtype=torch.bool)
    else:
        local_contrast = resize_mask(local_contrast, p_max.shape[-2:])
        nonflat_mask = local_contrast.detach() > float(nonflat_thr)
    candidate = base_candidate & nonflat_mask
    return {
        "candidate": candidate.float(),
        "base_candidate": base_candidate.float(),
        "nonflat_mask": nonflat_mask.float(),
        "far_mask": far_mask,
        "target_near": target_near,
        "target": target,
    }


def topk_mask(score: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros_like(score, dtype=torch.bool)
    if int(k) <= 0:
        return out
    flat_score = score.reshape(score.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1).bool()
    flat_out = out.reshape(out.shape[0], -1)
    for b in range(score.shape[0]):
        valid_idx = torch.nonzero(flat_valid[b], as_tuple=False).flatten()
        if valid_idx.numel() < 1:
            continue
        kk = min(int(k), int(valid_idx.numel()))
        top_idx = torch.topk(flat_score[b, valid_idx], k=kk, largest=True).indices
        flat_out[b, valid_idx[top_idx]] = True
    return out
