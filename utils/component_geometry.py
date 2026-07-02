import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure


def _as_binary_numpy(mask_2d):
    return (mask_2d.detach().cpu().numpy() > 0.5).astype(np.uint8)


def _component_regions(mask_2d):
    labels = measure.label(mask_2d.astype(np.uint8), connectivity=2)
    return measure.regionprops(labels)


def dilate_binary(mask, radius):
    if radius <= 0:
        return mask.float()
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(radius))


def build_center_heatmap(mask, sigma_min=1.0, sigma_scale=0.35):
    """Build Gaussian center heatmap and log-area map from binary target masks."""
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W]")

    device = mask.device
    dtype = mask.dtype
    b, _, h, w = mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    center = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    scale_map = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    valid = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    for bi in range(b):
        binary = _as_binary_numpy(mask[bi, 0])
        for region in _component_regions(binary):
            coords = torch.as_tensor(region.coords, device=device, dtype=torch.long)
            cy, cx = region.centroid
            area = float(region.area)
            sigma = max(float(sigma_min), float(sigma_scale) * math.sqrt(area))
            gaussian = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
            center[bi, 0] = torch.maximum(center[bi, 0], gaussian)
            peak_y = int(max(0, min(h - 1, round(cy))))
            peak_x = int(max(0, min(w - 1, round(cx))))
            center[bi, 0, peak_y, peak_x] = 1.0
            scale_value = math.log(area + 1.0)
            scale_map[bi, 0, coords[:, 0], coords[:, 1]] = scale_value
            valid[bi, 0, coords[:, 0], coords[:, 1]] = 1.0

    return center.clamp_(0.0, 1.0), scale_map, valid


def build_core_boundary_maps(mask, core_radius=1, boundary_radius=2):
    """Build target core and boundary ring maps."""
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W]")
    core = mask.float().clamp(0.0, 1.0)
    dilated = dilate_binary(core, boundary_radius).clamp(0.0, 1.0)
    boundary = (dilated - core).clamp(0.0, 1.0)
    ignore = (dilate_binary(core, boundary_radius + max(1, core_radius)) - dilated).clamp(0.0, 1.0)
    return core, boundary, ignore


def component_area_bins(mask, bins=(4, 9, 16, 36)):
    """Assign each component pixel to the nearest log-area bin center."""
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W]")

    device = mask.device
    b, _, h, w = mask.shape
    bin_centers = torch.as_tensor(bins, device=device, dtype=torch.float32).clamp_min(1.0)
    log_centers = torch.log(bin_centers)
    target = torch.zeros((b, h, w), device=device, dtype=torch.long)
    valid = torch.zeros((b, 1, h, w), device=device, dtype=torch.bool)
    counts = torch.zeros((len(bins),), device=device, dtype=torch.long)

    for bi in range(b):
        binary = _as_binary_numpy(mask[bi, 0])
        for region in _component_regions(binary):
            coords = torch.as_tensor(region.coords, device=device, dtype=torch.long)
            log_area = torch.log(torch.tensor(float(region.area), device=device).clamp_min(1.0))
            cls = int(torch.argmin(torch.abs(log_area - log_centers)).item())
            target[bi, coords[:, 0], coords[:, 1]] = cls
            valid[bi, 0, coords[:, 0], coords[:, 1]] = True
            counts[cls] += 1

    return target, valid, counts
