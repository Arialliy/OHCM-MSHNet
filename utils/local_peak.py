import torch
import torch.nn.functional as F


def dilate_binary(mask, radius):
    if radius <= 0:
        return mask.float()
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(radius))


def local_peak_mask(prob, kernel_size=5):
    pooled = F.max_pool2d(prob, kernel_size, stride=1, padding=kernel_size // 2)
    return prob.eq(pooled)


def select_background_peaks(logits, target, topk_ratio=0.001, min_k=8, max_k=256, dilate_radius=3):
    prob = torch.sigmoid(logits.detach())
    target_dilated = dilate_binary(target.float(), radius=dilate_radius)
    far_bg = target_dilated < 0.5
    peaks = local_peak_mask(prob) & far_bg
    score = prob.masked_fill(~peaks, -1.0)

    b, _, h, w = score.shape
    k = int(max(min_k, min(max_k, topk_ratio * h * w)))
    k = min(k, h * w)
    flat = score.view(b, -1)
    vals, idx = torch.topk(flat, k=k, dim=1)
    selected = torch.zeros_like(flat, dtype=torch.bool)
    selected.scatter_(1, idx, vals > 0)
    return selected.view(b, 1, h, w)
