import torch
import torch.nn.functional as F


def _ensure_bchw(mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape HxW, BxHxW, or Bx1xHxW")
    return mask


def _compatible_generator(generator, device):
    if generator is None:
        return None
    gen_device = getattr(generator, "device", torch.device("cpu"))
    if torch.device(gen_device).type != torch.device(device).type:
        return None
    return generator


def dilate_mask(mask, radius):
    mask = _ensure_bchw(mask).float()
    radius = int(radius)
    if radius <= 0:
        return mask > 0
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius) > 0


def build_safe_center_mask(gt_mask, blob_radius, target_dilate_radius):
    total_radius = int(blob_radius) + int(target_dilate_radius)
    unsafe = dilate_mask(gt_mask, total_radius)
    return ~unsafe


def validate_no_target_overlap(pseudo_mask, gt_mask, target_dilate_radius):
    pseudo_mask = _ensure_bchw(pseudo_mask).bool()
    gt_mask = _ensure_bchw(gt_mask).to(device=pseudo_mask.device)
    if gt_mask.shape[-2:] != pseudo_mask.shape[-2:]:
        gt_mask = F.interpolate(gt_mask.float(), size=pseudo_mask.shape[-2:], mode="nearest") > 0
    target_dilate = dilate_mask(gt_mask, target_dilate_radius)
    overlap = (pseudo_mask & target_dilate).sum().item()
    return overlap == 0, int(overlap)


def sample_far_locations(mask, num_samples, radius, generator=None, blob_radius=0):
    mask = _ensure_bchw(mask)
    if mask.shape[0] != 1:
        raise ValueError("sample_far_locations expects a single-image mask")
    num_samples = int(num_samples)
    if num_samples <= 0:
        return torch.empty((0, 2), device=mask.device, dtype=torch.long)

    far = build_safe_center_mask(mask, blob_radius=blob_radius, target_dilate_radius=radius)
    coords = torch.nonzero(far[0, 0], as_tuple=False)
    if coords.numel() == 0:
        return torch.empty((0, 2), device=mask.device, dtype=torch.long)
    if coords.shape[0] <= num_samples:
        return coords

    generator = _compatible_generator(generator, mask.device)
    perm = torch.randperm(coords.shape[0], device=mask.device, generator=generator)
    return coords[perm[:num_samples]]


def generate_gaussian_blob(height, width, center, sigma, amplitude, device):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cy, cx = center
    sigma = max(float(sigma), 1e-6)
    return float(amplitude) * torch.exp(
        -((yy - int(cy)) ** 2 + (xx - int(cx)) ** 2) / (2.0 * sigma ** 2)
    )


def generate_pseudo_fp_batch(
    images,
    masks,
    *,
    num_blobs_per_image=3,
    far_radius=9,
    amplitude_range=(0.10, 0.50),
    sigma_range=(0.8, 2.5),
    mode="gaussian",
    target_dilate_radius=None,
    blob_radius=None,
    clamp_range=None,
    generator=None,
):
    if images.dim() != 4:
        raise ValueError("images must have shape BxCxHxW")
    masks = _ensure_bchw(masks).to(device=images.device)
    if masks.shape[0] != images.shape[0]:
        raise ValueError("images and masks must have the same batch size")
    if masks.shape[-2:] != images.shape[-2:]:
        masks = F.interpolate(masks.float(), size=images.shape[-2:], mode="nearest") > 0

    batch_size, _, height, width = images.shape
    device = images.device
    generator = _compatible_generator(generator, device)
    images_aug = images.clone()
    pseudo_fp_mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=images.dtype)
    target_dilate_radius = far_radius if target_dilate_radius is None else int(target_dilate_radius)
    if blob_radius is None:
        blob_radius = int(torch.ceil(torch.tensor(float(max(sigma_range)) * 2.0)).item())

    if mode not in {"gaussian", "residual", "mixed"}:
        raise ValueError("mode must be gaussian, residual, or mixed")

    for batch_idx in range(batch_size):
        coords = sample_far_locations(
            masks[batch_idx:batch_idx + 1],
            num_blobs_per_image,
            target_dilate_radius,
            generator=generator,
            blob_radius=blob_radius,
        )
        for coord in coords:
            cy, cx = int(coord[0].item()), int(coord[1].item())
            sigma = torch.empty(1, device=device).uniform_(*sigma_range, generator=generator).item()
            amp = torch.empty(1, device=device).uniform_(*amplitude_range, generator=generator).item()
            blob = generate_gaussian_blob(height, width, (cy, cx), sigma, amp, device)
            if mode in ("residual", "mixed"):
                blob = blob * (1.0 + 0.12 * torch.randn_like(blob)).clamp(0.65, 1.35)
            images_aug[batch_idx, 0] = images_aug[batch_idx, 0] + blob.to(images.dtype)
            support = (blob > amp * 0.3).to(images.dtype)
            pseudo_fp_mask[batch_idx, 0] = torch.maximum(pseudo_fp_mask[batch_idx, 0], support)

        ok, _ = validate_no_target_overlap(
            pseudo_fp_mask[batch_idx:batch_idx + 1],
            masks[batch_idx:batch_idx + 1],
            target_dilate_radius,
        )
        if not ok:
            safe = build_safe_center_mask(
                masks[batch_idx:batch_idx + 1],
                blob_radius=0,
                target_dilate_radius=target_dilate_radius,
            ).to(device=device)
            pseudo_fp_mask[batch_idx:batch_idx + 1] *= safe.float()

    if clamp_range is not None:
        lo, hi = clamp_range
        images_aug = torch.clamp(images_aug, float(lo), float(hi))

    valid_mask = pseudo_fp_mask.sum(dim=(1, 2, 3)) > 0
    return images_aug, pseudo_fp_mask, valid_mask
