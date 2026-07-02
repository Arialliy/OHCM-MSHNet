import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.component_geometry import build_center_heatmap, build_core_boundary_maps, component_area_bins
from utils.local_peak import select_background_peaks


def sample_mask():
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 5:7, 5:7] = 1.0
    mask[:, :, 20:24, 18:23] = 1.0
    return mask


def test_center_heatmap_has_peak_for_each_component():
    center, _scale_map, _valid = build_center_heatmap(sample_mask())
    assert int((center == 1.0).sum().item()) >= 2


def test_center_heatmap_values_in_0_1():
    center, _scale_map, _valid = build_center_heatmap(sample_mask())
    assert center.min().item() >= 0.0
    assert center.max().item() <= 1.0


def test_scale_bins_not_all_zero():
    _target, _valid, counts = component_area_bins(sample_mask(), bins=(4, 9, 16, 36))
    assert int(counts.sum().item()) == 2
    assert int((counts > 0).sum().item()) >= 1


def test_boundary_ring_does_not_overlap_target_core():
    mask = sample_mask()
    core, boundary, _ignore = build_core_boundary_maps(mask, boundary_radius=2)
    assert int((core.bool() & boundary.bool()).sum().item()) == 0
    assert int(boundary.sum().item()) > 0


def test_local_peak_selector_returns_fixed_budget():
    logits = torch.randn(1, 1, 32, 32)
    target = sample_mask()
    peaks = select_background_peaks(logits, target, topk_ratio=0.01, min_k=8, max_k=8, dilate_radius=2)
    assert int(peaks.sum().item()) == 8


def test_local_peak_selector_ignores_target_dilation():
    logits = torch.randn(1, 1, 32, 32)
    logits[:, :, 5:7, 5:7] = 20.0
    target = sample_mask()
    peaks = select_background_peaks(logits, target, topk_ratio=0.01, min_k=8, max_k=8, dilate_radius=3)
    target_dilated = torch.nn.functional.max_pool2d(target, kernel_size=7, stride=1, padding=3) > 0.5
    assert int((peaks & target_dilated).sum().item()) == 0
