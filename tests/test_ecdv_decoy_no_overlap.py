import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evidence_conditioned_decoy import build_blob_support, sample_safe_centers
from utils.pseudo_fp_generator import validate_no_target_overlap


def test_decoy_no_target_dilate_overlap():
    target = torch.zeros(1, 1, 48, 48)
    target[:, :, 20:26, 20:26] = 1.0
    centers = sample_safe_centers(
        target,
        num_centers=8,
        patch_radius=4,
        target_dilate_radius=7,
        generator=torch.Generator().manual_seed(7),
    )

    assert centers.shape[0] > 0
    for center in centers:
        pseudo_mask = build_blob_support(48, 48, (int(center[0]), int(center[1])), patch_radius=4, device=target.device)
        ok, overlap = validate_no_target_overlap(pseudo_mask, target, target_dilate_radius=7)
        assert ok is True
        assert overlap == 0
