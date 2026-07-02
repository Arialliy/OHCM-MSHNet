import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.mscv_candidate import build_mscv_candidate_mask


def test_mscv_candidate_no_target_leakage():
    gt = torch.zeros(1, 1, 32, 32)
    gt[:, :, 14:18, 14:18] = 1.0
    p_max = torch.zeros_like(gt)
    p_std = torch.zeros_like(gt)
    local_contrast = torch.ones_like(gt)
    p_max[:, :, 2:6, 2:6] = 0.8
    p_std[:, :, 2:6, 2:6] = 0.2
    p_max[:, :, 15:17, 15:17] = 0.9
    p_std[:, :, 15:17, 15:17] = 0.3

    masks = build_mscv_candidate_mask(
        p_max,
        p_std,
        gt,
        far_radius=4,
        candidate_prob_thr=0.2,
        candidate_std_thr=0.05,
        local_contrast=local_contrast,
        nonflat_thr=0.05,
    )

    candidate = masks["candidate"].bool()
    target_near = masks["target_near"].bool()
    assert int((candidate & target_near).sum().item()) == 0
    assert int(candidate.sum().item()) > 0
