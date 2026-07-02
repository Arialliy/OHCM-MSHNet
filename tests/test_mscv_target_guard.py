import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import MSCVLoss


def make_output(validity_value):
    target = torch.zeros(1, 1, 16, 16)
    masks = [torch.zeros_like(target) for _ in range(4)]
    return {
        "masks": masks,
        "evidence_logit": torch.zeros_like(target),
        "final_logit": torch.zeros_like(target),
        "validity_logit": torch.full_like(target, float(validity_value)),
        "p_max": torch.zeros_like(target),
        "p_std": torch.zeros_like(target),
        "local_contrast": torch.ones_like(target),
        "suppression_map": torch.zeros_like(target),
        "beta": torch.tensor(0.0),
    }


def test_mscv_target_guard_penalizes_low_target_validity():
    loss_fn = MSCVLoss(mshnet_warm_epoch=0, lambda_valid=0.0, lambda_keep=0.0, lambda_suppress=0.0)
    gt = torch.zeros(1, 1, 16, 16)
    gt[:, :, 7:9, 7:9] = 1.0

    low = loss_fn(make_output(-4.0), gt, epoch=1)["target_guard_loss"]
    high = loss_fn(make_output(4.0), gt, epoch=1)["target_guard_loss"]

    assert float(low) > float(high)
