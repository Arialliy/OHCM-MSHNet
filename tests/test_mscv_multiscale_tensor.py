import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.MSCV_MSHNet import MSCVMSHNet


def test_mscv_multiscale_tensor_shape():
    model = MSCVMSHNet(input_channels=1, hidden_channels=4)
    evidence = torch.randn(2, 1, 32, 32)
    masks = [
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 1, 16, 16),
        torch.randn(2, 1, 8, 8),
        torch.randn(2, 1, 4, 4),
    ]
    local_contrast = torch.randn(2, 1, 32, 32)

    tensor, stats = model.build_multiscale_tensor(masks, evidence, local_contrast)

    assert tensor.shape == (2, 10, 32, 32)
    assert stats["p_mean"].shape == evidence.shape
    assert stats["p_std"].shape == evidence.shape
    assert stats["p_min"].shape == evidence.shape
    assert stats["p_max"].shape == evidence.shape
