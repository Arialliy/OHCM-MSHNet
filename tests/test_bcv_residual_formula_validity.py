import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.BCV_MSHNet import BCVMSHNet
from net import Net


class DummyEvidence(torch.nn.Module):
    def forward(self, x, warm_flag=True, return_feature=False):
        evidence = torch.ones_like(x[:, :1])
        masks = [evidence, evidence, evidence, evidence]
        feature = torch.cat([x[:, :1], x[:, :1]], dim=1)
        if return_feature:
            return masks, evidence, feature
        return masks, evidence


class ZeroBackground(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x[:, :1])


def test_bcv_residual_formula_validity_matches_residual_threshold_formula():
    model = BCVMSHNet(
        input_channels=1,
        hidden_channels=4,
        beta_max=0.2,
        validity_mode="residual_formula",
        residual_theta=1.0,
        residual_temp=0.5,
    )
    model.evidence_net = DummyEvidence()
    model.bg_branch = ZeroBackground()
    model.eval()
    x = torch.tensor([[[[0.0, 2.0], [2.0, 0.0]]]])

    with torch.no_grad():
        out = model(x, beta=0.1, return_dict=True)

    expected = torch.sigmoid((out["residual_norm"] - 1.0) / 0.5)
    assert torch.allclose(out["validity_prob"], expected, atol=1e-6)
    assert torch.allclose(out["validity_logit"], torch.logit(expected.clamp(1e-4, 1 - 1e-4)), atol=1e-6)


def test_net_passes_bcv_residual_formula_config():
    net = Net(
        "BCVMSHNet",
        mode="test",
        loss_cfg={
            "mshnet_in_channels": 1,
            "bcv_hidden_channels": 4,
            "bcv_validity_mode": "residual_formula",
            "bcv_residual_theta": 0.7,
            "bcv_residual_temp": 0.3,
        },
    )

    assert net.model.validity_mode == "residual_formula"
    assert net.model.residual_theta == 0.7
    assert net.model.residual_temp == 0.3
