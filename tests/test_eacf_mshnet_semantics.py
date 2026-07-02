import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.EACF_MSHNet import EACFMSHNet
from net import Net


def test_eacf_eta_zero_equals_base():
    model = EACFMSHNet(input_channels=1)
    model.eval()
    model.fusion.eta.data.zero_()
    x = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)

    diff = (out["final_logit"] - out["base_logit"]).abs().max().item()
    assert diff < 1e-7


def test_eacf_scale_weights_are_convex():
    model = EACFMSHNet(input_channels=1)
    model.eval()
    x = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)

    weights = out["scale_weights"]
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, :1]), atol=1e-6)


def test_eacf_freeze_backbone():
    model = EACFMSHNet(input_channels=1)
    model.freeze_backbone()
    model.train()

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.fusion.parameters())
    assert not model.backbone.training


def test_eacf_eval_returns_probability():
    net = Net("EACFMSHNet", mode="test", loss_cfg={"mshnet_in_channels": 1})
    net.eval()
    x = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        prob = net(x, epoch=999)

    assert prob.min().item() >= 0.0
    assert prob.max().item() <= 1.0


def test_eacf_training_loss_returns_total():
    net = Net(
        "EACFMSHNet",
        mode="train",
        loss_cfg={"mshnet_in_channels": 1, "mshnet_warm_epoch": 0},
    )
    net.train()
    x = torch.randn(2, 1, 64, 64)
    gt = torch.zeros(2, 1, 64, 64)
    gt[:, :, 30:33, 30:33] = 1.0

    out = net(x, epoch=1)
    loss_out = net.loss(out, gt, epoch=1)

    assert "total" in loss_out
    assert torch.isfinite(loss_out["total"])
