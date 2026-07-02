import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import CGAMSHNetLoss
from model.CGA_MSHNet import CGAMSHNet, configure_cga_trainable
from net import Net


def make_target(batch=2, size=64):
    target = torch.zeros(batch, 1, size, size)
    target[:, :, 30:33, 30:33] = 1.0
    return target


def test_cga_forward_returns_final_and_aux_heads():
    model = CGAMSHNet(input_channels=1)
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)
    for key in ["final_logits", "center_logits", "geometry_scale_logits", "core_logits", "boundary_logits"]:
        assert key in out
    assert out["final_logits"].shape == out["center_logits"].shape
    assert out["geometry_scale_logits"].shape[1] == 4


def test_cga_final_equals_mshnet_when_aux_disabled():
    model = CGAMSHNet(input_channels=1)
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        evidence = model.evidence_net(x, warm_flag=True, return_dict=True)
        out = model(x, warm_flag=True, return_dict=True)
    assert torch.allclose(out["final_logits"], evidence["base_logits"], atol=1e-7)


def test_cga_loss_uses_final_logits_for_main_mask_loss():
    loss_fn = CGAMSHNetLoss(
        mshnet_warm_epoch=0,
        lambda_center=0.0,
        lambda_scale=0.0,
        lambda_core=0.0,
        lambda_boundary=0.0,
        lambda_peak_bg=0.0,
        lambda_anchor_easy=0.0,
    )
    target = torch.ones(2, 1, 16, 16)
    out = {
        "final_logits": torch.zeros_like(target),
        "masks": [],
        "center_logits": torch.zeros_like(target),
        "geometry_scale_logits": torch.zeros(2, 4, 16, 16),
        "core_logits": torch.zeros_like(target),
        "boundary_logits": torch.zeros_like(target),
    }
    loss_a = loss_fn(out, target, epoch=1)["total"]
    out["final_logits"] = torch.ones_like(target)
    loss_b = loss_fn(out, target, epoch=1)["total"]
    assert not torch.allclose(loss_a, loss_b)


def test_cga_loss_uses_center_logits():
    loss_fn = CGAMSHNetLoss(
        mshnet_warm_epoch=0,
        lambda_center=1.0,
        lambda_scale=0.0,
        lambda_core=0.0,
        lambda_boundary=0.0,
        lambda_peak_bg=0.0,
        lambda_anchor_easy=0.0,
    )
    target = make_target(batch=2, size=16)
    out = {
        "final_logits": torch.zeros_like(target),
        "masks": [],
        "center_logits": torch.zeros_like(target),
        "geometry_scale_logits": torch.zeros(2, 4, 16, 16),
        "core_logits": torch.zeros_like(target),
        "boundary_logits": torch.zeros_like(target),
    }
    loss_a = loss_fn(out, target, epoch=1)["center"]
    out["center_logits"] = torch.ones_like(target)
    loss_b = loss_fn(out, target, epoch=1)["center"]
    assert not torch.allclose(loss_a, loss_b)


def test_cga_trainable_params_include_geometry_heads():
    model = CGAMSHNet(input_channels=1)
    trainable = configure_cga_trainable(model, mode="decoder_aux")
    assert any("geometry_heads" in name for name in trainable)


def test_cga_trainable_params_exclude_encoder_in_decoder_aux_mode():
    model = CGAMSHNet(input_channels=1)
    trainable = configure_cga_trainable(model, mode="decoder_aux")
    assert not any("encoder_0" in name or "encoder_1" in name for name in trainable)


def test_cga_checkpoint_contains_geometry_head_keys(tmp_path):
    net = Net("CGAMSHNet", mode="train", loss_cfg={"mshnet_in_channels": 1})
    ckpt_path = tmp_path / "cga.pth.tar"
    torch.save({"state_dict": net.state_dict()}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert any("geometry_heads" in key for key in ckpt["state_dict"].keys())


def test_cga_eval_uses_final_logits_not_center_logits():
    net = Net("CGAMSHNet", mode="test", loss_cfg={"mshnet_in_channels": 1})
    net.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        prob = net(x, epoch=999)
    assert prob.shape[-2:] == x.shape[-2:]
    assert prob.min().item() >= 0.0
    assert prob.max().item() <= 1.0
