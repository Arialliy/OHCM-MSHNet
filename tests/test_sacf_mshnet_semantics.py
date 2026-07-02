import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import SACFMSHNetLoss
from model.SACF_MSHNet import SACFMSHNet
from net import Net


def test_sacf_weights_sum_to_one():
    model = SACFMSHNet(input_channels=1)
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)
    weights = out["fusion_weights"]
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, :1]), atol=1e-6)


def test_sacf_delta_is_bounded():
    model = SACFMSHNet(input_channels=1, delta_max=0.25)
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)
    assert out["fusion_delta"].abs().max().item() <= 0.25 + 1e-6


def test_sacf_gate_is_local_not_global_eta():
    model = SACFMSHNet(input_channels=1)
    names = dict(model.named_parameters())
    assert not any(name.endswith("eta") or ".eta" in name for name in names)
    x = torch.randn(2, 1, 64, 64)
    out = model(x, warm_flag=True, return_dict=True)
    gate = out["fusion_gate"]
    assert gate.shape[-2:] == x.shape[-2:]
    assert gate.shape[1] == 1
    assert gate.min().item() >= 0.0
    assert gate.max().item() <= 1.0


def test_sacf_loss_uses_final_logits():
    loss_fn = SACFMSHNetLoss(
        mshnet_warm_epoch=0,
        lambda_anchor=0.0,
        lambda_scale=0.0,
        lambda_disagree_bg=0.0,
    )
    target = torch.ones(2, 1, 16, 16)
    base = torch.zeros_like(target)
    scale_logits = [torch.zeros_like(target) for _ in range(4)]
    out_a = {
        "final_logits": torch.zeros_like(target),
        "base_logits": base,
        "scale_logits": scale_logits,
        "masks": [],
        "fusion_gate": torch.ones_like(target) * 0.1,
        "fusion_delta": torch.zeros_like(target),
    }
    out_b = {**out_a, "final_logits": torch.ones_like(target)}

    loss_a = loss_fn(out_a, target, epoch=1)["total"]
    loss_b = loss_fn(out_b, target, epoch=1)["total"]

    assert not torch.allclose(loss_a, loss_b)


def test_sacf_trainable_params_include_fusion_when_evidence_frozen():
    model = SACFMSHNet(input_channels=1)
    model.freeze_evidence()
    model.train()

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert any("fusion" in name for name in trainable)
    assert not any("evidence_net" in name for name in trainable)
    assert not model.evidence_net.training


def test_sacf_checkpoint_contains_fusion_keys(tmp_path):
    net = Net("SACFMSHNet", mode="train", loss_cfg={"mshnet_in_channels": 1})
    ckpt_path = tmp_path / "sacf.pth.tar"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "trainable_parameter_names": [
                name for name, param in net.named_parameters() if param.requires_grad
            ],
        },
        ckpt_path,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    keys = ckpt["state_dict"].keys()
    assert any("fusion" in key for key in keys)
    assert any("fusion" in name for name in ckpt["trainable_parameter_names"])


def test_sacf_activation_summary_blocks_identity_collapse(tmp_path):
    summary = {
        "gate_pass": True,
        "mean_abs_final_minus_base_prob": 0.0,
        "changed_pixel_ratio_at_0p5": 0.0,
        "fusion_gate_mean": 0.1,
        "fusion_delta_abs_mean": 0.01,
        "checkpoint_has_fusion_keys": True,
        "optimizer_has_fusion_params": True,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_sacf_ready.py"),
            "--activation_summary",
            str(summary_path),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "final_equals_base_identity_collapse" in result.stdout


def test_sacf_output_head_base_and_final_are_selectable():
    net = Net("SACFMSHNet", mode="test", loss_cfg={"mshnet_in_channels": 1})
    net.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        base = net(x, epoch=999, output_head="base")
        final = net(x, epoch=999, output_head="final")
    assert base.shape == final.shape
    assert base.min().item() >= 0.0
    assert final.max().item() <= 1.0
