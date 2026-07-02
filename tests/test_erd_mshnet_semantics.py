import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import ERDMSHNetLoss, dilate_mask, select_online_reliability_negatives
from model.ERD_MSHNet import ERDMSHNet, ERDMSHNetV3
from net import Net
from tools.official.build_reliability_labels import assert_train_split
from utils import apply_aug_ops, crop_by_coords


def test_erd_forward_shapes_and_gate_range():
    torch.manual_seed(1)
    model = ERDMSHNet(input_channels=1, rho=0.25, hidden_channels=8)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, gamma=1.0, return_feature=True)

    for key in ("masks", "evidence_logit", "reliability_logit", "final_logit", "gate", "feature"):
        assert key in out
    assert len(out["masks"]) == 4
    assert out["evidence_logit"].shape == (2, 1, 32, 32)
    assert out["reliability_logit"].shape == out["evidence_logit"].shape
    assert out["final_logit"].shape == out["evidence_logit"].shape
    assert out["gate"].shape == out["evidence_logit"].shape
    assert torch.min(out["gate"]).item() >= 0.25 - 1e-7
    assert torch.max(out["gate"]).item() <= 1.0 + 1e-7


def test_erd_gamma_zero_matches_evidence_logit():
    torch.manual_seed(2)
    model = ERDMSHNet(input_channels=1, rho=0.25, hidden_channels=8)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, gamma=0.0)

    assert torch.max(torch.abs(out["final_logit"] - out["evidence_logit"])).item() == 0.0


def test_erd_gate_is_suppress_only():
    torch.manual_seed(3)
    model = ERDMSHNet(input_channels=1, rho=0.25, hidden_channels=8)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, gamma=1.0)
        final_prob = torch.sigmoid(out["final_logit"])
        evidence_prob = torch.sigmoid(out["evidence_logit"])

    assert torch.max(final_prob - evidence_prob).item() <= 1e-7


def test_online_reliability_selector_is_dense_and_target_safe():
    evidence = torch.zeros(2, 1, 24, 24)
    gt = torch.zeros(2, 1, 24, 24)
    gt[:, :, 10:12, 10:12] = 1.0
    evidence[:, :, 0:5, 0:5] = 8.0
    evidence[:, :, 18:23, 18:23] = 6.0

    neg, counts = select_online_reliability_negatives(
        evidence,
        gt,
        far_radius=3,
        q=0.01,
        min_k=16,
        max_k=64,
    )
    target_dilate = dilate_mask(gt, 3)

    assert counts == [16, 16]
    assert int(neg.sum().item()) == 32
    assert int((neg * target_dilate).sum().item()) == 0


def test_erd_loss_backward_runs():
    torch.manual_seed(4)
    model = ERDMSHNet(input_channels=1, rho=0.25, hidden_channels=8)
    loss_fn = ERDMSHNetLoss(
        lambda_evidence=0.2,
        lambda_gate_pos=0.05,
        lambda_gate_neg=0.20,
        gate_start_epoch=1,
        gate_ramp_epochs=1,
        gate_neg_min_k=8,
        gate_neg_max_k=32,
    )
    x = torch.randn(2, 1, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 10:12, 10:12] = 1.0

    out = model(x, warm_flag=True, gamma=1.0)
    loss_out = loss_fn(out, target, epoch=5)
    loss_out["total"].backward()

    grads = [p.grad for p in model.reliability.parameters() if p.grad is not None]
    assert grads
    assert float(loss_out["gate_neg_per_image_min"]) >= 8.0


def test_erd_net_eval_returns_probability_and_export_uses_final_logit():
    torch.manual_seed(5)
    net = Net(
        model_name="ERDMSHNet",
        mode="test",
        loss_cfg={"mshnet_in_channels": 1, "erd_hidden_channels": 8, "erd_rho": 0.25},
    )
    net.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        pred = net(x)
        exported = net.export_logits_features(x)

    assert pred.min().item() >= 0.0
    assert pred.max().item() <= 1.0
    for key in ("logit", "target_logit", "clutter_logit", "reliability_logit", "feature", "masks", "gate"):
        assert key in exported
    assert torch.equal(exported["clutter_logit"], torch.zeros_like(exported["logit"]))
    assert exported["logit"].shape == exported["target_logit"].shape


def test_reliability_crop_alignment_helpers():
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    coords = (1, 4, 0, 3)
    cropped = crop_by_coords(arr, coords, patch_size=3)
    expected = arr[1:4, 0:3]
    assert np.array_equal(cropped, expected)

    ops = [1, 1, 1]
    aug = apply_aug_ops(cropped, ops)
    expected_aug = expected[::-1, :][:, ::-1].transpose(1, 0)
    assert np.array_equal(aug, expected_aug)


def test_no_reliability_labels_from_test_split():
    assert_train_split("/tmp/train_NUDT-SIRST.txt")
    for forbidden in (
        "/tmp/test_NUDT-SIRST.txt",
        "/tmp/hc-test_NUDT-SIRST.txt",
        "/tmp/blind_split.txt",
        "/tmp/external_split.txt",
    ):
        try:
            assert_train_split(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError("forbidden split should be rejected: %s" % forbidden)


def _force_head_constant(head, value):
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        head.net[-1].bias.fill_(float(value))


def test_erd_v3_forward_shapes():
    torch.manual_seed(11)
    model = ERDMSHNetV3(input_channels=1, aux_in_channels=16, hidden_channels=8, s_max=4.0)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_aux=True, return_feature=True)

    for key in (
        "logits",
        "final_logits",
        "evidence_logits",
        "protection_logits",
        "clutter_logits",
        "protection",
        "clutter",
        "suppression",
        "gate",
        "masks",
        "feature",
    ):
        assert key in out
    assert len(out["masks"]) == 4
    assert out["logits"].shape == (2, 1, 32, 32)
    assert out["evidence_logits"].shape == out["logits"].shape
    assert out["protection_logits"].shape == out["logits"].shape
    assert out["clutter_logits"].shape == out["logits"].shape


def test_erd_v3_suppress_only_logits():
    torch.manual_seed(12)
    model = ERDMSHNetV3(input_channels=1, aux_in_channels=16, hidden_channels=8, s_max=4.0)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_aux=True)

    assert torch.max(out["logits"] - out["evidence_logits"]).item() <= 1e-6


def test_erd_v3_smax_zero_equals_evidence():
    torch.manual_seed(13)
    model = ERDMSHNetV3(input_channels=1, aux_in_channels=16, hidden_channels=8, s_max=0.0)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_aux=True)

    assert torch.allclose(out["logits"], out["evidence_logits"], atol=1e-7)
    assert torch.max(torch.abs(out["suppression"])).item() == 0.0


def test_erd_v3_protection_one_blocks_suppression():
    torch.manual_seed(14)
    model = ERDMSHNetV3(input_channels=1, aux_in_channels=16, hidden_channels=8, s_max=4.0)
    model.eval()
    _force_head_constant(model.protection_head, 20.0)
    _force_head_constant(model.clutter_head, 20.0)
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_aux=True)

    assert torch.allclose(out["logits"], out["evidence_logits"], atol=1e-6)
    assert torch.max(out["suppression"]).item() <= 1e-6


def test_erd_v3_clutter_zero_blocks_suppression():
    torch.manual_seed(15)
    model = ERDMSHNetV3(input_channels=1, aux_in_channels=16, hidden_channels=8, s_max=4.0)
    model.eval()
    _force_head_constant(model.protection_head, -20.0)
    _force_head_constant(model.clutter_head, -20.0)
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, warm_flag=True, return_aux=True)

    assert torch.allclose(out["logits"], out["evidence_logits"], atol=1e-6)
    assert torch.max(out["suppression"]).item() <= 1e-6


def test_erd_v3_no_gt_used_in_forward():
    torch.manual_seed(16)
    model = ERDMSHNetV3(input_channels=1, aux_in_channels=16, hidden_channels=8, s_max=4.0)
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out1 = model(x, warm_flag=True, return_aux=True)["logits"]
        out2 = model(x, warm_flag=True, return_aux=True)["logits"]

    assert torch.allclose(out1, out2, atol=0.0)


def test_erd_v3_net_eval_and_export_aux_keys():
    torch.manual_seed(17)
    net = Net(
        model_name="ERDMSHNetV3",
        mode="test",
        loss_cfg={"mshnet_in_channels": 1, "erd_aux_in_channels": 16, "erd_hidden_channels": 8},
    )
    net.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        pred = net(x)
        exported = net.export_logits_features(x)

    assert pred.min().item() >= 0.0
    assert pred.max().item() <= 1.0
    for key in (
        "logit",
        "target_logit",
        "clutter_logit",
        "protection_logit",
        "feature",
        "masks",
        "gate",
        "suppression",
        "protection",
        "clutter",
    ):
        assert key in exported
    assert exported["logit"].shape == exported["target_logit"].shape


def test_erd_v3_pretrained_mapping_touches_only_evidence_branch():
    torch.manual_seed(18)
    source = Net(
        model_name="MSHNetOHEM",
        mode="train",
        loss_cfg={"mshnet_in_channels": 1, "mshnet_warm_epoch": 5},
    )
    target = Net(
        model_name="ERDMSHNetV3",
        mode="train",
        loss_cfg={"mshnet_in_channels": 1, "erd_aux_in_channels": 16, "erd_hidden_channels": 8},
    )
    before_protection = {
        key: value.detach().clone()
        for key, value in target.state_dict().items()
        if key.startswith("model.protection_head.") or key.startswith("model.clutter_head.")
    }
    mapped = {}
    for key, value in source.state_dict().items():
        if key.startswith("model."):
            mapped["model.evidence_net." + key[len("model."):]] = value
    target.load_state_dict(mapped, strict=False)

    after = target.state_dict()
    assert before_protection
    for key, value in before_protection.items():
        assert torch.equal(after[key], value)
