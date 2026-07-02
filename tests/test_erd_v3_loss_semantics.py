import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import ERDMSHNetV3Loss, binary_dilate


def _fake_outputs(z_e, z_f=None, z_t=None, z_c=None):
    if z_f is None:
        z_f = z_e.clone()
    if z_t is None:
        z_t = torch.zeros_like(z_e)
    if z_c is None:
        z_c = torch.zeros_like(z_e)
    return {
        "logits": z_f,
        "final_logits": z_f,
        "evidence_logits": z_e,
        "protection_logits": z_t,
        "clutter_logits": z_c,
        "protection": torch.sigmoid(z_t),
        "clutter": torch.sigmoid(z_c),
        "suppression": torch.clamp(z_e - z_f, min=0.0),
        "masks": [],
    }


def test_erd_v3_loss_selects_far_background_only():
    loss_fn = ERDMSHNetV3Loss(far_radius=3, neg_topk_ratio=0.05)
    evidence = torch.zeros(2, 1, 24, 24)
    target = torch.zeros(2, 1, 24, 24)
    target[:, :, 10:12, 10:12] = 1.0
    evidence[:, :, 0:5, 0:5] = 8.0
    evidence[:, :, 18:23, 18:23] = 6.0

    neg, counts = loss_fn.select_online_negatives(evidence, target)
    target_dilate = binary_dilate(target, 3)

    assert min(counts) > 0
    assert int(neg.sum().item()) == sum(counts)
    assert int((neg.float() * target_dilate).sum().item()) == 0


def test_erd_v3_loss_has_zero_target_leakage_in_negatives():
    loss_fn = ERDMSHNetV3Loss(far_radius=4, neg_topk_ratio=0.2)
    evidence = torch.zeros(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    evidence[:, :, 7:9, 7:9] = 20.0
    evidence[:, :, 0:4, 0:4] = 10.0

    neg, _ = loss_fn.select_online_negatives(evidence, target)
    target_dilate = binary_dilate(target, 4)

    assert int((neg.float() * target_dilate).sum().item()) == 0


def test_erd_v3_preserve_loss_positive_when_target_suppressed():
    loss_fn = ERDMSHNetV3Loss(target_protect_radius=1, neg_topk_ratio=0.05)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    z_e = torch.zeros(1, 1, 16, 16, requires_grad=True)
    z_f = torch.zeros(1, 1, 16, 16, requires_grad=True)
    z_e.data[:, :, 7:9, 7:9] = 5.0
    z_f.data[:, :, 7:9, 7:9] = -5.0

    out = loss_fn(_fake_outputs(z_e, z_f=z_f), target, epoch=5)

    assert float(out["loss_preserve"]) > 0.0
    assert out["total"].requires_grad


def test_erd_v3_preserve_loss_zero_when_target_preserved():
    loss_fn = ERDMSHNetV3Loss(target_protect_radius=1, neg_topk_ratio=0.05)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    z_e = torch.zeros(1, 1, 16, 16, requires_grad=True)
    z_e.data[:, :, 7:9, 7:9] = 5.0
    z_f = z_e.clone().detach().requires_grad_(True)

    out = loss_fn(_fake_outputs(z_e, z_f=z_f), target, epoch=5)

    assert torch.allclose(out["loss_preserve"], torch.tensor(0.0), atol=1e-8)


def test_erd_v3_loss_returns_aux_scalars():
    loss_fn = ERDMSHNetV3Loss(target_protect_radius=1, neg_topk_ratio=0.05)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    z_e = torch.randn(1, 1, 16, 16, requires_grad=True)
    z_f = z_e - torch.sigmoid(torch.randn(1, 1, 16, 16))
    z_f.requires_grad_(True)
    z_t = torch.randn(1, 1, 16, 16, requires_grad=True)
    z_c = torch.randn(1, 1, 16, 16, requires_grad=True)

    out = loss_fn(_fake_outputs(z_e, z_f=z_f, z_t=z_t, z_c=z_c), target, epoch=5)

    for key in (
        "loss_total",
        "loss_final",
        "loss_evidence",
        "loss_protect_pos",
        "loss_protect_neg",
        "loss_clutter_pos",
        "loss_clutter_neg",
        "loss_preserve",
        "online_neg_pixels",
        "mean_online_neg_pixels",
        "mean_protection_target",
        "mean_protection_far_bg",
        "mean_clutter_target",
        "mean_clutter_far_bg",
        "mean_suppression_target",
        "mean_suppression_far_bg",
    ):
        assert key in out
        assert torch.isfinite(out[key]).all()
