import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import PFRLoss, build_training_masks, select_topk_far_background
from model.PFR_MSHNet import PFRMSHNet


class DummyEvidence(torch.nn.Module):
    def forward(self, x, warm_flag=True, return_feature=False):
        evidence = x[:, :1]
        feature = torch.cat([x, x], dim=1)
        masks = [evidence]
        if return_feature:
            return masks, evidence, feature
        return masks, evidence


def test_pfr_zero_residual_equals_evidence_logits():
    model = PFRMSHNet(DummyEvidence(), feature_channels=2, beta=0.5)
    x = torch.randn(2, 1, 16, 16)

    out = model(x, return_dict=True)

    assert torch.max(torch.abs(out["logits"] - out["evidence_logits"])).item() == 0.0
    assert torch.max(torch.abs(out["delta_logits"])).item() == 0.0


def test_pfr_forward_return_dict_keys():
    model = PFRMSHNet(DummyEvidence(), feature_channels=2, beta=0.5)
    x = torch.randn(2, 1, 16, 16)

    out = model(x, return_dict=True)

    assert "evidence_logits" in out
    assert "final_logit" in out
    assert "residual_delta" in out
    assert "raw_delta" in out


def test_pfr_output_shapes_match():
    model = PFRMSHNet(DummyEvidence(), feature_channels=2, beta=0.5)
    x = torch.randn(2, 1, 16, 16)

    out = model(x, return_dict=True)

    assert out["evidence_logits"].shape == out["logits"].shape
    assert out["residual_delta"].shape == out["logits"].shape
    assert out["raw_delta"].shape == out["logits"].shape


def test_pfr_beta_zero_identity():
    model = PFRMSHNet(DummyEvidence(), feature_channels=2, beta=0.0)
    x = torch.randn(2, 1, 16, 16)

    out = model(x, return_dict=True)

    assert torch.allclose(out["logits"], out["evidence_logits"], atol=1e-7)


def test_pfr_output_head_evidence_final_residual():
    model = PFRMSHNet(DummyEvidence(), feature_channels=2, beta=0.5)
    x = torch.randn(2, 1, 16, 16)

    out = model(x, return_dict=True)
    evidence = model(x, output_head="evidence")
    final = model(x, output_head="final")
    residual = model(x, output_head="residual")

    assert torch.allclose(evidence, out["evidence_logits"])
    assert torch.allclose(final, out["logits"])
    assert torch.allclose(residual, out["residual_delta"])


def test_pfr_delta_is_bounded_by_beta():
    model = PFRMSHNet(DummyEvidence(), feature_channels=2, beta=0.25)
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(100.0)
        model.residual_head[-1].bias.fill_(100.0)
    x = torch.randn(2, 1, 16, 16)

    out = model(x, return_dict=True)

    assert torch.max(torch.abs(out["delta_logits"])).item() <= 0.25 + 1e-6


def test_train_pfr_guard_blocks_by_default():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_pfr_ready.py"),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "[BLOCKED] PFR-MSHNet is stopped" in result.stdout


def test_pfr_loss_does_not_select_target_as_far_negative():
    logits = torch.zeros(1, 1, 16, 16)
    logits[:, :, 0:4, 0:4] = 8.0
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0

    target_mask, _boundary_mask, far_bg_mask = build_training_masks(target, target_dilate=2, far_dilate=4)
    far_hard = select_topk_far_background(logits, far_bg_mask, topk_ratio=0.05)

    assert int((far_hard & target_mask).sum().item()) == 0
    assert int(far_hard.sum().item()) > 0


def test_pfr_target_protection_penalizes_negative_delta_on_target():
    loss_fn = PFRLoss(
        mshnet_warm_epoch=0,
        ohem_ratio=0.01,
        lambda_far_neg=0.0,
        lambda_target_protect=1.0,
        lambda_boundary_protect=0.0,
        lambda_residual_sparse=0.0,
    )
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    evidence = torch.zeros(1, 1, 16, 16)
    final = evidence.clone()
    delta = torch.zeros_like(evidence)
    delta[:, :, 7:9, 7:9] = -0.5
    outputs = {
        "logits": final,
        "evidence_logits": evidence,
        "delta_logits": delta,
        "masks": [final],
    }

    loss_out = loss_fn(outputs, target, epoch=1)

    assert float(loss_out["target_protect"]) > 0.0


def test_pfr_ready_guard_blocks_failed_candidate_audit(tmp_path):
    summary = {
        "gate_pass": False,
        "fail_reasons": ["target_leakage_pixels_nonzero"],
        "candidate_empty_image_ratio": 0.0,
        "target_leakage_pixels": 1,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_pfr_ready.py"),
            "--audit_summary",
            str(summary_path),
            "--stop_doc",
            str(tmp_path / "missing_stop_doc.md"),
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "PFR_NOT_READY" in result.stdout
