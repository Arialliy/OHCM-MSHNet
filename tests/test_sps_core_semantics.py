import json
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from loss import OHEMLogitLoss, SelfPerturbationStabilityLoss
from metrics import BinaryMetricsGPU
from probability import foreground_probability
from tools.official.evaluate_checkpoint_direct import resolve_mshnet_head
from tools.official.evaluate_prediction_exports import require_direct_export_parity
from tools.official.sps_perturbation_census import region_candidate_and_selected, sps_candidate_and_selected
from utils import get_img_norm_cfg


def test_foreground_probability_binary_and_softmax():
    binary_logits = torch.tensor([[[[0.0, 2.0]]]])
    binary_prob = foreground_probability(binary_logits)
    assert torch.allclose(binary_prob, torch.sigmoid(binary_logits))

    two_channel_logits = torch.tensor([[[[2.0]], [[4.0]]]])
    two_channel_prob = foreground_probability(two_channel_logits)
    expected = torch.softmax(two_channel_logits, dim=1)[:, 1:2]
    assert torch.allclose(two_channel_prob, expected)


def test_mshnet_auto_head_resolves_to_final():
    assert resolve_mshnet_head("MSHNet", "auto") == "final"
    assert resolve_mshnet_head("MSHNetOHEM", "auto") == "final"


def test_direct_export_parity_gate_requires_all_checks():
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp) / "exports"
        parity_dir = exports_dir / "direct_export_parity"
        parity_dir.mkdir(parents=True)
        payload = {
            "pass": True,
            "exports_dir": str(exports_dir.resolve()),
            "image_list": None,
            "checks": {
                "max_prob_diff": True,
                "mask_diff_pixels": True,
                "mIoU_diff": True,
                "Pd_diff": True,
                "FA_ppm_diff": True,
                "direct_target_gt_background": True,
                "export_target_gt_background": True,
            },
        }
        (parity_dir / "direct_export_parity_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        assert require_direct_export_parity(exports_dir, None)["pass"] is True

        payload["checks"]["FA_ppm_diff"] = False
        (parity_dir / "direct_export_parity_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        try:
            require_direct_export_parity(exports_dir, None)
        except SystemExit:
            pass
        else:
            raise AssertionError("parity gate should reject failed checks")


def test_sps_alpha_zero_equals_ohem():
    torch.manual_seed(7)
    final = torch.randn(2, 1, 16, 16, requires_grad=True)
    perturb = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:6, 4:6] = 1.0

    sps = SelfPerturbationStabilityLoss(
        rerank_strict_fallback=True,
        candidate_tau=2.0,
        candidate_topk_metric="sps_score",
        candidate_fallback_topk_ratio=1e-5,
    )
    ohem = OHEMLogitLoss(topk_ratio=0.01)

    sps_loss, stats = sps.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.0)
    ohem_loss = ohem(final, target)

    assert torch.allclose(sps_loss, ohem_loss)
    expected_budget = sum(
        int(((target[b] <= 0).sum().item()) * 0.01)
        for b in range(target.shape[0])
    )
    assert float(stats["sps_hard_pixels"]) == float(expected_budget)
    assert float(stats["sps_ohem_jaccard"]) == 1.0


def test_sps_alpha_zero_matches_ohem_grad_and_one_step():
    torch.manual_seed(19)
    x = torch.randn(2, 1, 8, 8)
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:4, 2:4] = 1.0
    model_sps = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
    model_ohem = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
    model_ohem.load_state_dict(model_sps.state_dict())

    sps = SelfPerturbationStabilityLoss(
        rerank_strict_fallback=True,
        candidate_tau=2.0,
        candidate_topk_metric="sps_score",
        candidate_fallback_topk_ratio=1e-5,
    )
    ohem = OHEMLogitLoss(topk_ratio=0.01)

    opt_sps = torch.optim.SGD(model_sps.parameters(), lr=0.05)
    opt_ohem = torch.optim.SGD(model_ohem.parameters(), lr=0.05)
    final_sps = model_sps(x)
    perturb_sps = final_sps.detach().clone().requires_grad_(True)
    final_ohem = model_ohem(x)
    sps_loss, _ = sps.rerank_ohem_loss(final_sps, perturb_sps, target, topk_ratio=0.01, alpha=0.0)
    ohem_loss = ohem(final_ohem, target)

    assert torch.allclose(sps_loss, ohem_loss, atol=1e-7, rtol=0.0)
    sps_loss.backward()
    ohem_loss.backward()
    for p_sps, p_ohem in zip(model_sps.parameters(), model_ohem.parameters()):
        assert torch.max(torch.abs(p_sps.grad - p_ohem.grad)).item() < 1e-7
    opt_sps.step()
    opt_ohem.step()
    for p_sps, p_ohem in zip(model_sps.parameters(), model_ohem.parameters()):
        assert torch.max(torch.abs(p_sps - p_ohem)).item() < 1e-7


def test_sps_align_back_inverts_label_preserving_transforms():
    sps = SelfPerturbationStabilityLoss()
    tensor = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).view(2, 1, 3, 4)
    assert torch.equal(sps.align_back(torch.flip(tensor, dims=[-1]), "hflip"), tensor)
    assert torch.equal(sps.align_back(torch.flip(tensor, dims=[-2]), "vflip"), tensor)
    assert torch.equal(sps.align_back(torch.flip(tensor, dims=[-2, -1]), "hvflip"), tensor)
    assert torch.equal(sps.align_back(tensor.transpose(-1, -2), "transpose"), tensor)


def test_sps_training_path_uses_batch_concat_forward():
    train_source = (PROJECT_ROOT / "train.py").read_text(encoding="utf-8")
    assert "torch.cat([img, sps_img], dim=0)" in train_source
    assert "final_all[:batch_size]" in train_source
    assert "final_all[batch_size:]" in train_source


def test_strict_fallback_keeps_budget_sized_shortlist():
    torch.manual_seed(11)
    final = torch.randn(1, 1, 32, 32, requires_grad=True)
    perturb = final.detach().clone().requires_grad_(True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 10:12, 10:12] = 1.0

    sps = SelfPerturbationStabilityLoss(
        rerank_strict_fallback=True,
        candidate_tau=2.0,
        candidate_topk_metric="sps_score",
        candidate_fallback_topk_ratio=1e-5,
    )
    _, stats = sps.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.2)

    assert float(stats["sps_candidate_pixels"]) == float(stats["sps_hard_pixels"])
    assert float(stats["sps_candidate_pixels"]) == 10.0


def test_rerank_candidate_topk_ratio_controls_candidate_pool():
    torch.manual_seed(13)
    final = torch.randn(1, 1, 32, 32, requires_grad=True)
    perturb = (final.detach() + 0.01 * torch.randn(1, 1, 32, 32)).requires_grad_(True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 10:12, 10:12] = 1.0

    common = dict(
        rerank_strict_fallback=True,
        candidate_tau=2.0,
        candidate_topk_metric="sps_score",
        candidate_fallback_topk_ratio=0.0,
    )
    small_pool = SelfPerturbationStabilityLoss(candidate_topk_ratio=0.02, **common)
    large_pool = SelfPerturbationStabilityLoss(candidate_topk_ratio=0.05, **common)

    _, stats_small = small_pool.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.2)
    _, stats_large = large_pool.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.2)

    assert float(stats_small["sps_fallback_images"]) == 0.0
    assert float(stats_large["sps_fallback_images"]) == 0.0
    assert float(stats_small["sps_candidate_pixels"]) > float(stats_small["sps_hard_pixels"])
    assert float(stats_large["sps_candidate_pixels"]) > float(stats_small["sps_candidate_pixels"])


def test_target_margin_metric_suppresses_target_like_instability():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1.0
    target[:, :, 0, 1] = 1.0
    instability = torch.zeros(1, 1, 4, 4)
    instability[:, :, 0, 0] = 0.10
    instability[:, :, 0, 1] = 0.20
    instability[:, :, 2, 2] = 0.14
    instability[:, :, 3, 3] = 0.30
    confidence = torch.ones_like(instability)
    hardness = torch.ones_like(instability)

    sps = SelfPerturbationStabilityLoss(
        candidate_topk_metric="target_margin_instability",
        target_margin_quantile=0.5,
    )
    metric = sps._candidate_metric(confidence, instability, hardness, target=target)

    assert float(metric[:, :, 2, 2]) == 0.0
    assert float(metric[:, :, 3, 3]) > 0.0


def test_candidate_min_metric_filters_zero_margin_topk():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1.0
    target[:, :, 0, 1] = 1.0
    instability = torch.zeros(1, 1, 4, 4)
    instability[:, :, 0, 0] = 0.10
    instability[:, :, 0, 1] = 0.20
    instability[:, :, 2, 2] = 0.14
    instability[:, :, 3, 3] = 0.30
    confidence = torch.ones_like(instability)
    hardness = torch.ones_like(instability)
    far_bg = target <= 0

    sps = SelfPerturbationStabilityLoss(
        candidate_topk_ratio=1.0,
        candidate_topk_metric="target_margin_instability",
        candidate_min_metric=0.0,
        target_margin_quantile=0.5,
    )
    candidate = sps._candidate_mask(
        confidence,
        far_bg,
        instability=instability,
        hardness=hardness,
        target=target,
    )

    assert int(candidate.sum().item()) == 1
    assert bool(candidate[:, :, 3, 3].item()) is True
    assert bool(candidate[:, :, 2, 2].item()) is False



def test_target_contrast_metric_is_continuous_relative_to_target():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1.0
    target[:, :, 0, 1] = 1.0
    instability = torch.zeros(1, 1, 4, 4)
    instability[:, :, 0, 0] = 0.10
    instability[:, :, 0, 1] = 0.20
    instability[:, :, 2, 2] = 0.14
    instability[:, :, 3, 3] = 0.30
    confidence = torch.ones_like(instability)
    hardness = torch.ones_like(instability)

    sps = SelfPerturbationStabilityLoss(
        candidate_topk_metric="target_contrast_instability",
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
    )
    metric = sps._candidate_metric(confidence, instability, hardness, target=target)

    assert 0.0 < float(metric[:, :, 2, 2]) < 0.5
    assert float(metric[:, :, 3, 3]) > 0.5


def test_sps_census_target_contrast_uses_contrast_signal():
    prob_w = np.full((4, 4), 0.01, dtype=np.float32)
    prob_p = np.full((4, 4), 0.01, dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    gt[0, 0] = True
    gt[0, 1] = True
    far_mask = ~gt
    instability = np.zeros((4, 4), dtype=np.float32)
    instability[0, 0] = 0.10
    instability[0, 1] = 0.20
    instability[2, 2] = 0.14
    instability[3, 3] = 0.30
    prob_w[2, 2] = 0.99
    prob_p[2, 2] = 0.99

    _, selected = sps_candidate_and_selected(
        prob_w=prob_w,
        prob_p=prob_p,
        instability=instability,
        gt=gt,
        far_mask=far_mask,
        candidate_tau=0.7,
        candidate_topk_ratio=1.0,
        candidate_topk_metric="target_contrast_sps_score",
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        budget_q=0.08,
        kmax=1,
        eta=1.0,
    )

    assert bool(selected[2, 2]) is True
    assert bool(selected[3, 3]) is False


def test_candidate_pool_metric_decoupled_from_rerank_metric():
    prob_w = np.full((4, 4), 0.2, dtype=np.float32)
    prob_p = np.full((4, 4), 0.2, dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    gt[0, 0] = True
    gt[0, 1] = True
    far_mask = ~gt
    instability = np.zeros((4, 4), dtype=np.float32)
    instability[0, 0] = 0.10
    instability[0, 1] = 0.20
    instability[2, 2] = 0.14
    instability[3, 3] = 0.30
    prob_w[2, 2] = 0.99
    prob_p[2, 2] = 0.99

    candidate, selected, stats = sps_candidate_and_selected(
        prob_w=prob_w,
        prob_p=prob_p,
        instability=instability,
        gt=gt,
        far_mask=far_mask,
        candidate_tau=0.7,
        candidate_topk_ratio=0.08,
        candidate_topk_metric="target_contrast_sps_score",
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        budget_q=1.0,
        kmax=1,
        eta=1.0,
        candidate_pool_metric="target_contrast_instability",
        rerank_signal_metric="target_contrast",
        rerank_base_metric="weak_neg_loss",
        return_stats=True,
    )

    assert bool(candidate[3, 3]) is True
    assert bool(candidate[2, 2]) is False
    assert bool(selected[3, 3]) is True
    assert stats["candidate_pool_metric"] == "target_contrast_instability"
    assert stats["rerank_signal_metric"] == "target_contrast"


def test_target_contrast_pool_does_not_use_hardness():
    prob_w = np.full((4, 4), 0.05, dtype=np.float32)
    prob_p = np.full((4, 4), 0.05, dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    gt[0, 0] = True
    gt[0, 1] = True
    far_mask = ~gt
    instability = np.zeros((4, 4), dtype=np.float32)
    instability[0, 0] = 0.10
    instability[0, 1] = 0.20
    instability[2, 2] = 0.14
    instability[3, 3] = 0.30
    prob_w[2, 2] = 0.99
    prob_p[2, 2] = 0.99

    candidate, _, stats = sps_candidate_and_selected(
        prob_w=prob_w,
        prob_p=prob_p,
        instability=instability,
        gt=gt,
        far_mask=far_mask,
        candidate_tau=0.7,
        candidate_topk_ratio=0.08,
        candidate_topk_metric="sps_score",
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        budget_q=1.0,
        kmax=1,
        eta=1.0,
        candidate_pool_metric="target_contrast_instability",
        rerank_signal_metric="target_contrast",
        rerank_base_metric="weak_neg_loss",
        return_stats=True,
    )

    assert bool(candidate[3, 3]) is True
    assert bool(candidate[2, 2]) is False
    assert stats["candidate_to_budget_ratio"] == 1.0


def test_no_fallback_to_ohem_in_gate0_diagnostic_mode():
    prob_w = np.full((4, 4), 0.01, dtype=np.float32)
    prob_p = np.full((4, 4), 0.01, dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    far_mask = ~gt
    instability = np.zeros((4, 4), dtype=np.float32)

    candidate, selected, stats = sps_candidate_and_selected(
        prob_w=prob_w,
        prob_p=prob_p,
        instability=instability,
        gt=gt,
        far_mask=far_mask,
        candidate_tau=0.7,
        candidate_topk_ratio=0.0,
        candidate_topk_metric="confidence",
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        budget_q=1.0,
        kmax=1,
        eta=1.0,
        return_stats=True,
    )

    assert int(candidate.sum()) == 0
    assert int(selected.sum()) == 0
    assert stats["fallback_used"] is False
    assert stats["budget_pixels"] == 0


def test_candidate_to_budget_ratio_uses_fixed_ohem_budget():
    prob_w = np.full((4, 4), 0.2, dtype=np.float32)
    prob_p = np.full((4, 4), 0.2, dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    far_mask = ~gt
    instability = np.arange(16, dtype=np.float32).reshape(4, 4)

    candidate, selected, stats = sps_candidate_and_selected(
        prob_w=prob_w,
        prob_p=prob_p,
        instability=instability,
        gt=gt,
        far_mask=far_mask,
        candidate_tau=0.0,
        candidate_topk_ratio=0.5,
        candidate_topk_metric="instability",
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        budget_q=0.1,
        kmax=256,
        eta=1.0,
        fixed_budget_pixels=4,
        return_stats=True,
    )

    assert int(candidate.sum()) == 8
    assert int(selected.sum()) == 4
    assert stats["budget_pixels"] == 4
    assert stats["candidate_to_budget_ratio"] == 2.0


def test_no_silent_ohem_fallback_when_candidate_under_budget():
    prob_w = np.full((4, 4), 0.2, dtype=np.float32)
    prob_p = np.full((4, 4), 0.2, dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    far_mask = ~gt
    instability = np.arange(16, dtype=np.float32).reshape(4, 4)

    candidate, selected, stats = sps_candidate_and_selected(
        prob_w=prob_w,
        prob_p=prob_p,
        instability=instability,
        gt=gt,
        far_mask=far_mask,
        candidate_tau=0.0,
        candidate_topk_ratio=0.0625,
        candidate_topk_metric="instability",
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        budget_q=0.1,
        kmax=256,
        eta=1.0,
        fixed_budget_pixels=4,
        return_stats=True,
    )

    assert int(candidate.sum()) == 1
    assert int(selected.sum()) == 1
    assert stats["candidate_under_budget"] is True
    assert stats["fallback_used"] is False
    assert stats["candidate_to_budget_ratio"] == 0.25


def test_region_candidate_excludes_dilated_target():
    prob_w = np.full((8, 8), 0.05, dtype=np.float32)
    prob_p = np.full((8, 8), 0.05, dtype=np.float32)
    instability = np.zeros((8, 8), dtype=np.float32)
    gt = np.zeros((8, 8), dtype=bool)
    gt[3:5, 3:5] = True
    far_mask = np.ones((8, 8), dtype=bool)
    far_mask[2:6, 2:6] = False
    prob_w[3, 3] = 0.9
    prob_p[3, 3] = 0.9
    instability[3, 3] = 1.0
    prob_w[1:3, 1:3] = 0.9
    prob_p[1:3, 1:3] = 0.9
    instability[1:3, 1:3] = 1.0

    candidate, selected, stats, _ = region_candidate_and_selected(
        prob_w,
        prob_p,
        instability,
        gt,
        far_mask,
        mode="region_component",
        budget_pixels=2,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        region_min_area=1,
        region_max_area=16,
        region_conf_min=0.1,
        region_signal_min=0.1,
        region_pool_topq=0.0,
        peak_topk_ratio=0.005,
        peak_nms_radius=3,
        peak_window_radius=4,
        peak_min_conf=0.1,
        peak_min_signal=0.1,
    )

    assert bool(np.logical_and(candidate, ~far_mask).any()) is False
    assert bool(np.logical_and(selected, gt).any()) is False
    assert stats["selected_pixels"] == 2


def test_region_candidate_budget_matches_ohem_when_enough_candidates():
    prob_w = np.full((8, 8), 0.05, dtype=np.float32)
    prob_p = np.full((8, 8), 0.05, dtype=np.float32)
    instability = np.zeros((8, 8), dtype=np.float32)
    gt = np.zeros((8, 8), dtype=bool)
    far_mask = np.ones((8, 8), dtype=bool)
    prob_w[1:4, 1:4] = 0.8
    prob_p[1:4, 1:4] = 0.8
    instability[1:4, 1:4] = 1.0

    _, selected, stats, _ = region_candidate_and_selected(
        prob_w,
        prob_p,
        instability,
        gt,
        far_mask,
        mode="region_component",
        budget_pixels=4,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        region_min_area=1,
        region_max_area=16,
        region_conf_min=0.1,
        region_signal_min=0.1,
        region_pool_topq=0.0,
        peak_topk_ratio=0.005,
        peak_nms_radius=3,
        peak_window_radius=4,
        peak_min_conf=0.1,
        peak_min_signal=0.1,
    )

    assert int(selected.sum()) == 4
    assert stats["candidate_under_budget"] is False
    assert stats["candidate_to_budget_ratio"] > 1.0


def test_region_candidate_reports_under_budget_when_not_enough():
    prob_w = np.full((8, 8), 0.05, dtype=np.float32)
    prob_p = np.full((8, 8), 0.05, dtype=np.float32)
    instability = np.zeros((8, 8), dtype=np.float32)
    gt = np.zeros((8, 8), dtype=bool)
    far_mask = np.ones((8, 8), dtype=bool)
    prob_w[1, 1] = 0.8
    prob_p[1, 1] = 0.8
    instability[1, 1] = 1.0

    candidate, selected, stats, _ = region_candidate_and_selected(
        prob_w,
        prob_p,
        instability,
        gt,
        far_mask,
        mode="region_component",
        budget_pixels=4,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        region_min_area=1,
        region_max_area=16,
        region_conf_min=0.1,
        region_signal_min=0.1,
        region_pool_topq=0.0,
        peak_topk_ratio=0.005,
        peak_nms_radius=3,
        peak_window_radius=4,
        peak_min_conf=0.1,
        peak_min_signal=0.1,
    )

    assert int(candidate.sum()) == 1
    assert int(selected.sum()) == 1
    assert stats["candidate_under_budget"] is True
    assert stats["candidate_to_budget_ratio"] == 0.25


def test_peak_region_nms_suppresses_close_peaks():
    prob_w = np.full((12, 12), 0.05, dtype=np.float32)
    prob_p = np.full((12, 12), 0.05, dtype=np.float32)
    instability = np.zeros((12, 12), dtype=np.float32)
    gt = np.zeros((12, 12), dtype=bool)
    far_mask = np.ones((12, 12), dtype=bool)
    for y, x in [(2, 2), (3, 3), (9, 9)]:
        prob_w[y, x] = 0.9
        prob_p[y, x] = 0.9
        instability[y, x] = 1.0

    _, _, stats, rows = region_candidate_and_selected(
        prob_w,
        prob_p,
        instability,
        gt,
        far_mask,
        mode="peak_region",
        budget_pixels=4,
        target_margin_quantile=0.5,
        target_margin_temp=0.02,
        target_margin_min=0.0,
        region_min_area=1,
        region_max_area=32,
        region_conf_min=0.0,
        region_signal_min=0.0,
        region_pool_topq=0.0,
        peak_topk_ratio=1.0,
        peak_nms_radius=3,
        peak_window_radius=1,
        peak_min_conf=0.1,
        peak_min_signal=0.1,
    )

    assert stats["candidate_region_count"] == 2
    assert len(rows) == 2


def test_target_margin_sps_score_is_zero_without_positive_margin():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1.0
    target[:, :, 0, 1] = 1.0
    instability = torch.zeros(1, 1, 4, 4)
    instability[:, :, 0, 0] = 0.10
    instability[:, :, 0, 1] = 0.20
    instability[:, :, 2, 2] = 0.14
    instability[:, :, 3, 3] = 0.30
    confidence = torch.ones_like(instability)
    hardness = torch.ones_like(instability)

    sps = SelfPerturbationStabilityLoss(
        candidate_topk_metric="target_margin_sps_score",
        target_margin_quantile=0.5,
    )
    metric = sps._candidate_metric(confidence, instability, hardness, target=target)

    assert float(metric[:, :, 2, 2]) == 0.0
    assert float(metric[:, :, 3, 3]) > 0.0


def test_target_margin_mode_keeps_fixed_negative_budget():
    torch.manual_seed(17)
    final = torch.randn(1, 1, 32, 32, requires_grad=True)
    perturb = (final.detach() + 0.05 * torch.randn(1, 1, 32, 32)).requires_grad_(True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 10:12, 10:12] = 1.0

    sps = SelfPerturbationStabilityLoss(
        mode="target_margin",
        candidate_topk_ratio=0.02,
        candidate_topk_metric="target_margin_sps_score",
        target_margin_quantile=0.85,
        rerank_strict_fallback=True,
    )
    _, stats = sps.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.2)

    expected_budget = int((target.numel() - int(target.sum().item())) * 0.01)
    assert float(stats["sps_hard_pixels"]) == float(expected_budget)


def test_disable_far_mask_expands_candidate_pool():
    final = torch.zeros(1, 1, 16, 16, requires_grad=True)
    perturb = torch.zeros(1, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0

    common = dict(
        candidate_topk_ratio=1.0,
        candidate_topk_metric="confidence",
        candidate_fallback_topk_ratio=0.0,
        rerank_strict_fallback=True,
        adaptive_radius=False,
        dilate_radius=3,
    )
    with_far_mask = SelfPerturbationStabilityLoss(**common)
    without_far_mask = SelfPerturbationStabilityLoss(disable_far_mask=True, **common)

    _, stats_far = with_far_mask.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.2)
    _, stats_all = without_far_mask.rerank_ohem_loss(final, perturb, target, topk_ratio=0.01, alpha=0.2)

    assert float(stats_all["sps_candidate_pixels"]) > float(stats_far["sps_candidate_pixels"])




def test_candidate_min_confidence_filters_topk_seed():
    target = torch.zeros(1, 1, 4, 4)
    far_bg = target <= 0
    confidence = torch.zeros_like(target)
    confidence[:, :, 0, 0] = 0.2
    confidence[:, :, 3, 3] = 0.8
    instability = torch.zeros_like(target)
    instability[:, :, 0, 0] = 10.0
    instability[:, :, 3, 3] = 1.0
    hardness = torch.ones_like(target)

    sps = SelfPerturbationStabilityLoss(
        candidate_topk_ratio=1.0,
        candidate_topk_metric="instability",
        candidate_min_confidence=0.5,
    )
    candidate = sps._candidate_mask(
        confidence,
        far_bg,
        instability=instability,
        hardness=hardness,
        target=target,
    )

    assert bool(candidate[:, :, 0, 0].item()) is False
    assert bool(candidate[:, :, 3, 3].item()) is True


def test_candidate_expansion_stays_inside_far_mask():
    target = torch.zeros(1, 1, 9, 9)
    target[:, :, 4, 4] = 1.0
    confidence = torch.zeros_like(target)
    confidence[:, :, 1, 1] = 1.0
    instability = torch.ones_like(target)
    hardness = torch.ones_like(target)
    far_bg = target <= 0
    far_bg[:, :, 1, 2] = False

    no_expand = SelfPerturbationStabilityLoss(
        candidate_topk_ratio=0.0,
        candidate_tau=0.5,
        candidate_expand_radius=0,
        adaptive_radius=False,
        dilate_radius=0,
    )
    expand = SelfPerturbationStabilityLoss(
        candidate_topk_ratio=0.0,
        candidate_tau=0.5,
        candidate_expand_radius=1,
        adaptive_radius=False,
        dilate_radius=0,
    )
    seed = no_expand._candidate_mask(confidence, far_bg, instability=instability, hardness=hardness, target=target)
    grown = expand._candidate_mask(confidence, far_bg, instability=instability, hardness=hardness, target=target)

    assert int(seed.sum().item()) == 1
    assert int(grown.sum().item()) > int(seed.sum().item())
    assert bool(grown[:, :, 1, 2].item()) is False
    assert bool(torch.logical_and(grown, ~far_bg).any().item()) is False


def test_binary_metrics_reports_pixel_recall_not_pd():
    metric = BinaryMetricsGPU()
    empty = metric.get()
    assert "PixelRecall" in empty
    assert "Pd" not in empty

    pred = torch.tensor([[[[0.6, 0.1], [0.2, 0.7]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    metric.update(pred, target)
    result = metric.get()

    assert result["PixelRecall"] == 1.0
    assert "Pd" not in result


def test_unknown_dataset_norm_uses_train_split_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "ToyExternal"
        (dataset / "img_idx").mkdir(parents=True)
        (dataset / "images").mkdir()
        (dataset / "img_idx" / "train_ToyExternal.txt").write_text("train_img\n", encoding="utf-8")
        (dataset / "img_idx" / "test_ToyExternal.txt").write_text("missing_test_img\n", encoding="utf-8")
        pixels = np.array([[10, 14], [18, 22]], dtype=np.uint8)
        Image.fromarray(pixels).save(dataset / "images" / "train_img.png")

        cfg = get_img_norm_cfg("ToyExternal", str(root))

    assert abs(cfg["mean"] - float(pixels.mean())) < 1e-6
    assert abs(cfg["std"] - float(pixels.std())) < 1e-6


def test_checkpoint_state_contains_reproducibility_metadata():
    old_argv = sys.argv[:]
    sys.argv = ["train.py"]
    try:
        train_module = importlib.import_module("train")
    finally:
        sys.argv = old_argv

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "ToyTrain"
        (dataset / "img_idx").mkdir(parents=True)
        split = dataset / "img_idx" / "train_ToyTrain.txt"
        split.write_text("a\nb\n", encoding="utf-8")
        train_module.opt.dataset_dir = str(root)
        train_module.opt.dataset_name = "ToyTrain"

        net = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        generator = torch.Generator().manual_seed(123)

        state = train_module.make_checkpoint_state(3, net, optimizer, scheduler, [1.0], generator)

    for key in (
        "state_dict",
        "optimizer",
        "scheduler",
        "config",
        "seed",
        "rng_state",
        "dataloader_generator_state",
        "git_commit",
        "best_metric",
        "dataset_split_hash",
        "checkpoint_schema_version",
    ):
        assert key in state
    assert state["dataset_split_hash"]["status"] == "ok"
    assert len(state["dataset_split_hash"]["sha256"]) == 64
    assert state["checkpoint_schema_version"] >= 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
