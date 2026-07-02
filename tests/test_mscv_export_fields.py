import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from net import Net


def test_mscv_export_fields():
    net = Net(
        "MSCVMSHNet",
        mode="test",
        loss_cfg={
            "mshnet_in_channels": 1,
            "mscv_hidden_channels": 4,
            "mscv_beta_max": 0.1,
            "mscv_eval_beta": 0.0,
        },
    )
    net.eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        export = net.export_logits_features(x)

    for key in [
        "logit",
        "final_logit",
        "target_logit",
        "evidence_logit",
        "validity_logit",
        "validity_prob",
        "suppression_map",
        "p_mean",
        "p_std",
        "p_min",
        "p_max",
        "clutter_logit",
        "feature",
        "masks",
    ]:
        assert key in export
    assert export["final_logit"].shape == export["evidence_logit"].shape
    assert export["validity_prob"].min().item() >= 0.0
    assert export["validity_prob"].max().item() <= 1.0
