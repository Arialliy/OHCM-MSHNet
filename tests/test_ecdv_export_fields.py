import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from net import Net


def test_ecdv_export_fields():
    net = Net(
        "ECDVMSHNet",
        mode="test",
        loss_cfg={
            "mshnet_in_channels": 1,
            "ecdv_hidden_channels": 4,
            "ecdv_beta_max": 0.1,
            "ecdv_eval_beta": 0.0,
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
        "risk_logit",
        "risk_prob",
        "suppression_map",
        "clutter_logit",
        "feature",
        "masks",
    ]:
        assert key in export
    assert export["final_logit"].shape == export["evidence_logit"].shape
    assert export["risk_prob"].min().item() >= 0.0
    assert export["risk_prob"].max().item() <= 1.0
