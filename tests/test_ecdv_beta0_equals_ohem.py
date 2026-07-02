import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.ECDV_MSHNet import ECDVMSHNet


class DummyEvidence(torch.nn.Module):
    def forward(self, x, warm_flag=True, return_feature=False):
        evidence = x[:, :1] * 2.0 - 0.25
        feature = torch.cat([x[:, :1], x[:, :1]], dim=1)
        masks = [evidence]
        if return_feature:
            return masks, evidence, feature
        return masks, evidence


def test_ecdv_beta0_equals_evidence_logit():
    model = ECDVMSHNet(input_channels=1, hidden_channels=4, beta_max=0.2)
    model.evidence_net = DummyEvidence()
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, beta=0.0, return_dict=True)

    assert torch.max(torch.abs(out["final_logit"] - out["evidence_logit"])).item() == 0.0
    assert torch.max(torch.abs(out["suppression_map"])).item() == 0.0
