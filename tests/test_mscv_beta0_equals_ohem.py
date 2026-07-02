import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.MSCV_MSHNet import MSCVMSHNet


class DummyEvidence(torch.nn.Module):
    def forward(self, x, warm_flag=True, return_feature=False):
        z = x[:, :1] * 2.0 - 0.5
        masks = [z - 0.1, z, z + 0.1, z + 0.2]
        feature = torch.cat([x[:, :1], x[:, :1]], dim=1)
        if return_feature:
            return masks, z, feature
        return masks, z


def test_mscv_beta0_equals_ohem():
    model = MSCVMSHNet(input_channels=1, hidden_channels=4, beta_max=0.2)
    model.evidence_net = DummyEvidence()
    model.eval()
    x = torch.randn(2, 1, 32, 32)

    with torch.no_grad():
        out = model(x, beta=0.0, return_dict=True)

    assert torch.max(torch.abs(out["final_logit"] - out["evidence_logit"])).item() == 0.0
    assert torch.max(torch.abs(out["suppression_map"])).item() == 0.0
