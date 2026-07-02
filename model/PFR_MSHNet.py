import torch
import torch.nn as nn
import torch.nn.functional as F


class PFRMSHNet(nn.Module):
    """Protected Far-background Residual MSHNet.

    The evidence branch is a normal MSHNet backbone. The residual branch is
    bounded and zero-initialized, so the model starts from the evidence logits.
    """

    def __init__(self, evidence_net: nn.Module, feature_channels: int = 16, beta: float = 0.5):
        super().__init__()
        self.evidence_net = evidence_net
        self.beta = float(beta)
        self.residual_head = nn.Sequential(
            nn.Conv2d(feature_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, x, warm_flag=True, return_dict: bool = False, output_head: str = "final"):
        if output_head not in ("final", "evidence", "residual"):
            raise ValueError("output_head must be one of: final, evidence, residual")

        masks, evidence_logits, feature = self.evidence_net(x, warm_flag, return_feature=True)
        if feature.shape[-2:] != evidence_logits.shape[-2:]:
            feature = F.interpolate(feature, size=evidence_logits.shape[-2:], mode="bilinear", align_corners=False)
        delta_raw = self.residual_head(feature)
        delta_logits = self.beta * torch.tanh(delta_raw)
        final_logits = evidence_logits + delta_logits
        output = {
            "logits": final_logits,
            "final_logit": final_logits,
            "evidence_logits": evidence_logits,
            "evidence_logit": evidence_logits,
            "residual_delta": delta_logits,
            "delta_logits": delta_logits,
            "raw_delta": delta_raw,
            "delta_raw": delta_raw,
            "beta": torch.as_tensor(self.beta, device=x.device, dtype=final_logits.dtype),
            "features": feature,
            "feature": feature,
            "masks": masks,
        }
        if return_dict:
            return output
        if output_head == "evidence":
            return evidence_logits
        if output_head == "residual":
            return delta_logits
        return final_logits
