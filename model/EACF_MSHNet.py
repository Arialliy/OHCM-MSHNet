import torch
import torch.nn as nn
import torch.nn.functional as F

from .MSHNet import MSHNet


class CrossScaleFusion(nn.Module):
    """Evidence-anchored cross-scale dynamic fusion."""

    def __init__(self, feature_channels=16, hidden_channels=16, eta_max=0.5):
        super().__init__()
        self.eta_max = float(eta_max)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        gate_in = 4 + 4 + 5 + hidden_channels
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True),
        )

        self.eta = nn.Parameter(torch.tensor(0.0))
        self._init_gate()

    def _init_gate(self):
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @staticmethod
    def _upsample_scale_logits(masks, size):
        z0, z1, z2, z3 = masks
        return torch.cat(
            [
                F.interpolate(z0, size=size, mode="bilinear", align_corners=True),
                F.interpolate(z1, size=size, mode="bilinear", align_corners=True),
                F.interpolate(z2, size=size, mode="bilinear", align_corners=True),
                F.interpolate(z3, size=size, mode="bilinear", align_corners=True),
            ],
            dim=1,
        )

    def forward(self, masks, base_logit, decoder_feature):
        size = base_logit.shape[-2:]
        scale_logits = self._upsample_scale_logits(masks, size=size)
        scale_probs = torch.sigmoid(scale_logits)

        p_mean = scale_probs.mean(dim=1, keepdim=True)
        p_var = scale_probs.var(dim=1, keepdim=True, unbiased=False)
        p_max = scale_probs.max(dim=1, keepdim=True).values
        p_min = scale_probs.min(dim=1, keepdim=True).values
        p_range = p_max - p_min

        feat = self.feature_proj(decoder_feature)
        gate_input = torch.cat(
            [scale_logits, scale_probs, p_mean, p_var, p_range, p_max, p_min, feat],
            dim=1,
        )
        scale_weights = torch.softmax(self.gate(gate_input), dim=1)

        consensus_logit = (scale_weights * scale_logits).sum(dim=1, keepdim=True)
        eta = torch.clamp(self.eta, min=0.0, max=self.eta_max)
        final_logit = base_logit + eta * (consensus_logit - base_logit)

        return {
            "final_logit": final_logit,
            "base_logit": base_logit,
            "consensus_logit": consensus_logit,
            "scale_logits": scale_logits,
            "scale_probs": scale_probs,
            "scale_weights": scale_weights,
            "scale_var": p_var,
            "scale_range": p_range,
            "eta": eta.detach(),
        }


class EACFMSHNet(nn.Module):
    def __init__(self, input_channels=1, eta_max=0.5):
        super().__init__()
        self.backbone = MSHNet(input_channels)
        self.fusion = CrossScaleFusion(feature_channels=16, hidden_channels=16, eta_max=eta_max)
        self._backbone_frozen = False

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, x, warm_flag=True, return_dict=True):
        base = self.backbone(x, warm_flag=warm_flag, return_dict=True)
        if len(base["masks"]) != 4:
            out = {
                "final_logit": base["base_logit"],
                "base_logit": base["base_logit"],
                "masks": base["masks"],
                "decoder_feature": base["decoder_feature"],
                "eta": torch.tensor(0.0, device=x.device),
            }
            return out if return_dict else (base["masks"], base["base_logit"])

        fused = self.fusion(base["masks"], base["base_logit"], base["decoder_feature"])
        fused["masks"] = base["masks"]
        fused["decoder_feature"] = base["decoder_feature"]
        return fused if return_dict else (base["masks"], fused["final_logit"])
