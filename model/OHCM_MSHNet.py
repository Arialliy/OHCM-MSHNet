import torch
import torch.nn as nn

from model.MSHNet import MSHNet


class ClutterHead(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        hidden_channels = max(8, in_channels // 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature):
        return self.head(feature)


class OHCMMSHNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        gamma_max=0.3,
        use_clutter_head=True,
        use_inhibition=True,
    ):
        super().__init__()
        self.backbone = MSHNet(input_channels)
        self.gamma_max = float(gamma_max)
        self.use_clutter_head = bool(use_clutter_head)
        self.use_inhibition = bool(use_inhibition)
        self.clutter_head = ClutterHead(in_channels=16)

    def forward(self, x, warm_flag=True, gamma=None, return_feature=False):
        masks, target_logit, feature = self.backbone(x, warm_flag, return_feature=True)
        if self.use_clutter_head:
            clutter_logit = self.clutter_head(feature)
        else:
            clutter_logit = torch.zeros_like(target_logit)

        if gamma is None:
            gamma = self.gamma_max
        if self.use_inhibition:
            final_logit = target_logit - float(gamma) * clutter_logit
        else:
            final_logit = target_logit

        out = {
            "masks": masks,
            "target_logit": target_logit,
            "clutter_logit": clutter_logit,
            "final_logit": final_logit,
            "feature": feature,
            "gamma": torch.tensor(float(gamma), device=target_logit.device, dtype=target_logit.dtype),
        }
        if return_feature:
            return out
        return out
