import torch.nn as nn

from .MSHNet import MSHNet


class ComponentGeometryHeads(nn.Module):
    def __init__(self, in_channels=16, hidden=32, num_scale_bins=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.center_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.scale_head = nn.Conv2d(hidden, num_scale_bins, kernel_size=1)
        self.core_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, feat):
        shared = self.shared(feat)
        return {
            "center_logits": self.center_head(shared),
            "geometry_scale_logits": self.scale_head(shared),
            "core_logits": self.core_head(shared),
            "boundary_logits": self.boundary_head(shared),
        }


class CGAMSHNet(nn.Module):
    """Component-Geometry Aligned MSHNet."""

    def __init__(self, input_channels=1, num_scale_bins=4):
        super().__init__()
        self.evidence_net = MSHNet(input_channels)
        self.geometry_heads = ComponentGeometryHeads(
            in_channels=16,
            hidden=32,
            num_scale_bins=num_scale_bins,
        )

    def forward(self, x, warm_flag=True, return_dict=True):
        evidence = self.evidence_net(x, warm_flag=warm_flag, return_dict=True)
        geometry = self.geometry_heads(evidence["decoder_feature"])
        out = {
            "final_logits": evidence["base_logits"],
            "final_logit": evidence["base_logits"],
            "base_logits": evidence["base_logits"],
            "base_logit": evidence["base_logits"],
            "scale_logits_up": evidence["scale_logits_up"],
            "scale_logits": evidence["scale_logits"],
            "masks": evidence["masks"],
            "decoder_feature": evidence["decoder_feature"],
            **geometry,
        }
        if return_dict:
            return out
        return evidence["masks"], evidence["base_logits"]


def configure_cga_trainable(model, mode="decoder_aux"):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if mode == "decoder_aux":
            if (
                "geometry_heads" in name
                or "evidence_net.decoder_0" in name
                or "evidence_net.output_0" in name
                or "evidence_net.final" in name
            ):
                param.requires_grad = True
        elif mode == "aux_only":
            if "geometry_heads" in name:
                param.requires_grad = True
        else:
            raise ValueError(mode)
    return [name for name, param in model.named_parameters() if param.requires_grad]
