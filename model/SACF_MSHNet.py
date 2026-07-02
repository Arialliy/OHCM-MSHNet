import torch
import torch.nn as nn

from .MSHNet import MSHNet


class ScaleAgreementFusion(nn.Module):
    def __init__(self, hidden_channels=16, delta_max=1.0):
        super().__init__()
        self.delta_max = float(delta_max)
        in_channels = 9
        self.weight_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        last_gate = self.gate_head[-2]
        nn.init.zeros_(last_gate.weight)
        nn.init.constant_(last_gate.bias, -2.0)

    def forward(self, scale_logits, base_logits):
        z = torch.cat(scale_logits, dim=1)
        p = torch.sigmoid(z)

        p_mean = p.mean(dim=1, keepdim=True)
        p_var = p.var(dim=1, keepdim=True, unbiased=False)
        p_max = p.max(dim=1, keepdim=True).values
        p_min = p.min(dim=1, keepdim=True).values
        p_range = p_max - p_min

        feat = torch.cat([p, p_mean, p_var, p_max, p_min, p_range], dim=1)
        weights = torch.softmax(self.weight_head(feat), dim=1)
        gate = self.gate_head(feat)

        consensus = (weights * z).sum(dim=1, keepdim=True)
        delta = torch.clamp(consensus - base_logits, -self.delta_max, self.delta_max)
        final_logits = base_logits + gate * delta

        return {
            "final_logits": final_logits,
            "final_logit": final_logits,
            "base_logits": base_logits,
            "base_logit": base_logits,
            "consensus_logits": consensus,
            "scale_logits": scale_logits,
            "fusion_weights": weights,
            "fusion_gate": gate,
            "fusion_delta": delta,
            "scale_prob_mean": p_mean,
            "scale_prob_var": p_var,
            "scale_prob_range": p_range,
        }


class SACFMSHNet(nn.Module):
    def __init__(self, evidence_net=None, input_channels=1, hidden_channels=16, delta_max=1.0):
        super().__init__()
        self.evidence_net = evidence_net if evidence_net is not None else MSHNet(input_channels)
        self.fusion = ScaleAgreementFusion(hidden_channels=hidden_channels, delta_max=delta_max)
        self._evidence_frozen = False

    def freeze_evidence(self):
        for param in self.evidence_net.parameters():
            param.requires_grad = False
        self.evidence_net.eval()
        self._evidence_frozen = True

    def unfreeze_evidence(self):
        for param in self.evidence_net.parameters():
            param.requires_grad = True
        self._evidence_frozen = False

    def train(self, mode=True):
        super().train(mode)
        if self._evidence_frozen:
            self.evidence_net.eval()
        return self

    def forward(self, x, warm_flag=True, return_dict=True, output_head="final"):
        evidence = self.evidence_net(x, warm_flag=warm_flag, return_dict=True)
        scale_logits = evidence.get("scale_logits", [])
        if len(scale_logits) != 4:
            out = {
                "final_logits": evidence["base_logits"],
                "final_logit": evidence["base_logits"],
                "base_logits": evidence["base_logits"],
                "base_logit": evidence["base_logits"],
                "scale_logits": scale_logits,
                "fusion_weights": None,
                "fusion_gate": torch.zeros_like(evidence["base_logits"]),
                "fusion_delta": torch.zeros_like(evidence["base_logits"]),
            }
        else:
            out = self.fusion(scale_logits=scale_logits, base_logits=evidence["base_logits"])

        out["masks"] = evidence.get("masks", [])
        out["decoder_feature"] = evidence.get("decoder_feature", None)
        if return_dict:
            return out
        if output_head == "base":
            return out["base_logits"]
        if output_head == "final":
            return out["final_logits"]
        raise ValueError("Unsupported output_head: %s" % output_head)
