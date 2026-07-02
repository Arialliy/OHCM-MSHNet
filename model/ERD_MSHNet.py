from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MSHNet import MSHNet


class ReliabilityHead(nn.Module):
    """Estimate whether target-like evidence is reliable."""

    def __init__(self, feature_channels: int = 16, hidden_channels: int = 32):
        super().__init__()
        in_channels = int(feature_channels) + 4 + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, feature: torch.Tensor, masks: list[torch.Tensor], evidence_logit: torch.Tensor) -> torch.Tensor:
        h, w = evidence_logit.shape[-2:]
        aligned = []
        for mask in masks:
            if mask.shape[-2:] != (h, w):
                mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=True)
            aligned.append(mask)

        while len(aligned) < 4:
            aligned.append(evidence_logit)
        ms_logits = torch.cat(aligned[:4], dim=1)
        scale_var = torch.var(ms_logits, dim=1, keepdim=True, unbiased=False)

        feature = self._align(feature, evidence_logit)
        x = torch.cat([feature, ms_logits, scale_var], dim=1)
        return self.net(x)


class ERDMSHNet(nn.Module):
    """Evidence-Reliability Decoupled MSHNet with suppress-only gated fusion."""

    def __init__(
        self,
        input_channels: int = 1,
        feature_channels: int = 16,
        hidden_channels: int = 32,
        rho: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not (0.0 < float(rho) <= 1.0):
            raise ValueError("rho must be in (0, 1].")
        self.evidence = MSHNet(input_channels)
        self.reliability = ReliabilityHead(
            feature_channels=feature_channels,
            hidden_channels=hidden_channels,
        )
        self.rho = float(rho)
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        gamma: float = 1.0,
        return_feature: bool = False,
    ) -> dict:
        masks, evidence_logit, feature = self.evidence(
            x,
            warm_flag=warm_flag,
            return_feature=True,
        )

        reliability_logit = self.reliability(feature, masks, evidence_logit)
        reliability = torch.sigmoid(reliability_logit)

        gate = self.rho + (1.0 - self.rho) * reliability
        gate = gate.clamp(min=self.eps, max=1.0)
        final_logit = evidence_logit + float(gamma) * torch.log(gate)

        out = {
            "masks": masks,
            "evidence_logit": evidence_logit,
            "reliability_logit": reliability_logit,
            "final_logit": final_logit,
            "gate": gate,
        }
        if return_feature:
            out["feature"] = feature
        return out


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = int(k) // 2
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=int(k), padding=p, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProtectionHead(nn.Module):
    """Predict target support that should be protected from suppression."""

    def __init__(self, in_ch: int, hidden_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, hidden_ch, 3),
            ConvBNAct(hidden_ch, hidden_ch, 3),
            nn.Conv2d(int(hidden_ch), 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ClutterHead(nn.Module):
    """Predict target-like background clutter to suppress."""

    def __init__(self, in_ch: int, hidden_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, hidden_ch, 3),
            ConvBNAct(hidden_ch, hidden_ch, 3),
            nn.Conv2d(int(hidden_ch), 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ERDMSHNetV3(nn.Module):
    """ERD-MSHNet v3 with target-preserving clutter suppression."""

    def __init__(
        self,
        evidence_net: nn.Module | None = None,
        input_channels: int = 1,
        aux_in_channels: int = 16,
        hidden_channels: int = 32,
        s_max: float = 4.0,
    ):
        super().__init__()
        self.evidence_net = evidence_net if evidence_net is not None else MSHNet(int(input_channels))
        self.protection_head = ProtectionHead(int(aux_in_channels), int(hidden_channels))
        self.clutter_head = ClutterHead(int(aux_in_channels), int(hidden_channels))
        self.s_max = float(s_max)

    @staticmethod
    def _parse_evidence_outputs(evidence_outputs):
        if isinstance(evidence_outputs, dict):
            logits = None
            for key in ("logits", "final_logits", "final_logit", "evidence_logits", "evidence_logit", "out", "output"):
                if key in evidence_outputs:
                    logits = evidence_outputs[key]
                    break
            if logits is None:
                raise RuntimeError("Cannot parse evidence logits from dict output.")
            feature = evidence_outputs.get("feature", evidence_outputs.get("decoder_feature", logits))
            masks = evidence_outputs.get("masks", [])
            return masks, logits, feature

        if isinstance(evidence_outputs, (list, tuple)):
            if len(evidence_outputs) >= 3:
                return evidence_outputs[0], evidence_outputs[1], evidence_outputs[2]
            if len(evidence_outputs) >= 2:
                return evidence_outputs[0], evidence_outputs[1], evidence_outputs[1]
            if len(evidence_outputs) == 1:
                return [], evidence_outputs[0], evidence_outputs[0]

        if torch.is_tensor(evidence_outputs):
            return [], evidence_outputs, evidence_outputs

        raise RuntimeError("Cannot parse ERD-v3 evidence outputs.")

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        return_aux: bool = False,
        return_feature: bool = False,
    ):
        evidence_outputs = self.evidence_net(
            x,
            warm_flag=warm_flag,
            return_feature=True,
        )
        masks, evidence_logits, aux_feat = self._parse_evidence_outputs(evidence_outputs)

        if aux_feat.shape[-2:] != evidence_logits.shape[-2:]:
            aux_feat = F.interpolate(
                aux_feat,
                size=evidence_logits.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        protection_logits = self.protection_head(aux_feat)
        clutter_logits = self.clutter_head(aux_feat)
        protection = torch.sigmoid(protection_logits)
        clutter = torch.sigmoid(clutter_logits)
        suppression = self.s_max * clutter * (1.0 - protection)
        final_logits = evidence_logits - suppression

        if not return_aux and not return_feature:
            return final_logits

        out = {
            "logits": final_logits,
            "final_logits": final_logits,
            "final_logit": final_logits,
            "evidence_logits": evidence_logits,
            "evidence_logit": evidence_logits,
            "protection_logits": protection_logits,
            "protection_logit": protection_logits,
            "clutter_logits": clutter_logits,
            "clutter_logit": clutter_logits,
            "protection": protection,
            "clutter": clutter,
            "suppression": suppression,
            "gate": torch.exp(-suppression).clamp(min=0.0, max=1.0),
            "masks": masks,
        }
        if return_feature:
            out["feature"] = aux_feat
        return out
