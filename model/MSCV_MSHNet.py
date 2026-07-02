import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MSHNet import MSHNet


class MultiScaleVerifier(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class MSCVMSHNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_channels=32,
        beta_max=0.1,
        evidence_threshold=0.0,
        detach_verifier_input=True,
        contrast_kernel=9,
    ):
        super().__init__()
        self.evidence_net = MSHNet(input_channels)
        self.verifier = MultiScaleVerifier(in_channels=10, hidden_channels=hidden_channels)
        self.beta_max = float(beta_max)
        self.evidence_threshold = float(evidence_threshold)
        self.detach_verifier_input = bool(detach_verifier_input)
        self.contrast_kernel = int(contrast_kernel)

    def _odd_kernel(self, kernel):
        kernel = max(3, int(kernel))
        return kernel + 1 if kernel % 2 == 0 else kernel

    def _upsample_like(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def local_contrast(self, x, size):
        kernel = self._odd_kernel(self.contrast_kernel)
        gray = x[:, :1]
        local_mean = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
        contrast = (gray - local_mean).abs()
        contrast = F.interpolate(contrast, size=size, mode="bilinear", align_corners=False)
        denom = contrast.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return contrast / denom

    def build_multiscale_tensor(self, masks, evidence_logit, local_contrast):
        aux_logits = [self._upsample_like(mask, evidence_logit) for mask in masks]
        if len(aux_logits) != 4:
            raise ValueError("MSCV requires four MSHNet auxiliary scale logits.")
        probs = [torch.sigmoid(logit) for logit in aux_logits]
        p_stack = torch.stack(probs, dim=0)
        p_mean = p_stack.mean(dim=0)
        p_std = p_stack.std(dim=0, unbiased=False)
        p_min = p_stack.min(dim=0).values
        p_max = p_stack.max(dim=0).values
        verifier_input = torch.cat(
            [
                probs[0],
                probs[1],
                probs[2],
                probs[3],
                p_mean,
                p_std,
                p_min,
                p_max,
                evidence_logit,
                local_contrast,
            ],
            dim=1,
        )
        return verifier_input, {
            "aux_logits": aux_logits,
            "p_mean": p_mean,
            "p_std": p_std,
            "p_min": p_min,
            "p_max": p_max,
        }

    def forward(self, x, warm_flag=True, beta=0.0, local_contrast=None, return_dict=True):
        masks, evidence_logit, feature = self.evidence_net(x, warm_flag=warm_flag, return_feature=True)
        if len(masks) != 4:
            masks, evidence_logit, feature = self.evidence_net(x, warm_flag=True, return_feature=True)
        if local_contrast is None:
            local_contrast = self.local_contrast(x, evidence_logit.shape[-2:])
        elif local_contrast.shape[-2:] != evidence_logit.shape[-2:]:
            local_contrast = F.interpolate(local_contrast, size=evidence_logit.shape[-2:], mode="bilinear", align_corners=False)

        verifier_input, ms = self.build_multiscale_tensor(masks, evidence_logit, local_contrast)
        verifier_input_for_head = verifier_input.detach() if self.detach_verifier_input else verifier_input
        validity_logit = self.verifier(verifier_input_for_head)
        validity_prob = torch.sigmoid(validity_logit)
        beta_eff = min(max(float(beta), 0.0), self.beta_max)
        evidence_candidate = F.relu(evidence_logit - self.evidence_threshold)
        suppression_map = beta_eff * (1.0 - validity_prob) * evidence_candidate
        final_logit = evidence_logit - suppression_map
        if not return_dict:
            return masks, final_logit
        return {
            "masks": masks,
            "aux_logits": ms["aux_logits"],
            "evidence_logit": evidence_logit,
            "target_logit": evidence_logit,
            "validity_logit": validity_logit,
            "validity_prob": validity_prob,
            "suppression_map": suppression_map,
            "final_logit": final_logit,
            "p_mean": ms["p_mean"],
            "p_std": ms["p_std"],
            "p_min": ms["p_min"],
            "p_max": ms["p_max"],
            "local_contrast": local_contrast,
            "verifier_input": verifier_input,
            "feature": feature,
            "beta": torch.tensor(beta_eff, device=final_logit.device),
        }
