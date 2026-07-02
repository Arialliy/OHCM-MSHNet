import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MSHNet import MSHNet


class EvidenceConditionedVerifier(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=32):
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


class ECDVMSHNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_channels=32,
        beta_max=0.1,
        evidence_threshold=0.0,
        detach_verifier_input=True,
        contrast_kernel=9,
        highpass_kernel=9,
    ):
        super().__init__()
        self.evidence_net = MSHNet(input_channels)
        self.verifier = EvidenceConditionedVerifier(in_channels=4, hidden_channels=hidden_channels)
        self.beta_max = float(beta_max)
        self.evidence_threshold = float(evidence_threshold)
        self.detach_verifier_input = bool(detach_verifier_input)
        self.contrast_kernel = int(contrast_kernel)
        self.highpass_kernel = int(highpass_kernel)

    def _odd_kernel(self, kernel):
        kernel = max(3, int(kernel))
        return kernel + 1 if kernel % 2 == 0 else kernel

    def local_contrast(self, x, size):
        kernel = self._odd_kernel(self.contrast_kernel)
        gray = x[:, :1]
        local_mean = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
        contrast = (gray - local_mean).abs()
        contrast = F.interpolate(contrast, size=size, mode="bilinear", align_corners=True)
        denom = contrast.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return contrast / denom

    def highpass_image(self, x, size):
        kernel = self._odd_kernel(self.highpass_kernel)
        gray = x[:, :1]
        lowpass = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
        highpass = gray - lowpass
        highpass = F.interpolate(highpass, size=size, mode="bilinear", align_corners=True)
        denom = highpass.flatten(1).abs().amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return highpass / denom

    def forward(self, x, warm_flag=True, beta=0.0, local_contrast=None, highpass_image=None, return_dict=True):
        masks, evidence_logit, feature = self.evidence_net(x, warm_flag=warm_flag, return_feature=True)
        evidence_prob = torch.sigmoid(evidence_logit)
        if local_contrast is None:
            local_contrast = self.local_contrast(x, evidence_logit.shape[-2:])
        elif local_contrast.shape[-2:] != evidence_logit.shape[-2:]:
            local_contrast = F.interpolate(local_contrast, size=evidence_logit.shape[-2:], mode="bilinear", align_corners=True)
        if highpass_image is None:
            highpass_image = self.highpass_image(x, evidence_logit.shape[-2:])
        elif highpass_image.shape[-2:] != evidence_logit.shape[-2:]:
            highpass_image = F.interpolate(highpass_image, size=evidence_logit.shape[-2:], mode="bilinear", align_corners=True)

        verifier_input = torch.cat([evidence_logit, evidence_prob, local_contrast, highpass_image], dim=1)
        if self.detach_verifier_input:
            verifier_input = verifier_input.detach()
        risk_logit = self.verifier(verifier_input)
        risk_prob = torch.sigmoid(risk_logit)
        beta_eff = min(max(float(beta), 0.0), self.beta_max)
        suppression_map = beta_eff * risk_prob * F.relu(evidence_logit - self.evidence_threshold)
        final_logit = evidence_logit - suppression_map
        if not return_dict:
            return masks, final_logit
        return {
            "masks": masks,
            "evidence_logit": evidence_logit,
            "target_logit": evidence_logit,
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
            "suppression_map": suppression_map,
            "final_logit": final_logit,
            "feature": feature,
            "beta": torch.tensor(beta_eff, device=final_logit.device),
        }
