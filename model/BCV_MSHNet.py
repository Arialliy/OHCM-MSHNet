import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MSHNet import MSHNet


class BackgroundContextBranch(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 5, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class ValidityVerifier(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32):
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


class BCVMSHNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_channels=32,
        beta_max=0.1,
        evidence_threshold=0.0,
        detach_verifier_input=True,
        contrast_kernel=9,
        validity_mode="learned",
        residual_theta=1.0,
        residual_temp=0.2,
        shape_theta=0.0,
        shape_temp=0.2,
    ):
        super().__init__()
        self.evidence_net = MSHNet(input_channels)
        self.bg_branch = BackgroundContextBranch(in_channels=input_channels, hidden_channels=hidden_channels)
        self.verifier = ValidityVerifier(in_channels=6, hidden_channels=hidden_channels)
        self.beta_max = float(beta_max)
        self.evidence_threshold = float(evidence_threshold)
        self.detach_verifier_input = bool(detach_verifier_input)
        self.contrast_kernel = int(contrast_kernel)
        self.validity_mode = str(validity_mode)
        if self.validity_mode not in ("learned", "residual_formula", "shape_formula"):
            raise ValueError("BCV validity_mode must be 'learned', 'residual_formula', or 'shape_formula'")
        self.residual_theta = float(residual_theta)
        self.residual_temp = max(float(residual_temp), 1e-6)
        self.shape_theta = float(shape_theta)
        self.shape_temp = max(float(shape_temp), 1e-6)

    def _odd_kernel(self, kernel):
        kernel = max(3, int(kernel))
        return kernel + 1 if kernel % 2 == 0 else kernel

    def local_contrast(self, x, size):
        kernel = self._odd_kernel(self.contrast_kernel)
        gray = x[:, :1]
        local_mean = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
        contrast = (gray - local_mean).abs()
        if contrast.shape[-2:] != size:
            contrast = F.interpolate(contrast, size=size, mode="bilinear", align_corners=False)
        denom = contrast.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return contrast / denom

    @staticmethod
    def background_gradient(bg):
        grad_x = torch.abs(bg[:, :, :, 1:] - bg[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.abs(bg[:, :, 1:, :] - bg[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return grad_x + grad_y

    def residual_shape_score_map(self, residual_norm):
        local_mean = F.avg_pool2d(residual_norm, kernel_size=7, stride=1, padding=3)
        local_peak = residual_norm - F.avg_pool2d(residual_norm, kernel_size=3, stride=1, padding=1)
        return local_peak + 0.5 * (residual_norm - local_mean)

    def forward(self, x, warm_flag=True, beta=0.0, local_contrast=None, shape_score_map=None, return_dict=True):
        masks, evidence_logit, feature = self.evidence_net(x, warm_flag=warm_flag, return_feature=True)
        if len(masks) != 4:
            masks, evidence_logit, feature = self.evidence_net(x, warm_flag=True, return_feature=True)
        p_e = torch.sigmoid(evidence_logit)
        bg = self.bg_branch(x[:, :1])
        if bg.shape[-2:] != evidence_logit.shape[-2:]:
            bg_for_verifier = F.interpolate(bg, size=evidence_logit.shape[-2:], mode="bilinear", align_corners=False)
            img_for_verifier = F.interpolate(x[:, :1], size=evidence_logit.shape[-2:], mode="bilinear", align_corners=False)
        else:
            bg_for_verifier = bg
            img_for_verifier = x[:, :1]
        residual = torch.abs(img_for_verifier - bg_for_verifier)
        residual_norm = residual / (residual.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        if local_contrast is None:
            local_contrast = self.local_contrast(x, evidence_logit.shape[-2:])
        elif local_contrast.shape[-2:] != evidence_logit.shape[-2:]:
            local_contrast = F.interpolate(local_contrast, size=evidence_logit.shape[-2:], mode="bilinear", align_corners=False)
        bg_grad = self.background_gradient(bg_for_verifier)

        verifier_input = torch.cat(
            [evidence_logit, p_e, residual, residual_norm, local_contrast, bg_grad],
            dim=1,
        )
        verifier_input_for_head = verifier_input.detach() if self.detach_verifier_input else verifier_input
        if shape_score_map is None:
            shape_score_map = self.residual_shape_score_map(residual_norm)
        elif shape_score_map.shape[-2:] != evidence_logit.shape[-2:]:
            shape_score_map = F.interpolate(shape_score_map, size=evidence_logit.shape[-2:], mode="bilinear", align_corners=False)

        if self.validity_mode == "shape_formula":
            validity_prob = torch.sigmoid((shape_score_map - self.shape_theta) / self.shape_temp)
            validity_logit = torch.logit(validity_prob.clamp(1e-4, 1.0 - 1e-4))
        elif self.validity_mode == "residual_formula":
            validity_prob = torch.sigmoid((residual_norm - self.residual_theta) / self.residual_temp)
            validity_logit = torch.logit(validity_prob.clamp(1e-4, 1.0 - 1e-4))
        else:
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
            "evidence_logit": evidence_logit,
            "target_logit": evidence_logit,
            "background": bg_for_verifier,
            "background_raw": bg,
            "residual": residual,
            "residual_norm": residual_norm,
            "local_contrast": local_contrast,
            "background_gradient": bg_grad,
            "shape_score_map": shape_score_map,
            "validity_logit": validity_logit,
            "validity_prob": validity_prob,
            "suppression_map": suppression_map,
            "final_logit": final_logit,
            "verifier_input": verifier_input,
            "feature": feature,
            "beta": torch.tensor(beta_eff, device=final_logit.device),
        }
