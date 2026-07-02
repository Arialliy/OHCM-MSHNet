import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import json
import math
import numpy as np
from utils.component_geometry import build_center_heatmap, build_core_boundary_maps, component_area_bins
from utils.local_peak import select_background_peaks
from utils.mscv_candidate import build_mscv_candidate_mask

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss


def HC_LLloss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    batch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]

    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h

    smooth = 1e-8

    for i in range(batch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()
        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (
            torch.square(
                torch.arctan(pred_centery / (pred_centerx + smooth))
                - torch.arctan(target_centery / (target_centerx + smooth))
            )
        )

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = torch.min(pred_length, target_length) / (
            torch.max(pred_length, target_length) + smooth
        )

        loss = loss + (1 - length_loss + angle_loss) / batch_size

    return loss


class HCSLSIoULoss(nn.Module):
    """
    SLS loss for HCNet.
    Input is logits.
    """

    def __init__(self):
        super(HCSLSIoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch=10, epoch=0, with_shape=True):
        target = target.float()

        if target.shape[-2:] != pred_log.shape[-2:]:
            target = F.interpolate(
                target,
                size=pred_log.shape[-2:],
                mode='nearest'
            )

        pred = torch.sigmoid(pred_log)

        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (
            torch.max(pred_sum, target_sum) + dis + smooth + 1e-8
        )

        iou = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth + 1e-8
        )

        if epoch > warm_epoch:
            siou_loss = alpha * iou

            if with_shape:
                loss = 1 - siou_loss.mean() + HC_LLloss(pred, target)
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - iou.mean()

        return loss


class HardClutterLoss(nn.Module):
    """
    Online Hard Clutter Mining Loss.
    Select top-k predicted probabilities in GT background and suppress them.
    """

    def __init__(
        self,
        topk_ratio=0.01,
        dilate_kernel=7,
        gamma=2.0,
        eps=1e-6,
    ):
        super(HardClutterLoss, self).__init__()

        self.topk_ratio = topk_ratio
        self.dilate_kernel = dilate_kernel
        self.gamma = gamma
        self.eps = eps

    def dilate_mask(self, gt_mask):
        padding = self.dilate_kernel // 2

        gt_dilate = F.max_pool2d(
            gt_mask.float(),
            kernel_size=self.dilate_kernel,
            stride=1,
            padding=padding,
        )

        gt_dilate = (gt_dilate > 0).float()
        return gt_dilate

    def select_topk_hard_pixels(self, pred_prob, background_mask):
        B, C, H, W = pred_prob.shape
        hard_mask = torch.zeros_like(pred_prob)

        flat_prob = pred_prob.view(B, -1)
        flat_bg = background_mask.view(B, -1)

        for b in range(B):
            valid_bg = flat_bg[b] > 0

            if valid_bg.sum() < 1:
                continue

            valid_score = flat_prob[b][valid_bg]

            k = max(1, int(valid_score.numel() * self.topk_ratio))
            k = min(k, valid_score.numel())

            _, topk_idx_in_valid = torch.topk(valid_score, k=k, largest=True)

            valid_indices = torch.nonzero(valid_bg, as_tuple=False).squeeze(1)
            selected_indices = valid_indices[topk_idx_in_valid]

            hard_flat = torch.zeros_like(flat_prob[b])
            hard_flat[selected_indices] = 1.0

            hard_mask[b] = hard_flat.view(C, H, W)

        return hard_mask

    def forward(self, pred_logit, gt_mask):
        gt_mask = gt_mask.float()

        if gt_mask.shape[-2:] != pred_logit.shape[-2:]:
            gt_mask = F.interpolate(
                gt_mask,
                size=pred_logit.shape[-2:],
                mode='nearest'
            )

        pred_prob = torch.sigmoid(pred_logit)

        gt_dilate = self.dilate_mask(gt_mask)
        background_mask = 1.0 - gt_dilate

        with torch.no_grad():
            hard_mask = self.select_topk_hard_pixels(
                pred_prob.detach(),
                background_mask,
            )

        hard_prob = pred_prob * hard_mask

        loss = torch.pow(hard_prob, self.gamma).sum() / (
            hard_mask.sum() + self.eps
        )

        stats = {
            'hard_pixels': hard_mask.sum().detach(),
            'hard_ratio': hard_mask.mean().detach(),
            'hard_prob_mean': (
                hard_prob.sum() / (hard_mask.sum() + self.eps)
            ).detach(),
        }

        return loss, stats


class HCNetLoss(nn.Module):
    """
    HCNet v1 loss:

        L = L_sls + lambda_hc * L_hc
    """

    def __init__(
        self,
        sls_warm_epoch=10,
        hc_warm_epoch=10,
        lambda_hc=0.0,
        topk_ratio=0.01,
        dilate_kernel=7,
        gamma=2.0,
    ):
        super(HCNetLoss, self).__init__()

        self.sls_warm_epoch = sls_warm_epoch
        self.hc_warm_epoch = hc_warm_epoch
        self.lambda_hc = lambda_hc

        self.base_loss = HCSLSIoULoss()
        self.hc_loss = HardClutterLoss(
            topk_ratio=topk_ratio,
            dilate_kernel=dilate_kernel,
            gamma=gamma,
        )

    def forward(self, pred_logit, gt_mask, epoch=0):
        loss_sls = self.base_loss(
            pred_logit,
            gt_mask,
            warm_epoch=self.sls_warm_epoch,
            epoch=epoch,
            with_shape=True,
        )

        if self.lambda_hc > 0 and epoch > self.hc_warm_epoch:
            loss_hc, hc_stats = self.hc_loss(pred_logit, gt_mask)
        else:
            loss_hc = torch.tensor(
                0.0,
                device=pred_logit.device,
                dtype=pred_logit.dtype,
            )
            hc_stats = {
                'hard_pixels': torch.tensor(0.0, device=pred_logit.device),
                'hard_ratio': torch.tensor(0.0, device=pred_logit.device),
                'hard_prob_mean': torch.tensor(0.0, device=pred_logit.device),
            }

        loss_total = loss_sls + self.lambda_hc * loss_hc

        return {
            'total': loss_total,
            'sls': loss_sls.detach(),
            'hc': loss_hc.detach(),
            'hard_pixels': hc_stats['hard_pixels'],
            'hard_ratio': hc_stats['hard_ratio'],
            'hard_prob_mean': hc_stats['hard_prob_mean'],
        }


class FocalLogitLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logit, target):
        target = target.float()
        if target.shape[-2:] != pred_logit.shape[-2:]:
            target = F.interpolate(target, size=pred_logit.shape[-2:], mode='nearest')
        bce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
        prob = torch.sigmoid(pred_logit)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * torch.pow(1.0 - pt, self.gamma) * bce).mean()


class OHEMLogitLoss(nn.Module):
    def __init__(self, topk_ratio=0.01):
        super().__init__()
        self.topk_ratio = topk_ratio

    def forward(self, pred_logit, target):
        target = target.float()
        if target.shape[-2:] != pred_logit.shape[-2:]:
            target = F.interpolate(target, size=pred_logit.shape[-2:], mode='nearest')
        loss_map = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
        losses = []
        for b in range(pred_logit.shape[0]):
            flat_loss = loss_map[b].flatten()
            flat_target = target[b].flatten()
            pos_loss = flat_loss[flat_target > 0]
            neg_loss = flat_loss[flat_target <= 0]
            if neg_loss.numel() > 0:
                k = max(1, int(neg_loss.numel() * self.topk_ratio))
                k = min(k, neg_loss.numel())
                neg_loss = torch.topk(neg_loss, k=k, largest=True).values
            if pos_loss.numel() > 0:
                losses.append(torch.cat([pos_loss, neg_loss]).mean())
            elif neg_loss.numel() > 0:
                losses.append(neg_loss.mean())
        if not losses:
            return pred_logit.sum() * 0.0
        return torch.stack(losses).mean()


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask.float()
    k = 2 * int(radius) + 1
    return (F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=int(radius)) > 0).float()


def far_background_mask(gt_mask, radius):
    return 1.0 - dilate_mask(gt_mask.float(), radius)


def select_online_reliability_negatives(
    evidence_logit,
    gt_mask,
    far_radius=5,
    q=0.01,
    min_k=16,
    max_k=512,
):
    """Select detached high-evidence far-background pixels as reliability negatives."""
    with torch.no_grad():
        if gt_mask.shape[-2:] != evidence_logit.shape[-2:]:
            gt_mask = F.interpolate(gt_mask.float(), size=evidence_logit.shape[-2:], mode='nearest')
        else:
            gt_mask = gt_mask.float()

        prob = torch.sigmoid(evidence_logit.detach())
        far_bg = far_background_mask(gt_mask, far_radius)
        score = prob * far_bg

        neg_mask = torch.zeros_like(score)
        counts = []

        for b in range(score.shape[0]):
            valid = far_bg[b].flatten() > 0
            n_valid = int(valid.sum().item())
            if n_valid < 1:
                counts.append(0)
                continue

            k = int(n_valid * float(q))
            k = max(int(min_k), k)
            k = min(int(max_k), k, n_valid)

            values = score[b].flatten()[valid]
            top_idx = torch.topk(values, k=k, largest=True).indices
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()

            flat = torch.zeros_like(score[b].flatten())
            flat[valid_idx[top_idx]] = 1.0
            neg_mask[b] = flat.view_as(score[b])
            counts.append(k)

        return neg_mask, counts


def binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    k = 2 * int(radius) + 1
    return (F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=int(radius)) > 0).float()


def topk_mask(score: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """Return a boolean mask for top-k scores inside each image's valid mask."""
    out = torch.zeros_like(score, dtype=torch.bool)
    if int(k) <= 0:
        return out
    flat_score = score.reshape(score.shape[0], -1)
    flat_valid = valid.reshape(valid.shape[0], -1).bool()
    flat_out = out.reshape(out.shape[0], -1)
    for b in range(score.shape[0]):
        idx = torch.nonzero(flat_valid[b], as_tuple=False).flatten()
        if idx.numel() < 1:
            continue
        kk = min(int(k), int(idx.numel()))
        top = torch.topk(flat_score[b, idx], k=kk, largest=True).indices
        flat_out[b, idx[top]] = True
    return out


def build_training_masks(target, target_dilate: int, far_dilate: int):
    target = target.float()
    target_mask = target > 0
    target_support = binary_dilate(target, target_dilate) > 0
    boundary_mask = target_support & (~target_mask)
    far_bg_mask = (binary_dilate(target, far_dilate) <= 0)
    return target_mask, boundary_mask, far_bg_mask


def select_topk_far_background(logits, far_bg_mask, topk_ratio: float):
    with torch.no_grad():
        far_bg_mask = far_bg_mask.bool()
        score = logits.detach()
        out = torch.zeros_like(score, dtype=torch.bool)
        flat_score = score.reshape(score.shape[0], -1)
        flat_valid = far_bg_mask.reshape(far_bg_mask.shape[0], -1)
        flat_out = out.reshape(out.shape[0], -1)
        for b in range(score.shape[0]):
            idx = torch.nonzero(flat_valid[b], as_tuple=False).flatten()
            if idx.numel() < 1:
                continue
            k = max(1, int(idx.numel() * float(topk_ratio)))
            k = min(k, idx.numel())
            selected = torch.topk(flat_score[b, idx], k=k, largest=True).indices
            flat_out[b, idx[selected]] = True
        return out


class PFRLoss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        ohem_ratio=0.01,
        lambda_far_neg: float = 0.5,
        lambda_target_protect: float = 1.0,
        lambda_boundary_protect: float = 0.5,
        lambda_residual_sparse: float = 0.01,
        far_topk_ratio: float = 0.005,
        target_dilate: int = 3,
        far_dilate: int = 9,
    ):
        super().__init__()
        self.base_loss = MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            ohem_ratio=ohem_ratio,
        )
        self.lambda_far_neg = float(lambda_far_neg)
        self.lambda_target_protect = float(lambda_target_protect)
        self.lambda_boundary_protect = float(lambda_boundary_protect)
        self.lambda_residual_sparse = float(lambda_residual_sparse)
        self.far_topk_ratio = float(far_topk_ratio)
        self.target_dilate = int(target_dilate)
        self.far_dilate = int(far_dilate)

    def forward(self, outputs, target, epoch=0):
        z_final = outputs["logits"]
        z_e = outputs["evidence_logits"].detach()
        delta = outputs["delta_logits"]
        masks = outputs["masks"]

        if target.shape[-2:] != z_final.shape[-2:]:
            target = F.interpolate(target.float(), size=z_final.shape[-2:], mode='nearest')
        else:
            target = target.float()

        base_out = self.base_loss(masks, z_final, target, epoch=epoch)
        loss_seg = base_out["total"] if isinstance(base_out, dict) else base_out
        target_mask, boundary_mask, far_bg_mask = build_training_masks(
            target,
            target_dilate=self.target_dilate,
            far_dilate=self.far_dilate,
        )
        far_hard_mask = select_topk_far_background(z_e, far_bg_mask, self.far_topk_ratio)

        zero = z_final.sum() * 0.0
        loss_far_neg = F.softplus(z_final[far_hard_mask]).mean() if far_hard_mask.any() else zero
        loss_target_protect = F.relu(-delta[target_mask]).mean() if target_mask.any() else zero
        loss_boundary_protect = F.relu(-delta[boundary_mask]).mean() if boundary_mask.any() else zero
        loss_residual_sparse = delta.abs().mean()
        loss_total = (
            loss_seg
            + self.lambda_far_neg * loss_far_neg
            + self.lambda_target_protect * loss_target_protect
            + self.lambda_boundary_protect * loss_boundary_protect
            + self.lambda_residual_sparse * loss_residual_sparse
        )
        return {
            "total": loss_total,
            "seg": loss_seg.detach(),
            "far_neg": loss_far_neg.detach(),
            "target_protect": loss_target_protect.detach(),
            "boundary_protect": loss_boundary_protect.detach(),
            "residual_sparse": loss_residual_sparse.detach(),
            "far_hard_pixels": far_hard_mask.sum().detach(),
            "target_pixels": target_mask.sum().detach(),
            "boundary_pixels": boundary_mask.sum().detach(),
            "delta_abs_mean": delta.abs().mean().detach(),
            "delta_target_neg_mean": (
                F.relu(-delta[target_mask]).mean().detach() if target_mask.any() else torch.tensor(0.0, device=z_final.device)
            ),
        }


class ERDMSHNetLoss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        ohem_ratio=0.01,
        lambda_evidence=0.2,
        lambda_gate_pos=0.05,
        lambda_gate_neg=0.20,
        gate_start_epoch=20,
        gate_ramp_epochs=30,
        gate_target_radius=2,
        gate_far_radius=5,
        gate_neg_q=0.01,
        gate_neg_min_k=16,
        gate_neg_max_k=512,
    ):
        super().__init__()
        self.mshnet_warm_epoch = int(mshnet_warm_epoch)
        self.lambda_evidence = float(lambda_evidence)
        self.lambda_gate_pos = float(lambda_gate_pos)
        self.lambda_gate_neg = float(lambda_gate_neg)
        self.gate_start_epoch = int(gate_start_epoch)
        self.gate_ramp_epochs = int(gate_ramp_epochs)
        self.gate_target_radius = int(gate_target_radius)
        self.gate_far_radius = int(gate_far_radius)
        self.gate_neg_q = float(gate_neg_q)
        self.gate_neg_min_k = int(gate_neg_min_k)
        self.gate_neg_max_k = int(gate_neg_max_k)
        self.final_loss = MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            lambda_variant=0.2,
            ohem_ratio=ohem_ratio,
        )
        self.evidence_ohem = OHEMLogitLoss(topk_ratio=ohem_ratio)

    def _gate_weight(self, epoch):
        if epoch <= self.gate_start_epoch:
            return 0.0
        progress = float(epoch - self.gate_start_epoch) / float(max(1, self.gate_ramp_epochs))
        return min(1.0, max(0.0, progress))

    def _masked_bce(self, logit, label_value, mask):
        mask = mask.float()
        mask_sum = mask.sum()
        if float(mask_sum.detach().cpu()) < 1.0:
            return logit.sum() * 0.0
        target = torch.full_like(logit, float(label_value))
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        return (loss * mask).sum() / mask_sum.clamp_min(1.0)

    def forward(self, output, gt_mask, epoch=0):
        masks = output['masks']
        evidence_logit = output['evidence_logit']
        reliability_logit = output['reliability_logit']
        final_logit = output['final_logit']

        if gt_mask.shape[-2:] != final_logit.shape[-2:]:
            gt_mask = F.interpolate(gt_mask.float(), size=final_logit.shape[-2:], mode='nearest')
        else:
            gt_mask = gt_mask.float()

        final_out = self.final_loss(masks, final_logit, gt_mask, epoch=epoch)
        loss_final = final_out['total'] if isinstance(final_out, dict) else final_out
        loss_evidence = self.evidence_ohem(evidence_logit, gt_mask)

        gate_pos_mask = dilate_mask(gt_mask, self.gate_target_radius)
        loss_gate_pos = self._masked_bce(reliability_logit, 1.0, gate_pos_mask)

        gate_neg_mask, neg_counts = select_online_reliability_negatives(
            evidence_logit=evidence_logit,
            gt_mask=gt_mask,
            far_radius=self.gate_far_radius,
            q=self.gate_neg_q,
            min_k=self.gate_neg_min_k,
            max_k=self.gate_neg_max_k,
        )
        loss_gate_neg = self._masked_bce(reliability_logit, 0.0, gate_neg_mask)

        gate_w = self._gate_weight(epoch)
        loss_total = (
            loss_final
            + self.lambda_evidence * loss_evidence
            + gate_w * self.lambda_gate_pos * loss_gate_pos
            + gate_w * self.lambda_gate_neg * loss_gate_neg
        )

        return {
            'total': loss_total,
            'sls': loss_final.detach(),
            'evidence': loss_evidence.detach(),
            'gate_pos': loss_gate_pos.detach(),
            'gate_neg': loss_gate_neg.detach(),
            'gate_w': torch.tensor(gate_w, device=final_logit.device),
            'gate_neg_pixels': gate_neg_mask.sum().detach(),
            'gate_pos_pixels': gate_pos_mask.sum().detach(),
            'gate_neg_per_image_min': torch.tensor(min(neg_counts) if neg_counts else 0, device=final_logit.device),
            'gate_neg_per_image_mean': torch.tensor(
                float(sum(neg_counts)) / max(1, len(neg_counts)),
                device=final_logit.device,
            ),
        }


class ERDMSHNetV3Loss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        ohem_ratio=0.01,
        lambda_evidence=0.2,
        far_radius=7,
        target_protect_radius=2,
        neg_topk_ratio=0.01,
        lambda_protect_pos=0.5,
        lambda_protect_neg=0.25,
        lambda_clutter_pos=0.5,
        lambda_clutter_neg=0.25,
        lambda_preserve=0.5,
        preserve_margin=0.02,
    ):
        super().__init__()
        self.far_radius = int(far_radius)
        self.target_protect_radius = int(target_protect_radius)
        self.neg_topk_ratio = float(neg_topk_ratio)
        self.lambda_evidence = float(lambda_evidence)
        self.lambda_protect_pos = float(lambda_protect_pos)
        self.lambda_protect_neg = float(lambda_protect_neg)
        self.lambda_clutter_pos = float(lambda_clutter_pos)
        self.lambda_clutter_neg = float(lambda_clutter_neg)
        self.lambda_preserve = float(lambda_preserve)
        self.preserve_margin = float(preserve_margin)
        self.final_loss = MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            lambda_variant=0.2,
            ohem_ratio=ohem_ratio,
        )
        self.evidence_ohem = OHEMLogitLoss(topk_ratio=ohem_ratio)

    @staticmethod
    def _resize_target(target, ref):
        target = target.float()
        if target.ndim == 3:
            target = target[:, None]
        if target.shape[-2:] != ref.shape[-2:]:
            target = F.interpolate(target, size=ref.shape[-2:], mode='nearest')
        return target

    @staticmethod
    def _masked_bce(logit, label_value, mask):
        mask = mask.float()
        denom = mask.sum()
        if float(denom.detach().cpu()) < 1.0:
            return logit.sum() * 0.0
        target = torch.full_like(logit, float(label_value))
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        return (loss * mask).sum() / denom.clamp_min(1.0)

    @staticmethod
    def _masked_mean(value, mask):
        mask = mask.float()
        denom = mask.sum()
        if float(denom.detach().cpu()) < 1.0:
            return torch.tensor(0.0, device=value.device, dtype=value.dtype)
        return (value * mask).sum() / denom.clamp_min(1.0)

    def select_online_negatives(self, evidence_logits, y):
        with torch.no_grad():
            target = self._resize_target(y, evidence_logits)
            prob = torch.sigmoid(evidence_logits.detach())
            far_bg = (1.0 - binary_dilate(target, self.far_radius)).bool()
            neg_mask = torch.zeros_like(prob, dtype=torch.bool)
            counts = []
            flat_prob = prob.reshape(prob.shape[0], -1)
            flat_far = far_bg.reshape(far_bg.shape[0], -1)
            flat_target = target.reshape(target.shape[0], -1)
            flat_out = neg_mask.reshape(neg_mask.shape[0], -1)
            for b in range(prob.shape[0]):
                valid = flat_far[b] > 0
                num_valid = int(valid.sum().item())
                if num_valid < 1:
                    counts.append(0)
                    continue
                num_bg = int((flat_target[b] <= 0).sum().item())
                budget = max(1, int(math.floor(float(num_bg) * self.neg_topk_ratio)))
                budget = min(budget, num_valid)
                valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
                top = torch.topk(flat_prob[b, valid_idx], k=budget, largest=True).indices
                flat_out[b, valid_idx[top]] = True
                counts.append(int(budget))
            return neg_mask, counts

    def forward(self, outputs, y, epoch=0):
        z_f = outputs.get('logits', outputs.get('final_logits', outputs.get('final_logit')))
        z_e = outputs.get('evidence_logits', outputs.get('evidence_logit'))
        z_t = outputs.get('protection_logits', outputs.get('protection_logit'))
        z_c = outputs.get('clutter_logits', outputs.get('clutter_logit'))
        if z_f is None or z_e is None or z_t is None or z_c is None:
            raise KeyError("ERDMSHNetV3Loss requires logits/evidence/protection/clutter outputs.")

        y = self._resize_target(y, z_f)
        masks = outputs.get('masks', [])
        loss_final_out = self.final_loss(masks, z_f, y, epoch=epoch)
        loss_final = loss_final_out['total'] if isinstance(loss_final_out, dict) else loss_final_out
        loss_evidence = self.evidence_ohem(z_e, y)

        target_support = binary_dilate(y, self.target_protect_radius).clamp(0.0, 1.0)
        far_bg = (1.0 - binary_dilate(y, self.far_radius)).clamp(0.0, 1.0)
        neg_mask, neg_counts = self.select_online_negatives(z_e, y)
        neg_float = neg_mask.float()

        loss_protect_pos = self._masked_bce(z_t, 1.0, target_support)
        loss_protect_neg = self._masked_bce(z_t, 0.0, neg_float)
        loss_clutter_pos = self._masked_bce(z_c, 1.0, neg_float)
        loss_clutter_neg = self._masked_bce(z_c, 0.0, target_support)

        p_e = torch.sigmoid(z_e.detach())
        p_f = torch.sigmoid(z_f)
        preserve = torch.pow(F.relu((p_e - self.preserve_margin) - p_f), 2.0)
        loss_preserve = (preserve * target_support).sum() / target_support.sum().clamp_min(1.0)

        total = (
            loss_final
            + self.lambda_evidence * loss_evidence
            + self.lambda_protect_pos * loss_protect_pos
            + self.lambda_protect_neg * loss_protect_neg
            + self.lambda_clutter_pos * loss_clutter_pos
            + self.lambda_clutter_neg * loss_clutter_neg
            + self.lambda_preserve * loss_preserve
        )

        protection = outputs.get('protection', torch.sigmoid(z_t))
        clutter = outputs.get('clutter', torch.sigmoid(z_c))
        suppression = outputs.get('suppression', torch.clamp(z_e - z_f, min=0.0))
        device = z_f.device
        return {
            'total': total,
            'loss_total': total.detach(),
            'loss_final': loss_final.detach(),
            'loss_evidence': loss_evidence.detach(),
            'loss_protect_pos': loss_protect_pos.detach(),
            'loss_protect_neg': loss_protect_neg.detach(),
            'loss_clutter_pos': loss_clutter_pos.detach(),
            'loss_clutter_neg': loss_clutter_neg.detach(),
            'loss_preserve': loss_preserve.detach(),
            'online_neg_pixels': neg_float.sum().detach(),
            'mean_online_neg_pixels': torch.tensor(
                float(sum(neg_counts)) / max(1, len(neg_counts)),
                device=device,
            ),
            'mean_protection_target': self._masked_mean(protection.detach(), target_support).detach(),
            'mean_protection_far_bg': self._masked_mean(protection.detach(), far_bg).detach(),
            'mean_clutter_target': self._masked_mean(clutter.detach(), target_support).detach(),
            'mean_clutter_far_bg': self._masked_mean(clutter.detach(), far_bg).detach(),
            'mean_suppression_target': self._masked_mean(suppression.detach(), target_support).detach(),
            'mean_suppression_far_bg': self._masked_mean(suppression.detach(), far_bg).detach(),
            'target_support_pixels': target_support.sum().detach(),
            'far_bg_pixels': far_bg.sum().detach(),
        }


class TopKNegativeLogitLoss(nn.Module):
    def __init__(self, topk_ratio=0.01, dilate_kernel=7):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.dilate_kernel = dilate_kernel

    def forward(self, pred_logit, target):
        target = target.float()
        if target.shape[-2:] != pred_logit.shape[-2:]:
            target = F.interpolate(target, size=pred_logit.shape[-2:], mode='nearest')
        padding = self.dilate_kernel // 2
        safe_target = F.max_pool2d(target, kernel_size=self.dilate_kernel, stride=1, padding=padding)
        background = safe_target <= 0
        prob = torch.sigmoid(pred_logit.detach())
        bce_zero = F.binary_cross_entropy_with_logits(pred_logit, torch.zeros_like(pred_logit), reduction='none')
        losses = []
        for b in range(pred_logit.shape[0]):
            valid = background[b].flatten()
            if valid.sum() < 1:
                continue
            score = prob[b].flatten()[valid]
            loss = bce_zero[b].flatten()[valid]
            k = max(1, int(score.numel() * self.topk_ratio))
            k = min(k, score.numel())
            idx = torch.topk(score, k=k, largest=True).indices
            losses.append(loss[idx].mean())
        if not losses:
            return pred_logit.sum() * 0.0
        return torch.stack(losses).mean()


class SelfPerturbationStabilityLoss(nn.Module):
    """
    Teacher-free perturbation stability loss.

    The second prediction is produced by the same model under a label-preserving
    perturbation and aligned back to the original coordinates before loss
    computation. The detached instability map only selects hard negatives; the
    negative BCE still backpropagates through the current model logits.
    """

    def __init__(
        self,
        dilate_radius=5,
        candidate_tau=0.3,
        candidate_topk_ratio=0.0,
        candidate_topk_metric='confidence',
        candidate_min_metric=None,
        candidate_min_confidence=0.0,
        candidate_fallback_topk_ratio=0.0,
        candidate_expand_radius=0,
        candidate_expand_min_confidence=0.0,
        target_margin_quantile=0.85,
        target_margin_temp=0.01,
        target_margin_min=0.0,
        rerank_strict_fallback=True,
        budget_q=0.1,
        kmax=256,
        eta=1.0,
        mode='sps',
        disable_far_mask=False,
        adaptive_radius=True,
        radius_kappa=1.0,
        radius_r0=2.0,
        radius_min=3,
        radius_max=9,
        target_safe=False,
        target_safe_u_low=0.02,
        target_safe_u_high=0.08,
        target_safe_conf_min=0.55,
        target_safe_conf_floor=0.35,
        target_safe_alpha_floor=0.0,
        eps=1e-6,
    ):
        super().__init__()
        self.dilate_radius = int(dilate_radius)
        self.candidate_tau = float(candidate_tau)
        self.candidate_topk_ratio = float(candidate_topk_ratio)
        self.candidate_topk_metric = str(candidate_topk_metric)
        self.candidate_min_metric = None if candidate_min_metric is None else float(candidate_min_metric)
        self.candidate_min_confidence = float(candidate_min_confidence)
        self.candidate_fallback_topk_ratio = float(candidate_fallback_topk_ratio)
        self.candidate_expand_radius = int(candidate_expand_radius)
        self.candidate_expand_min_confidence = float(candidate_expand_min_confidence)
        self.target_margin_quantile = float(target_margin_quantile)
        self.target_margin_temp = float(target_margin_temp)
        self.target_margin_min = float(target_margin_min)
        self.rerank_strict_fallback = bool(rerank_strict_fallback)
        self.budget_q = float(budget_q)
        self.kmax = int(kmax)
        self.eta = float(eta)
        self.mode = str(mode)
        self.disable_far_mask = bool(disable_far_mask)
        self.adaptive_radius = bool(adaptive_radius)
        self.radius_kappa = float(radius_kappa)
        self.radius_r0 = float(radius_r0)
        self.radius_min = int(radius_min)
        self.radius_max = int(radius_max)
        self.target_safe = bool(target_safe)
        self.target_safe_u_low = float(target_safe_u_low)
        self.target_safe_u_high = float(target_safe_u_high)
        self.target_safe_conf_min = float(target_safe_conf_min)
        self.target_safe_conf_floor = float(target_safe_conf_floor)
        self.target_safe_alpha_floor = float(target_safe_alpha_floor)
        self.eps = float(eps)

    def _resize(self, tensor, size, mode='nearest'):
        if tensor.shape[-2:] == size:
            return tensor
        if mode == 'bilinear':
            return F.interpolate(tensor, size=size, mode=mode, align_corners=True)
        return F.interpolate(tensor, size=size, mode=mode)

    def _dilate_fixed(self, mask, radius):
        radius = int(radius)
        if radius <= 0:
            return mask.float()
        kernel = radius * 2 + 1
        return (F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0).float()

    def _dilate(self, mask):
        if self.dilate_radius <= 0:
            return mask.float()
        if not self.adaptive_radius:
            return self._dilate_fixed(mask, self.dilate_radius)
        dilated = torch.zeros_like(mask.float())
        for b in range(mask.shape[0]):
            area = float(mask[b].sum().detach().cpu())
            if area <= 0:
                continue
            radius = int(math.ceil(self.radius_kappa * math.sqrt(area / math.pi)) + self.radius_r0)
            radius = max(self.radius_min, min(self.radius_max, radius))
            dilated[b:b + 1] = self._dilate_fixed(mask[b:b + 1], radius)
        return dilated

    def _far_background(self, target):
        if self.disable_far_mask:
            return (target <= 0).float()
        return 1.0 - self._dilate(target)

    def align_back(self, tensor, op):
        if op == 'hflip':
            return torch.flip(tensor, dims=[-1])
        if op == 'vflip':
            return torch.flip(tensor, dims=[-2])
        if op == 'transpose':
            return tensor.transpose(-1, -2)
        if op == 'hvflip':
            return torch.flip(tensor, dims=[-2, -1])
        if op in (None, 'identity'):
            return tensor
        raise ValueError(f"Unsupported SPS perturbation op: {op}")

    def _select_hard_mask(self, score, candidate):
        selected = torch.zeros_like(score)
        candidate_counts = []
        selected_counts = []
        for b in range(score.shape[0]):
            valid = candidate[b].flatten() > 0
            num_valid = int(valid.sum().item())
            candidate_counts.append(num_valid)
            if num_valid < 1:
                selected_counts.append(0)
                continue
            values = score[b].flatten()[valid]
            k = max(1, int(math.floor(self.budget_q * num_valid)))
            k = min(k, max(1, self.kmax), num_valid)
            top_idx = torch.topk(values, k=k, largest=True).indices
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            flat = torch.zeros_like(score[b].flatten())
            flat[valid_idx[top_idx]] = 1.0
            selected[b] = flat.view_as(score[b])
            selected_counts.append(k)
        return selected, candidate_counts, selected_counts

    def _target_margin_signal(self, instability, target):
        if target is None:
            raise ValueError(f"SPS candidate_topk_metric={self.candidate_topk_metric} requires target masks.")
        q = max(0.0, min(1.0, self.target_margin_quantile))
        margin = max(0.0, self.target_margin_min)
        signal = torch.zeros_like(instability)
        for b in range(instability.shape[0]):
            pos = target[b].flatten() > 0
            if int(pos.sum().item()) < 1:
                ref = torch.tensor(0.0, device=instability.device, dtype=instability.dtype)
            else:
                values = torch.sort(instability[b].flatten()[pos].detach()).values
                idx = int(math.ceil(q * max(0, int(values.numel()) - 1)))
                idx = max(0, min(idx, int(values.numel()) - 1))
                ref = values[idx]
            signal[b] = torch.clamp(instability[b] - ref - margin, min=0.0)
        return signal

    def _target_contrast_signal(self, instability, target):
        if target is None:
            raise ValueError(f"SPS candidate_topk_metric={self.candidate_topk_metric} requires target masks.")
        q = max(0.0, min(1.0, self.target_margin_quantile))
        margin = max(0.0, self.target_margin_min)
        temp = max(self.eps, float(self.target_margin_temp))
        signal = torch.zeros_like(instability)
        for b in range(instability.shape[0]):
            pos = target[b].flatten() > 0
            if int(pos.sum().item()) < 1:
                ref = torch.tensor(0.0, device=instability.device, dtype=instability.dtype)
            else:
                values = torch.sort(instability[b].flatten()[pos].detach()).values
                idx = int(math.ceil(q * max(0, int(values.numel()) - 1)))
                idx = max(0, min(idx, int(values.numel()) - 1))
                ref = values[idx]
            signal[b] = torch.sigmoid((instability[b] - ref - margin) / temp)
        return signal

    def _candidate_metric(self, confidence, instability, hardness, target=None):
        if self.candidate_topk_metric == 'confidence':
            return confidence
        if self.candidate_topk_metric == 'instability':
            return instability
        if self.candidate_topk_metric == 'sps_score':
            return hardness * torch.pow(instability + self.eps, self.eta)
        if self.candidate_topk_metric == 'target_margin_instability':
            return self._target_margin_signal(instability, target)
        if self.candidate_topk_metric == 'target_margin_sps_score':
            signal = self._target_margin_signal(instability, target)
            return hardness * torch.pow(signal, self.eta)
        if self.candidate_topk_metric == 'target_contrast_instability':
            return self._target_contrast_signal(instability, target)
        if self.candidate_topk_metric == 'target_contrast_sps_score':
            signal = self._target_contrast_signal(instability, target)
            return hardness * torch.pow(signal, self.eta)
        raise ValueError(f"Unsupported SPS candidate_topk_metric: {self.candidate_topk_metric}")

    def _fill_topk_candidates(self, candidate, metric, far_bg, ratio, only_empty, confidence=None):
        if ratio <= 0:
            return candidate
        if metric is None:
            raise ValueError(f"SPS candidate_topk_metric={self.candidate_topk_metric} requires a metric tensor.")
        flat_metric = metric.flatten(start_dim=1)
        flat_far = far_bg.flatten(start_dim=1) > 0
        flat_candidate = candidate.flatten(start_dim=1)
        for b in range(candidate.shape[0]):
            if only_empty and bool(flat_candidate[b].any()):
                continue
            valid = flat_far[b]
            if self.candidate_min_metric is not None:
                valid = valid & (flat_metric[b] > self.candidate_min_metric)
            if confidence is not None and self.candidate_min_confidence > 0:
                valid = valid & (confidence.flatten(start_dim=1)[b] >= self.candidate_min_confidence)
            num_valid = int(valid.sum().item())
            if num_valid < 1:
                continue
            k = max(1, int(math.floor(ratio * num_valid)))
            k = min(k, num_valid)
            values = flat_metric[b][valid]
            top_idx = torch.topk(values, k=k, largest=True).indices
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            flat_candidate[b, valid_idx[top_idx]] = True
        return candidate

    def _candidate_mask(self, confidence, far_bg, instability=None, hardness=None, target=None, metric=None):
        if metric is None:
            metric = self._candidate_metric(confidence, instability, hardness, target=target)
        if self.candidate_topk_ratio > 0:
            candidate = torch.zeros_like(confidence, dtype=torch.bool)
            candidate = self._fill_topk_candidates(candidate, metric, far_bg, self.candidate_topk_ratio, only_empty=False, confidence=confidence)
            return self._expand_candidate_mask(candidate, far_bg, confidence)

        candidate = (far_bg > 0) & (confidence > self.candidate_tau)
        candidate = self._fill_topk_candidates(
            candidate,
            metric,
            far_bg,
            self.candidate_fallback_topk_ratio,
            only_empty=True,
            confidence=confidence,
        )
        return self._expand_candidate_mask(candidate, far_bg, confidence)

    def _expand_candidate_mask(self, candidate, far_bg, confidence=None):
        radius = int(self.candidate_expand_radius)
        if radius <= 0:
            return candidate
        kernel = 2 * radius + 1
        expanded = F.max_pool2d(candidate.float(), kernel_size=kernel, stride=1, padding=radius) > 0
        expanded = expanded & (far_bg > 0)
        if confidence is not None and self.candidate_expand_min_confidence > 0:
            expanded = expanded & (confidence >= self.candidate_expand_min_confidence)
        return expanded

    def _target_safety(self, target, instability, confidence):
        scales = []
        target_u = []
        target_conf = []
        for b in range(target.shape[0]):
            pos = target[b].flatten() > 0
            if int(pos.sum().item()) < 1:
                scales.append(torch.tensor(1.0, device=target.device, dtype=target.dtype))
                target_u.append(torch.tensor(0.0, device=target.device, dtype=target.dtype))
                target_conf.append(torch.tensor(1.0, device=target.device, dtype=target.dtype))
                continue

            u_mean = instability[b].flatten()[pos].mean()
            conf_mean = confidence[b].flatten()[pos].mean()
            target_u.append(u_mean)
            target_conf.append(conf_mean)

            if not self.target_safe:
                scales.append(torch.tensor(1.0, device=target.device, dtype=target.dtype))
                continue

            u_span = max(self.eps, self.target_safe_u_high - self.target_safe_u_low)
            conf_span = max(self.eps, self.target_safe_conf_min - self.target_safe_conf_floor)
            u_risk = torch.clamp((u_mean - self.target_safe_u_low) / u_span, min=0.0, max=1.0)
            conf_risk = torch.clamp((self.target_safe_conf_min - conf_mean) / conf_span, min=0.0, max=1.0)
            risk = torch.maximum(u_risk, conf_risk)
            floor = max(0.0, min(1.0, self.target_safe_alpha_floor))
            scales.append(1.0 - (1.0 - floor) * risk)

        return torch.stack(scales), torch.stack(target_u), torch.stack(target_conf)

    def _rerank_fallback_candidate(self, fallback_pool, metric, budget):
        if not self.rerank_strict_fallback:
            return fallback_pool.clone(), 1

        num_valid = int(fallback_pool.sum().item())
        if num_valid < 1:
            return fallback_pool.clone(), 1

        ratio_k = 0
        if self.candidate_fallback_topk_ratio > 0:
            ratio_k = int(math.floor(self.candidate_fallback_topk_ratio * num_valid))
        k = max(int(budget), ratio_k, 1)
        k = min(k, num_valid)

        valid_idx = torch.nonzero(fallback_pool, as_tuple=False).flatten()
        values = metric[fallback_pool]
        top_idx = torch.topk(values, k=k, largest=True, sorted=False).indices
        candidate = torch.zeros_like(fallback_pool, dtype=torch.bool)
        candidate[valid_idx[top_idx]] = True
        return candidate, 1

    def forward(self, final_logit, perturb_logit, gt_mask, op='hflip'):
        perturb_logit = self.align_back(perturb_logit, op)
        if perturb_logit.shape[-2:] != final_logit.shape[-2:]:
            perturb_logit = self._resize(perturb_logit, final_logit.shape[-2:], mode='bilinear')

        gt = self._resize(gt_mask.float(), final_logit.shape[-2:], mode='nearest')
        far_bg = self._far_background(gt)

        prob = torch.sigmoid(final_logit)
        perturb_prob = torch.sigmoid(perturb_logit)
        if self.mode == 'global_consistency':
            loss = torch.abs(prob - perturb_prob).mean()
            zeros = torch.tensor(0.0, device=final_logit.device)
            ones = torch.tensor(1.0, device=final_logit.device)
            return loss, {
                'sps_hard_pixels': zeros,
                'sps_candidate_pixels': zeros,
                'sps_weight_sum': zeros,
                'sps_instability_mean': torch.abs(prob.detach() - perturb_prob.detach()).mean().detach(),
                'sps_conf_mean': torch.maximum(prob.detach(), perturb_prob.detach()).mean().detach(),
                'sps_score_mean': zeros,
                'sps_neg_loss': zeros,
                'sps_ohem_jaccard': zeros,
                'sps_fallback_images': zeros,
                'sps_target_alpha_scale': ones,
                'sps_target_instability_mean': zeros,
                'sps_target_conf_mean': ones,
            }

        bce_a = F.binary_cross_entropy_with_logits(final_logit, torch.zeros_like(final_logit), reduction='none')
        bce_b = F.binary_cross_entropy_with_logits(perturb_logit, torch.zeros_like(perturb_logit), reduction='none')
        hardness = 0.5 * (bce_a + bce_b)
        with torch.no_grad():
            instability = torch.abs(prob.detach() - perturb_prob.detach())
            confidence = torch.maximum(prob.detach(), perturb_prob.detach())
            hardness_detached = hardness.detach()
            candidate = self._candidate_mask(
                confidence,
                far_bg,
                instability=instability,
                hardness=hardness_detached,
                target=gt,
            )
            if self.mode == 'confidence_only':
                score = far_bg * hardness_detached
            elif self.mode == 'instability_only':
                score = far_bg * torch.pow(instability + self.eps, self.eta)
            elif self.mode == 'target_margin':
                if self.candidate_topk_metric.startswith('target_margin_'):
                    signal_map = self._target_margin_signal(instability, gt)
                elif self.candidate_topk_metric.startswith('target_contrast_'):
                    signal_map = self._target_contrast_signal(instability, gt)
                else:
                    signal_map = instability
                score = far_bg * torch.pow(signal_map + self.eps, self.eta)
            elif self.candidate_topk_metric.startswith('target_margin_'):
                margin_signal = self._target_margin_signal(instability, gt)
                score = far_bg * hardness_detached * torch.pow(margin_signal + self.eps, self.eta)
            elif self.candidate_topk_metric.startswith('target_contrast_'):
                contrast_signal = self._target_contrast_signal(instability, gt)
                score = far_bg * hardness_detached * torch.pow(contrast_signal + self.eps, self.eta)
            else:
                score = far_bg * hardness_detached * torch.pow(instability + self.eps, self.eta)
            hard_mask, candidate_counts, selected_counts = self._select_hard_mask(score, candidate)
            denom = hard_mask.sum()

        zero = final_logit.sum() * 0.0
        if denom <= 0:
            loss = zero
        else:
            loss = (hardness * hard_mask).sum() / (denom + self.eps)
        selected_float = hard_mask.float()
        selected_denom = selected_float.sum()
        stats = {
            'sps_hard_pixels': selected_denom.detach(),
            'sps_candidate_pixels': torch.tensor(float(sum(candidate_counts)), device=final_logit.device),
            'sps_weight_sum': score.detach().sum(),
            'sps_instability_mean': (
                (instability * selected_float).sum() / (selected_denom + self.eps)
            ).detach(),
            'sps_conf_mean': (
                (confidence * selected_float).sum() / (selected_denom + self.eps)
            ).detach(),
            'sps_score_mean': (
                (score.detach() * selected_float).sum() / (selected_denom + self.eps)
            ).detach(),
            'sps_neg_loss': loss.detach(),
            'sps_ohem_jaccard': torch.tensor(0.0, device=final_logit.device),
            'sps_fallback_images': torch.tensor(0.0, device=final_logit.device),
            'sps_target_alpha_scale': torch.tensor(1.0, device=final_logit.device),
            'sps_target_instability_mean': torch.tensor(0.0, device=final_logit.device),
            'sps_target_conf_mean': torch.tensor(1.0, device=final_logit.device),
        }
        return loss, stats

    def rerank_ohem_loss(self, final_logit, perturb_logit, gt_mask, op='hflip', topk_ratio=0.01, alpha=1.0):
        """Replace OHEM negative mining with fixed-budget stability-guided reranking.

        The negative budget matches OHEM: each image selects
        floor(num_background_pixels * topk_ratio) negatives. Stability only
        changes the ranking score; selected logits still receive gradients.
        """
        perturb_logit = self.align_back(perturb_logit, op)
        if perturb_logit.shape[-2:] != final_logit.shape[-2:]:
            perturb_logit = self._resize(perturb_logit, final_logit.shape[-2:], mode='bilinear')

        gt = self._resize(gt_mask.float(), final_logit.shape[-2:], mode='nearest')
        target = gt.float()
        far_bg = self._far_background(target)

        weak_target_loss = F.binary_cross_entropy_with_logits(final_logit, target, reduction='none')
        weak_neg_loss = F.binary_cross_entropy_with_logits(final_logit, torch.zeros_like(final_logit), reduction='none')
        pert_neg_loss = F.binary_cross_entropy_with_logits(perturb_logit, torch.zeros_like(perturb_logit), reduction='none')
        neg_loss = 0.5 * (weak_neg_loss + pert_neg_loss)

        with torch.no_grad():
            weak_prob = torch.sigmoid(final_logit.detach())
            pert_prob = torch.sigmoid(perturb_logit.detach())
            instability = torch.abs(weak_prob - pert_prob)
            confidence = torch.maximum(weak_prob, pert_prob)
            hardness = neg_loss.detach()
            candidate_metric = self._candidate_metric(
                confidence,
                instability,
                hardness,
                target=target,
            )
            candidate_map = self._candidate_mask(
                confidence,
                far_bg,
                instability=instability,
                hardness=hardness,
                target=target,
                metric=candidate_metric,
            )
            if self.candidate_topk_metric.startswith('target_margin_'):
                rerank_signal = self._target_margin_signal(instability, target)
            elif self.candidate_topk_metric.startswith('target_contrast_'):
                rerank_signal = self._target_contrast_signal(instability, target)
            else:
                rerank_signal = instability
            alpha_scale, target_u_mean, target_conf_mean = self._target_safety(
                target,
                instability,
                torch.minimum(weak_prob, pert_prob),
            )

        losses = []
        selected_masks = []
        candidate_counts = []
        selected_counts = []
        fallback_counts = []
        jaccard_values = []
        for b in range(final_logit.shape[0]):
            flat_target = target[b].flatten()
            pos = flat_target > 0
            neg = flat_target <= 0
            num_neg = int(neg.sum().item())
            if num_neg < 1:
                if bool(pos.any()):
                    losses.append(weak_target_loss[b].flatten()[pos].mean())
                continue

            budget = max(1, int(math.floor(float(topk_ratio) * num_neg)))
            budget = min(budget, num_neg)

            flat_conf = confidence[b].flatten()
            image_alpha = float(alpha) * float(alpha_scale[b].detach().cpu())
            disabled_rerank = image_alpha <= 0 or self.mode == 'none'
            flat_far = (far_bg[b].flatten() > 0) & neg
            if disabled_rerank:
                candidate = neg.clone()
            else:
                candidate = (candidate_map[b].flatten() > 0) & neg
            fallback_used = 0
            if not disabled_rerank and int(candidate.sum().item()) < budget:
                fallback_metric = candidate_metric[b].flatten()
                candidate, fallback_used = self._rerank_fallback_candidate(
                    flat_far,
                    fallback_metric,
                    budget,
                )
            if not disabled_rerank and int(candidate.sum().item()) < 1:
                disabled_rerank = True
                candidate = neg.clone()
                fallback_used = 1
            candidate_counts.append(int(candidate.sum().item()))
            fallback_counts.append(fallback_used)

            flat_hardness = weak_neg_loss[b].flatten() if disabled_rerank else hardness[b].flatten()
            base_score = flat_hardness[candidate]
            if disabled_rerank:
                score = base_score
            else:
                if self.mode == 'confidence_only':
                    signal = flat_conf[candidate]
                elif self.mode in ('instability_only', 'target_margin'):
                    signal = rerank_signal[b].flatten()[candidate]
                    base_score = torch.ones_like(base_score)
                else:
                    signal = rerank_signal[b].flatten()[candidate]
                norm_signal = signal / (signal.mean() + self.eps)
                score = base_score * (1.0 + image_alpha * norm_signal)

            k = min(budget, int(score.numel()))
            chosen_local = torch.topk(score.detach(), k=k, largest=True, sorted=False).indices
            candidate_idx = torch.nonzero(candidate, as_tuple=False).flatten()
            chosen = candidate_idx[chosen_local]

            ohem_values = flat_hardness[neg]
            ohem_k = min(budget, int(ohem_values.numel()))
            ohem_local = torch.topk(ohem_values, k=ohem_k, largest=True, sorted=False).indices
            neg_idx = torch.nonzero(neg, as_tuple=False).flatten()
            ohem_chosen = neg_idx[ohem_local]
            inter = len(set(chosen.detach().cpu().tolist()).intersection(ohem_chosen.detach().cpu().tolist()))
            union = max(1, len(set(chosen.detach().cpu().tolist()).union(ohem_chosen.detach().cpu().tolist())))
            jaccard_values.append(float(inter) / float(union))

            mask_flat = torch.zeros_like(flat_target, dtype=final_logit.dtype)
            mask_flat[chosen] = 1.0
            selected_masks.append(mask_flat.view_as(target[b]))
            selected_counts.append(k)

            if disabled_rerank:
                selected_neg = torch.topk(
                    weak_neg_loss[b].flatten()[neg],
                    k=k,
                    largest=True,
                ).values
            else:
                selected_neg = neg_loss[b].flatten()[chosen]
            if bool(pos.any()):
                pos_loss = weak_target_loss[b].flatten()[pos]
                losses.append(torch.cat([pos_loss, selected_neg]).mean())
            else:
                losses.append(selected_neg.mean())

        if losses:
            loss = torch.stack(losses).mean()
        else:
            loss = final_logit.sum() * 0.0

        if selected_masks:
            selected = torch.stack(selected_masks, dim=0)
            selected_denom = selected.sum()
        else:
            selected = torch.zeros_like(final_logit)
            selected_denom = torch.tensor(0.0, device=final_logit.device)

        stats = {
            'sps_hard_pixels': selected_denom.detach(),
            'sps_candidate_pixels': torch.tensor(float(sum(candidate_counts)), device=final_logit.device),
            'sps_weight_sum': selected_denom.detach(),
            'sps_instability_mean': ((instability * selected).sum() / (selected_denom + self.eps)).detach(),
            'sps_conf_mean': ((confidence * selected).sum() / (selected_denom + self.eps)).detach(),
            'sps_score_mean': torch.tensor(float(sum(jaccard_values) / max(1, len(jaccard_values))), device=final_logit.device),
            'sps_neg_loss': loss.detach(),
            'sps_ohem_jaccard': torch.tensor(float(sum(jaccard_values) / max(1, len(jaccard_values))), device=final_logit.device),
            'sps_fallback_images': torch.tensor(float(sum(fallback_counts)), device=final_logit.device),
            'sps_target_alpha_scale': alpha_scale.mean().detach(),
            'sps_target_instability_mean': target_u_mean.mean().detach(),
            'sps_target_conf_mean': target_conf_mean.mean().detach(),
        }
        return loss, stats


def _parse_odd_scales(value):
    if value is None:
        return [3, 5, 7]
    if isinstance(value, str):
        items = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    scales = []
    for item in items:
        scale = int(round(float(item)))
        scale = max(1, scale)
        if scale % 2 == 0:
            lower = max(1, scale - 1)
            upper = scale + 1
            scale = lower if abs(lower - float(item)) <= abs(upper - float(item)) else upper
        if scale not in scales:
            scales.append(scale)
    return sorted(scales) if scales else [3, 5, 7]


class TargetScaleRegionLoss(nn.Module):
    """
    TSR-OHEM training-only region objective.

    Hard background regions are fixed target-scale windows selected without a
    probability threshold. Only selected coordinates are detached; final logits
    inside the selected regions still receive gradients.
    """

    def __init__(
        self,
        target_scales='3,5,7',
        beta=0.5,
        topk=3,
        nms_iou=0.3,
        weight_temp=0.2,
        target_temp=0.25,
        hard_temp=0.25,
        rank_temp=0.5,
        margin=0.5,
        topq=0.25,
        dilation_radius=0,
        use_consensus=True,
        loss_mode='rank',
    ):
        super().__init__()
        self.target_scales = _parse_odd_scales(target_scales)
        self.beta = float(beta)
        self.topk = int(topk)
        self.nms_iou = float(nms_iou)
        self.weight_temp = float(weight_temp)
        self.target_temp = float(target_temp)
        self.hard_temp = float(hard_temp)
        self.rank_temp = float(rank_temp)
        self.margin = float(margin)
        self.topq = float(topq)
        self.dilation_radius = int(dilation_radius)
        self.use_consensus = bool(use_consensus)
        self.loss_mode = str(loss_mode)

    def _resize(self, tensor, size, mode='nearest'):
        if tensor.shape[-2:] == size:
            return tensor
        if mode == 'bilinear':
            return F.interpolate(tensor, size=size, mode=mode, align_corners=True)
        return F.interpolate(tensor, size=size, mode=mode)

    def _auto_dilation_radius(self):
        median_scale = self.target_scales[len(self.target_scales) // 2]
        return max(3, int(math.ceil(float(median_scale) / 2.0)))

    def _safe_background(self, gt_mask, size):
        gt = self._resize(gt_mask.float(), size, mode='nearest')
        radius = self.dilation_radius if self.dilation_radius > 0 else self._auto_dilation_radius()
        if radius > 0:
            gt = F.max_pool2d(gt, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return (gt <= 0).float()

    @staticmethod
    def _box_iou(a, b):
        ay0, ay1, ax0, ax1 = a
        by0, by1, bx0, bx1 = b
        inter_y0 = max(ay0, by0)
        inter_y1 = min(ay1, by1)
        inter_x0 = max(ax0, bx0)
        inter_x1 = min(ax1, bx1)
        inter_h = max(0, inter_y1 - inter_y0)
        inter_w = max(0, inter_x1 - inter_x0)
        inter = inter_h * inter_w
        area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
        area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
        union = area_a + area_b - inter
        return float(inter) / float(union + 1e-6)

    @staticmethod
    def _window_box(center_y, center_x, scale, height, width, require_full):
        half = scale // 2
        y0 = int(center_y) - half
        y1 = int(center_y) + half + 1
        x0 = int(center_x) - half
        x1 = int(center_x) + half + 1
        if require_full and (y0 < 0 or x0 < 0 or y1 > height or x1 > width):
            return None
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(height, y1)
        x1 = min(width, x1)
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)

    def _nms(self, candidates):
        selected = []
        for cand in sorted(candidates, key=lambda item: item['score'], reverse=True):
            if len(selected) >= self.topk:
                break
            if all(self._box_iou(cand['box'], kept['box']) <= self.nms_iou for kept in selected):
                selected.append(cand)
        return selected

    def _assign_weights(self, regions):
        if not regions:
            return regions
        temp = max(self.weight_temp, 1e-6)
        max_score = max(region['score'] for region in regions)
        weights = [math.exp((region['score'] - max_score) / temp) for region in regions]
        denom = sum(weights) + 1e-12
        for region, weight in zip(regions, weights):
            region['weight'] = float(weight / denom)
        return regions

    def mine_hard_regions(self, final_logit, masks, gt_mask):
        size = final_logit.shape[-2:]
        with torch.no_grad():
            logits = [final_logit.detach()]
            if self.use_consensus:
                logits = [self._resize(mask.detach(), size, mode='bilinear') for mask in masks] + logits
            probs = [torch.sigmoid(logit) for logit in logits]
            prob_stack = torch.stack(probs, dim=0)
            mean_prob = prob_stack.mean(dim=0)
            if prob_stack.shape[0] > 1:
                uncertainty = prob_stack.std(dim=0, unbiased=False)
            else:
                uncertainty = torch.zeros_like(mean_prob)
            safe_bg = self._safe_background(gt_mask, size)

            batch_regions = []
            score_values = []
            uncertainty_values = []
            scale_values = []
            for b in range(final_logit.shape[0]):
                candidates = []
                for scale in self.target_scales:
                    pad = scale // 2
                    avg_resp = F.avg_pool2d(mean_prob, kernel_size=scale, stride=1, padding=pad)
                    max_resp = F.max_pool2d(mean_prob, kernel_size=scale, stride=1, padding=pad)
                    avg_uncertainty = F.avg_pool2d(uncertainty, kernel_size=scale, stride=1, padding=pad)
                    score_map = avg_resp * max_resp - self.beta * avg_uncertainty

                    kernel = torch.ones((1, 1, scale, scale), device=final_logit.device, dtype=final_logit.dtype)
                    safe_count = F.conv2d(safe_bg, kernel, stride=1, padding=pad)
                    valid = safe_count >= float(scale * scale - 1e-4)
                    local_max = score_map >= (
                        F.max_pool2d(score_map, kernel_size=2 * scale + 1, stride=1, padding=scale) - 1e-12
                    )
                    valid = valid & local_max

                    flat_valid = valid[b, 0].flatten()
                    if flat_valid.sum().item() <= 0:
                        continue
                    flat_scores = score_map[b, 0].flatten()
                    valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
                    valid_scores = flat_scores[valid_indices]
                    keep = min(valid_scores.numel(), max(self.topk * 20, self.topk))
                    top_values, top_order = torch.topk(valid_scores, k=keep, largest=True)
                    top_indices = valid_indices[top_order]
                    for value, flat_idx in zip(top_values.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
                        y = int(flat_idx // size[1])
                        x = int(flat_idx % size[1])
                        box = self._window_box(y, x, scale, size[0], size[1], require_full=True)
                        if box is None:
                            continue
                        y0, y1, x0, x1 = box
                        u_mean = float(uncertainty[b, 0, y0:y1, x0:x1].mean().detach().cpu())
                        candidates.append({
                            'batch': b,
                            'y': y,
                            'x': x,
                            'scale': int(scale),
                            'box': box,
                            'score': float(value),
                            'uncertainty': u_mean,
                            'weight': 0.0,
                        })
                selected = self._assign_weights(self._nms(candidates))
                batch_regions.append(selected)
                for region in selected:
                    score_values.append(region['score'])
                    uncertainty_values.append(region['uncertainty'])
                    scale_values.append(region['scale'])

            empty_count = sum(1 for regions in batch_regions if not regions)
            stats = {
                'hard_regions': torch.tensor(float(sum(len(regions) for regions in batch_regions)), device=final_logit.device),
                'empty_region_ratio': torch.tensor(float(empty_count) / max(1, len(batch_regions)), device=final_logit.device),
                'hard_region_score_mean': torch.tensor(float(np.mean(score_values)) if score_values else 0.0, device=final_logit.device),
                'hard_region_uncertainty_mean': torch.tensor(float(np.mean(uncertainty_values)) if uncertainty_values else 0.0, device=final_logit.device),
                'hard_region_scale_mean': torch.tensor(float(np.mean(scale_values)) if scale_values else 0.0, device=final_logit.device),
            }
            return batch_regions, stats

    def build_target_regions(self, gt_mask, size, device):
        from skimage import measure

        with torch.no_grad():
            gt = self._resize(gt_mask.float(), size, mode='nearest')
            gt_np = (gt[:, 0].detach().cpu().numpy() > 0.5)
            batch_regions = []
            scale_values = []
            for b in range(gt_np.shape[0]):
                label = measure.label(gt_np[b].astype(np.uint8), connectivity=2)
                regions = []
                for component in measure.regionprops(label):
                    area = float(component.area)
                    if area <= 0:
                        continue
                    diameter = 2.0 * math.sqrt(area / math.pi)
                    scale = min(self.target_scales, key=lambda item: abs(float(item) - diameter))
                    cy, cx = component.centroid
                    box = self._window_box(int(round(cy)), int(round(cx)), scale, size[0], size[1], require_full=False)
                    if box is None:
                        continue
                    regions.append({
                        'batch': b,
                        'y': int(round(cy)),
                        'x': int(round(cx)),
                        'scale': int(scale),
                        'box': box,
                        'area': area,
                        'diameter': diameter,
                    })
                    scale_values.append(scale)
                batch_regions.append(regions)
            empty_count = sum(1 for regions in batch_regions if not regions)
            stats = {
                'target_regions': torch.tensor(float(sum(len(regions) for regions in batch_regions)), device=device),
                'empty_target_ratio': torch.tensor(float(empty_count) / max(1, len(batch_regions)), device=device),
                'target_region_scale_mean': torch.tensor(float(np.mean(scale_values)) if scale_values else 0.0, device=device),
            }
            return batch_regions, stats

    def region_response(self, final_logit, region):
        b = region['batch']
        y0, y1, x0, x1 = region['box']
        patch = final_logit[b, 0, y0:y1, x0:x1].reshape(-1)
        if patch.numel() == 0:
            return final_logit.sum() * 0.0
        k = max(1, int(math.floor(self.topq * patch.numel())))
        k = min(k, patch.numel())
        return torch.topk(patch, k=k, largest=True).values.mean()

    def _rank_loss(self, final_logit, hard_regions, target_regions):
        losses = []
        target_weak_values = []
        hard_values = []
        gap_values = []
        for b in range(final_logit.shape[0]):
            if not hard_regions[b] or not target_regions[b]:
                continue
            target_scores = torch.stack([self.region_response(final_logit, region) for region in target_regions[b]])
            t_weak = -self.target_temp * (
                torch.logsumexp(-target_scores / max(self.target_temp, 1e-6), dim=0)
                - math.log(max(1, target_scores.numel()))
            )
            hard_scores = torch.stack([self.region_response(final_logit, region) for region in hard_regions[b]])
            weights = torch.tensor(
                [max(region.get('weight', 0.0), 1e-12) for region in hard_regions[b]],
                dtype=final_logit.dtype,
                device=final_logit.device,
            )
            h_hard = self.hard_temp * torch.logsumexp(
                torch.log(weights) + hard_scores / max(self.hard_temp, 1e-6),
                dim=0,
            )
            loss = self.rank_temp * F.softplus((h_hard - t_weak + self.margin) / max(self.rank_temp, 1e-6))
            losses.append(loss)
            target_weak_values.append(t_weak.detach())
            hard_values.append(h_hard.detach())
            gap_values.append((t_weak - h_hard).detach())
        return losses, target_weak_values, hard_values, gap_values

    def _target_weak_score(self, final_logit, regions):
        target_scores = torch.stack([self.region_response(final_logit, region) for region in regions])
        return -self.target_temp * (
            torch.logsumexp(-target_scores / max(self.target_temp, 1e-6), dim=0)
            - math.log(max(1, target_scores.numel()))
        )

    def _asymmetric_rank_loss(self, final_logit, hard_regions, target_regions):
        losses = []
        target_weak_values = []
        hard_values = []
        gap_values = []
        for b in range(final_logit.shape[0]):
            if not hard_regions[b] or not target_regions[b]:
                continue
            t_weak = self._target_weak_score(final_logit, target_regions[b]).detach()
            weighted_losses = []
            weights = []
            hard_scores = []
            for region in hard_regions[b]:
                h_score = self.region_response(final_logit, region)
                weight = float(region.get('weight', 1.0))
                weighted_losses.append(
                    weight * self.rank_temp * F.softplus((h_score - t_weak + self.margin) / max(self.rank_temp, 1e-6))
                )
                weights.append(weight)
                hard_scores.append(h_score.detach())
            denom = max(sum(weights), 1e-6)
            losses.append(torch.stack(weighted_losses).sum() / denom)
            hard_mean = torch.stack(hard_scores).mean()
            target_weak_values.append(t_weak.detach())
            hard_values.append(hard_mean.detach())
            gap_values.append((t_weak - hard_mean).detach())
        return losses, target_weak_values, hard_values, gap_values

    def _negative_bce_loss(self, final_logit, hard_regions):
        losses = []
        hard_values = []
        for b in range(final_logit.shape[0]):
            for region in hard_regions[b]:
                y0, y1, x0, x1 = region['box']
                patch = final_logit[b:b + 1, :, y0:y1, x0:x1]
                if patch.numel() == 0:
                    continue
                loss = F.binary_cross_entropy_with_logits(patch, torch.zeros_like(patch), reduction='mean')
                losses.append(loss * float(region.get('weight', 1.0)))
                hard_values.append(self.region_response(final_logit, region).detach())
        return losses, [], hard_values, []

    def forward(self, masks, final_logit, gt_mask):
        zero = final_logit.sum() * 0.0
        hard_regions, hard_stats = self.mine_hard_regions(final_logit, masks, gt_mask)
        target_regions, target_stats = self.build_target_regions(gt_mask, final_logit.shape[-2:], final_logit.device)

        if self.loss_mode == 'neg_bce':
            losses, target_weak_values, hard_values, gap_values = self._negative_bce_loss(final_logit, hard_regions)
        elif self.loss_mode == 'asym_rank':
            losses, target_weak_values, hard_values, gap_values = self._asymmetric_rank_loss(final_logit, hard_regions, target_regions)
        else:
            losses, target_weak_values, hard_values, gap_values = self._rank_loss(final_logit, hard_regions, target_regions)

        region_loss = torch.stack(losses).mean() if losses else zero
        stats = {}
        stats.update(hard_stats)
        stats.update(target_stats)
        stats['target_weak_logit'] = (
            torch.stack(target_weak_values).mean() if target_weak_values else torch.tensor(0.0, device=final_logit.device)
        )
        stats['hard_region_logit'] = (
            torch.stack(hard_values).mean() if hard_values else torch.tensor(0.0, device=final_logit.device)
        )
        stats['region_logit_gap'] = (
            torch.stack(gap_values).mean() if gap_values else torch.tensor(0.0, device=final_logit.device)
        )
        stats['valid_region_loss_images'] = torch.tensor(float(len(losses)), device=final_logit.device)
        return region_loss, stats


class PersistentClutterRegionLoss(TargetScaleRegionLoss):
    def __init__(self, bank_path, max_regions=3, min_weight=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.bank_path = str(bank_path)
        self.max_regions = int(max_regions)
        self.min_weight = float(min_weight)
        self._bank_by_image = {}
        self._load_bank()

    def _load_bank(self):
        with open(self.bank_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = data.get('records', data if isinstance(data, list) else [])
        by_image = {}
        for row in records:
            image_id = str(row.get('image_id', row.get('image', '')))
            if not image_id:
                continue
            cy = int(round(float(row.get('center_y', row.get('y', 0)))))
            cx = int(round(float(row.get('center_x', row.get('x', 0)))))
            scale = int(round(float(row.get('window_size', row.get('scale', 5)))))
            box = row.get('box', None)
            if box is not None and len(box) == 4:
                y0, y1, x0, x1 = [int(v) for v in box]
            else:
                half = scale // 2
                y0, y1, x0, x1 = cy - half, cy + half + 1, cx - half, cx + half + 1
            score = float(row.get('teacher_score', row.get('score', 0.0)))
            weight = float(row.get('weight', max(score, self.min_weight)))
            by_image.setdefault(image_id, []).append({
                'y': cy,
                'x': cx,
                'scale': scale,
                'box': (y0, y1, x0, x1),
                'score': score,
                'weight': max(weight, self.min_weight),
                'teacher_fp_flag': bool(row.get('teacher_fp_flag', False)),
            })
        for image_id, regions in by_image.items():
            regions.sort(key=lambda item: item.get('score', 0.0), reverse=True)
            by_image[image_id] = self._assign_weights(regions[:max(0, self.max_regions)])
        self._bank_by_image = by_image

    def _transform_box(self, box, height, width, ops):
        y0, y1, x0, x1 = box
        if int(ops[0]) == 1:
            y0, y1 = height - y1, height - y0
        if int(ops[1]) == 1:
            x0, x1 = width - x1, width - x0
        if int(ops[2]) == 1:
            y0, y1, x0, x1 = x0, x1, y0, y1
            height, width = width, height
        y0 = max(0, min(height, int(y0)))
        y1 = max(0, min(height, int(y1)))
        x0 = max(0, min(width, int(x0)))
        x1 = max(0, min(width, int(x1)))
        if y1 <= y0 or x1 <= x0:
            return None
        return y0, y1, x0, x1

    def lookup_regions(self, image_ids, aug_ops, size, device):
        height, width = int(size[0]), int(size[1])
        if image_ids is None:
            return [[] for _ in range(0)]
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        if aug_ops is None:
            aug_ops_np = np.zeros((len(image_ids), 3), dtype=np.int64)
        elif torch.is_tensor(aug_ops):
            aug_ops_np = aug_ops.detach().cpu().numpy()
        else:
            aug_ops_np = np.asarray(aug_ops)
        batch_regions = []
        score_values = []
        scale_values = []
        for b, image_id in enumerate(image_ids):
            regions = []
            ops = aug_ops_np[b] if aug_ops_np.ndim > 1 else aug_ops_np
            for region in self._bank_by_image.get(str(image_id), [])[:self.max_regions]:
                box = self._transform_box(region['box'], height, width, ops)
                if box is None:
                    continue
                item = dict(region)
                item['batch'] = b
                item['box'] = box
                y0, y1, x0, x1 = box
                item['y'] = int(round((y0 + y1 - 1) / 2.0))
                item['x'] = int(round((x0 + x1 - 1) / 2.0))
                regions.append(item)
                score_values.append(float(item.get('score', 0.0)))
                scale_values.append(float(item.get('scale', 0.0)))
            batch_regions.append(regions)
        empty_count = sum(1 for regions in batch_regions if not regions)
        stats = {
            'hard_regions': torch.tensor(float(sum(len(regions) for regions in batch_regions)), device=device),
            'empty_region_ratio': torch.tensor(float(empty_count) / max(1, len(batch_regions)), device=device),
            'hard_region_score_mean': torch.tensor(float(np.mean(score_values)) if score_values else 0.0, device=device),
            'hard_region_uncertainty_mean': torch.tensor(0.0, device=device),
            'hard_region_scale_mean': torch.tensor(float(np.mean(scale_values)) if scale_values else 0.0, device=device),
        }
        return batch_regions, stats

    def forward(self, masks, final_logit, gt_mask, image_ids=None, aug_ops=None):
        zero = final_logit.sum() * 0.0
        batch_size = final_logit.shape[0]
        if image_ids is None:
            hard_regions = [[] for _ in range(batch_size)]
            hard_stats = {
                'hard_regions': torch.tensor(0.0, device=final_logit.device),
                'empty_region_ratio': torch.tensor(1.0, device=final_logit.device),
                'hard_region_score_mean': torch.tensor(0.0, device=final_logit.device),
                'hard_region_uncertainty_mean': torch.tensor(0.0, device=final_logit.device),
                'hard_region_scale_mean': torch.tensor(0.0, device=final_logit.device),
            }
        else:
            hard_regions, hard_stats = self.lookup_regions(image_ids, aug_ops, final_logit.shape[-2:], final_logit.device)
        target_regions, target_stats = self.build_target_regions(gt_mask, final_logit.shape[-2:], final_logit.device)
        if self.loss_mode == 'neg_bce':
            losses, target_weak_values, hard_values, gap_values = self._negative_bce_loss(final_logit, hard_regions)
        elif self.loss_mode == 'asym_rank':
            losses, target_weak_values, hard_values, gap_values = self._asymmetric_rank_loss(final_logit, hard_regions, target_regions)
        else:
            losses, target_weak_values, hard_values, gap_values = self._rank_loss(final_logit, hard_regions, target_regions)

        region_loss = torch.stack(losses).mean() if losses else zero
        stats = {}
        stats.update(hard_stats)
        stats.update(target_stats)
        stats['target_weak_logit'] = (
            torch.stack(target_weak_values).mean() if target_weak_values else torch.tensor(0.0, device=final_logit.device)
        )
        stats['hard_region_logit'] = (
            torch.stack(hard_values).mean() if hard_values else torch.tensor(0.0, device=final_logit.device)
        )
        stats['region_logit_gap'] = (
            torch.stack(gap_values).mean() if gap_values else torch.tensor(0.0, device=final_logit.device)
        )
        stats['valid_region_loss_images'] = torch.tensor(
            float(sum(1 for regions in hard_regions if regions)), device=final_logit.device
        )
        return region_loss, stats


class MSHNetVariantLoss(nn.Module):
    def __init__(
        self,
        variant='baseline',
        mshnet_warm_epoch=5,
        lambda_variant=0.2,
        focal_alpha=0.25,
        focal_gamma=2.0,
        ohem_ratio=0.01,
        topk_ratio=0.01,
        topk_dilate_kernel=7,
        tsr_lambda_region=0.0,
        tsr_region_start_epoch=60,
        tsr_region_end_epoch=100,
        tsr_target_scales='3,5,7',
        tsr_region_loss_mode='rank',
        tsr_beta=0.5,
        tsr_topk=3,
        tsr_nms_iou=0.3,
        tsr_weight_temp=0.2,
        tsr_target_temp=0.25,
        tsr_hard_temp=0.25,
        tsr_rank_temp=0.5,
        tsr_margin=0.5,
        tsr_topq=0.25,
        tsr_dilate_radius=0,
        tsr_use_consensus=True,
        tsr_bank_path=None,
        tsr_bank_max_regions=3,
        sps_lambda=0.0,
        sps_start_epoch=60,
        sps_end_epoch=120,
        sps_mode='sps',
        sps_objective='additive',
        sps_two_view_base=True,
        sps_dilate_radius=5,
        sps_disable_far_mask=False,
        sps_candidate_tau=0.3,
        sps_candidate_topk_ratio=0.0,
        sps_candidate_topk_metric='confidence',
        sps_candidate_min_metric=None,
        sps_candidate_min_confidence=0.0,
        sps_candidate_fallback_topk_ratio=0.0,
        sps_candidate_expand_radius=0,
        sps_candidate_expand_min_confidence=0.0,
        sps_target_margin_quantile=0.85,
        sps_target_margin_temp=0.01,
        sps_target_margin_min=0.0,
        sps_rerank_strict_fallback=True,
        sps_budget_q=0.1,
        sps_kmax=256,
        sps_eta=1.0,
        sps_adaptive_radius=True,
        sps_radius_kappa=1.0,
        sps_radius_r0=2.0,
        sps_radius_min=3,
        sps_radius_max=9,
        sps_target_safe=False,
        sps_target_safe_u_low=0.02,
        sps_target_safe_u_high=0.08,
        sps_target_safe_conf_min=0.55,
        sps_target_safe_conf_floor=0.35,
        sps_target_safe_alpha_floor=0.0,
    ):
        super().__init__()
        self.variant = variant
        self.mshnet_warm_epoch = mshnet_warm_epoch
        self.lambda_variant = lambda_variant
        self.sls = HCSLSIoULoss()
        self.down = nn.MaxPool2d(2, 2)
        self.focal = FocalLogitLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ohem = OHEMLogitLoss(topk_ratio=ohem_ratio)
        self.topk_neg = TopKNegativeLogitLoss(topk_ratio=topk_ratio, dilate_kernel=topk_dilate_kernel)
        self.sps_lambda = float(sps_lambda)
        self.sps_start_epoch = int(sps_start_epoch)
        self.sps_end_epoch = int(sps_end_epoch)
        self.sps_mode = str(sps_mode)
        self.sps_objective = str(sps_objective)
        self.sps_two_view_base = bool(sps_two_view_base)
        self.sps_loss = SelfPerturbationStabilityLoss(
            dilate_radius=sps_dilate_radius,
            disable_far_mask=sps_disable_far_mask,
            candidate_tau=sps_candidate_tau,
            candidate_topk_ratio=sps_candidate_topk_ratio,
            candidate_topk_metric=sps_candidate_topk_metric,
            candidate_min_metric=sps_candidate_min_metric,
            candidate_min_confidence=sps_candidate_min_confidence,
            candidate_fallback_topk_ratio=sps_candidate_fallback_topk_ratio,
            candidate_expand_radius=sps_candidate_expand_radius,
            candidate_expand_min_confidence=sps_candidate_expand_min_confidence,
            target_margin_quantile=sps_target_margin_quantile,
            target_margin_temp=sps_target_margin_temp,
            target_margin_min=sps_target_margin_min,
            rerank_strict_fallback=sps_rerank_strict_fallback,
            budget_q=sps_budget_q,
            kmax=sps_kmax,
            eta=sps_eta,
            mode=sps_mode,
            adaptive_radius=sps_adaptive_radius,
            radius_kappa=sps_radius_kappa,
            radius_r0=sps_radius_r0,
            radius_min=sps_radius_min,
            radius_max=sps_radius_max,
            target_safe=sps_target_safe,
            target_safe_u_low=sps_target_safe_u_low,
            target_safe_u_high=sps_target_safe_u_high,
            target_safe_conf_min=sps_target_safe_conf_min,
            target_safe_conf_floor=sps_target_safe_conf_floor,
            target_safe_alpha_floor=sps_target_safe_alpha_floor,
        )
        self.tsr_lambda_region = float(tsr_lambda_region)
        self.tsr_region_start_epoch = int(tsr_region_start_epoch)
        self.tsr_region_end_epoch = int(tsr_region_end_epoch)
        region_kwargs = dict(
            target_scales=tsr_target_scales,
            beta=tsr_beta,
            topk=tsr_topk,
            nms_iou=tsr_nms_iou,
            weight_temp=tsr_weight_temp,
            target_temp=tsr_target_temp,
            hard_temp=tsr_hard_temp,
            rank_temp=tsr_rank_temp,
            margin=tsr_margin,
            topq=tsr_topq,
            dilation_radius=tsr_dilate_radius,
            use_consensus=tsr_use_consensus,
            loss_mode=tsr_region_loss_mode,
        )
        self.tsr_bank_path = tsr_bank_path
        if tsr_bank_path:
            self.region_loss = PersistentClutterRegionLoss(
                bank_path=tsr_bank_path,
                max_regions=tsr_bank_max_regions,
                **region_kwargs,
            )
        else:
            self.region_loss = TargetScaleRegionLoss(**region_kwargs)

    def _lambda_region(self, epoch):
        if self.variant != 'ohem' or self.tsr_lambda_region <= 0:
            return 0.0
        if epoch <= self.tsr_region_start_epoch:
            return 0.0
        ramp = max(1, self.tsr_region_end_epoch - self.tsr_region_start_epoch)
        progress = min(1.0, float(epoch - self.tsr_region_start_epoch) / float(ramp))
        return self.tsr_lambda_region * progress

    def _lambda_sps(self, epoch):
        if self.variant != 'sps_ohem' or self.sps_lambda <= 0 or self.sps_mode in ('none', 'two_view_ohem'):
            return 0.0
        if epoch <= self.sps_start_epoch:
            return 0.0
        ramp = max(1, self.sps_end_epoch - self.sps_start_epoch)
        progress = min(1.0, float(epoch - self.sps_start_epoch) / float(ramp))
        return self.sps_lambda * progress

    def forward(
        self,
        masks,
        final_pred,
        gt_mask,
        epoch=0,
        image_ids=None,
        aug_ops=None,
        sps_pred=None,
        sps_gt_mask=None,
        sps_op='hflip',
    ):
        labels = gt_mask
        loss_sls = self.sls(final_pred, gt_mask, warm_epoch=self.mshnet_warm_epoch, epoch=epoch)
        loss_total = loss_sls
        aux_count = 0
        for idx, mask_pred in enumerate(masks):
            if idx > 0:
                labels = self.down(labels)
            loss_total = loss_total + self.sls(mask_pred, labels, warm_epoch=self.mshnet_warm_epoch, epoch=epoch)
            aux_count += 1
        loss_total = loss_total / (aux_count + 1)

        if self.variant == 'focal':
            loss_variant = self.focal(final_pred, gt_mask)
        elif self.variant in ('ohem', 'sps_ohem'):
            loss_variant = self.ohem(final_pred, gt_mask)
        elif self.variant == 'topk_neg':
            loss_variant = self.topk_neg(final_pred, gt_mask)
        else:
            loss_variant = final_pred.sum() * 0.0

        loss_total = loss_total + self.lambda_variant * loss_variant

        if self.variant == 'sps_ohem' and self.sps_two_view_base and sps_pred is not None:
            sps_masks, sps_final_pred = sps_pred
            sps_gt = gt_mask if sps_gt_mask is None else sps_gt_mask
            labels_p = sps_gt
            loss_sls_p = self.sls(sps_final_pred, sps_gt, warm_epoch=self.mshnet_warm_epoch, epoch=epoch)
            loss_total_p = loss_sls_p
            aux_count_p = 0
            for idx, mask_pred in enumerate(sps_masks):
                if idx > 0:
                    labels_p = self.down(labels_p)
                loss_total_p = loss_total_p + self.sls(mask_pred, labels_p, warm_epoch=self.mshnet_warm_epoch, epoch=epoch)
                aux_count_p += 1
            loss_total_p = loss_total_p / (aux_count_p + 1)
            loss_variant_p = self.ohem(sps_final_pred, sps_gt)
            loss_total_p = loss_total_p + self.lambda_variant * loss_variant_p
            loss_total = 0.5 * (loss_total + loss_total_p)
            loss_sls = 0.5 * (loss_sls + loss_sls_p)
            loss_variant = 0.5 * (loss_variant + loss_variant_p)
        lambda_region = self._lambda_region(epoch)
        zero = final_pred.sum() * 0.0
        region_stats = {
            'hard_regions': torch.tensor(0.0, device=final_pred.device),
            'empty_region_ratio': torch.tensor(1.0, device=final_pred.device),
            'hard_region_score_mean': torch.tensor(0.0, device=final_pred.device),
            'hard_region_uncertainty_mean': torch.tensor(0.0, device=final_pred.device),
            'hard_region_scale_mean': torch.tensor(0.0, device=final_pred.device),
            'target_regions': torch.tensor(0.0, device=final_pred.device),
            'empty_target_ratio': torch.tensor(1.0, device=final_pred.device),
            'target_region_scale_mean': torch.tensor(0.0, device=final_pred.device),
            'target_weak_logit': torch.tensor(0.0, device=final_pred.device),
            'hard_region_logit': torch.tensor(0.0, device=final_pred.device),
            'region_logit_gap': torch.tensor(0.0, device=final_pred.device),
            'valid_region_loss_images': torch.tensor(0.0, device=final_pred.device),
        }
        sps_stats = {
            'sps_hard_pixels': torch.tensor(0.0, device=final_pred.device),
            'sps_candidate_pixels': torch.tensor(0.0, device=final_pred.device),
            'sps_weight_sum': torch.tensor(0.0, device=final_pred.device),
            'sps_instability_mean': torch.tensor(0.0, device=final_pred.device),
            'sps_conf_mean': torch.tensor(0.0, device=final_pred.device),
            'sps_score_mean': torch.tensor(0.0, device=final_pred.device),
            'sps_neg_loss': torch.tensor(0.0, device=final_pred.device),
            'sps_target_alpha_scale': torch.tensor(1.0, device=final_pred.device),
            'sps_target_instability_mean': torch.tensor(0.0, device=final_pred.device),
            'sps_target_conf_mean': torch.tensor(1.0, device=final_pred.device),
        }
        if lambda_region > 0:
            if self.tsr_bank_path:
                loss_region, region_stats = self.region_loss(
                    masks, final_pred, gt_mask, image_ids=image_ids, aug_ops=aug_ops
                )
            else:
                loss_region, region_stats = self.region_loss(masks, final_pred, gt_mask)
            loss_total = loss_total + float(lambda_region) * loss_region
        else:
            loss_region = zero

        lambda_sps = self._lambda_sps(epoch)
        if lambda_sps > 0 and sps_pred is not None:
            _, sps_final_pred = sps_pred
            if self.sps_objective == 'rerank' and self.sps_mode != 'global_consistency':
                loss_sps, sps_stats = self.sps_loss.rerank_ohem_loss(
                    final_pred,
                    sps_final_pred,
                    gt_mask,
                    op=sps_op,
                    topk_ratio=self.ohem.topk_ratio,
                    alpha=float(lambda_sps),
                )
                loss_total = loss_total - self.lambda_variant * loss_variant + self.lambda_variant * loss_sps
                loss_variant = loss_sps
            else:
                loss_sps, sps_stats = self.sps_loss(final_pred, sps_final_pred, gt_mask, op=sps_op)
                loss_total = loss_total + float(lambda_sps) * loss_sps
        else:
            loss_sps = zero

        return {
            'total': loss_total,
            'sls': loss_sls.detach(),
            'variant_loss': loss_variant.detach(),
            'region_loss': loss_region.detach(),
            'lambda_region': torch.tensor(float(lambda_region), device=final_pred.device),
            'sps_loss': loss_sps.detach(),
            'lambda_sps': torch.tensor(float(lambda_sps), device=final_pred.device),
            'aux_count': torch.tensor(float(aux_count), device=final_pred.device),
            **{key: value.detach() for key, value in region_stats.items()},
            **{key: value.detach() for key, value in sps_stats.items()},
        }


class ECDVLoss(nn.Module):
    def __init__(
        self,
        lambda_risk=1.0,
        lambda_target_guard=1.0,
        lambda_keep=1.0,
        lambda_suppress=0.2,
        target_dilate=5,
    ):
        super().__init__()
        self.lambda_risk = float(lambda_risk)
        self.lambda_target_guard = float(lambda_target_guard)
        self.lambda_keep = float(lambda_keep)
        self.lambda_suppress = float(lambda_suppress)
        self.target_dilate = int(target_dilate)

    def _resize_mask(self, mask, size):
        if mask.shape[-2:] == size:
            return mask.float()
        return F.interpolate(mask.float(), size=size, mode='nearest')

    def _dilate_mask(self, mask, radius):
        if radius <= 0:
            return mask.float()
        kernel = 2 * radius + 1
        return (F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0).float()

    def forward(self, output, gt, epoch=0, pseudo_fp_mask=None, stage='risk_only'):
        del epoch
        evidence_logit = output['evidence_logit']
        final_logit = output['final_logit']
        risk_logit = output['risk_logit']
        gt = self._resize_mask(gt, risk_logit.shape[-2:])
        target_near = self._dilate_mask(gt, radius=self.target_dilate)
        target_mask = gt.float()
        if pseudo_fp_mask is None:
            raise ValueError("ECDVLoss requires a Gate-B-passed pseudo_fp_mask from the decoy bank.")
        pseudo_fp_mask = self._resize_mask(pseudo_fp_mask, risk_logit.shape[-2:]).float()

        risk_pos_map = F.binary_cross_entropy_with_logits(
            risk_logit,
            torch.ones_like(risk_logit),
            reduction='none',
        )
        loss_risk = (risk_pos_map * pseudo_fp_mask).sum() / (pseudo_fp_mask.sum() + 1e-6)

        target_guard_map = F.binary_cross_entropy_with_logits(
            risk_logit,
            torch.zeros_like(risk_logit),
            reduction='none',
        )
        loss_target_guard = (target_guard_map * target_near).sum() / (target_near.sum() + 1e-6)

        keep_weight = 1.0 + 5.0 * target_mask
        keep_map = F.smooth_l1_loss(final_logit, evidence_logit.detach(), reduction='none')
        loss_keep = (keep_map * keep_weight).sum() / (keep_weight.sum() + 1e-6)

        if stage == 'calibration':
            suppress_map = F.softplus(final_logit)
            loss_suppress = (suppress_map * pseudo_fp_mask).sum() / (pseudo_fp_mask.sum() + 1e-6)
        elif stage == 'risk_only':
            loss_suppress = final_logit.sum() * 0.0
        else:
            raise ValueError("stage must be 'risk_only' or 'calibration'")

        total = (
            self.lambda_risk * loss_risk
            + self.lambda_target_guard * loss_target_guard
            + self.lambda_keep * loss_keep
            + self.lambda_suppress * loss_suppress
        )

        risk_prob = torch.sigmoid(risk_logit.detach())
        pseudo_pixels = pseudo_fp_mask.sum()
        target_pixels = target_mask.sum()
        return {
            'total': total,
            'evidence_loss': final_logit.sum().detach() * 0.0,
            'risk_loss': loss_risk.detach(),
            'target_guard_loss': loss_target_guard.detach(),
            'keep_loss': loss_keep.detach(),
            'suppress_loss': loss_suppress.detach(),
            'pseudo_fp_pixels': pseudo_pixels.detach(),
            'pseudo_fp_images': (pseudo_fp_mask.flatten(1).sum(dim=1) > 0).float().sum().detach(),
            'risk_prob_pseudo_mean': ((risk_prob * pseudo_fp_mask).sum() / (pseudo_pixels + 1e-6)).detach(),
            'risk_prob_target_mean': ((risk_prob * target_mask).sum() / (target_pixels + 1e-6)).detach(),
            'target_guard_pixels': target_near.sum().detach(),
            'ecdv_beta': output['beta'].detach() if torch.is_tensor(output.get('beta')) else torch.tensor(0.0, device=final_logit.device),
            'suppression_mean': output['suppression_map'].detach().mean(),
        }


class MSCVLoss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        ohem_ratio=0.01,
        lambda_valid=1.0,
        lambda_target_guard=1.0,
        lambda_keep=1.0,
        lambda_suppress=0.2,
        far_radius=7,
        candidate_prob_thr=0.2,
        candidate_std_thr=0.05,
        nonflat_thr=0.05,
    ):
        super().__init__()
        self.base_loss = MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            ohem_ratio=ohem_ratio,
        )
        self.lambda_valid = float(lambda_valid)
        self.lambda_target_guard = float(lambda_target_guard)
        self.lambda_keep = float(lambda_keep)
        self.lambda_suppress = float(lambda_suppress)
        self.far_radius = int(far_radius)
        self.candidate_prob_thr = float(candidate_prob_thr)
        self.candidate_std_thr = float(candidate_std_thr)
        self.nonflat_thr = float(nonflat_thr)

    @staticmethod
    def _resize_mask(mask, size):
        mask = mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.shape[-2:] != size:
            mask = F.interpolate(mask, size=size, mode='nearest')
        return mask

    @staticmethod
    def _masked_bce(logit, label_value, mask):
        mask = mask.float()
        denom = mask.sum()
        if float(denom.detach().cpu()) < 1.0:
            return logit.sum() * 0.0
        target = torch.full_like(logit, float(label_value))
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        return (loss * mask).sum() / denom.clamp_min(1.0)

    def forward(self, output, gt, epoch=0, stage='validity_only'):
        masks = output['masks']
        evidence_logit = output['evidence_logit']
        final_logit = output['final_logit']
        validity_logit = output['validity_logit']
        p_max = output['p_max']
        p_std = output['p_std']
        local_contrast = output.get('local_contrast', None)

        target = self._resize_mask(gt, validity_logit.shape[-2:])
        base_out = self.base_loss(masks, evidence_logit, target, epoch=epoch)
        loss_evidence = base_out['total'] if isinstance(base_out, dict) else base_out

        cand = build_mscv_candidate_mask(
            p_max,
            p_std,
            target,
            far_radius=self.far_radius,
            candidate_prob_thr=self.candidate_prob_thr,
            candidate_std_thr=self.candidate_std_thr,
            local_contrast=local_contrast,
            nonflat_thr=self.nonflat_thr,
        )
        neg_candidate = cand['candidate']
        target_near = cand['target_near']

        valid_pos_loss = self._masked_bce(validity_logit, 1.0, target)
        valid_neg_loss = self._masked_bce(validity_logit, 0.0, neg_candidate)
        loss_valid = valid_pos_loss + valid_neg_loss
        target_guard_loss = valid_pos_loss

        keep_weight = 1.0 + 5.0 * target
        keep_map = F.smooth_l1_loss(final_logit, evidence_logit.detach(), reduction='none')
        keep_loss = (keep_map * keep_weight).sum() / (keep_weight.sum() + 1e-6)

        if stage == 'calibration':
            suppress_map = F.softplus(final_logit)
            suppress_loss = (suppress_map * neg_candidate).sum() / (neg_candidate.sum() + 1e-6)
        elif stage == 'validity_only':
            suppress_loss = final_logit.sum() * 0.0
        else:
            raise ValueError("stage must be 'validity_only' or 'calibration'")

        total = (
            loss_evidence
            + self.lambda_valid * loss_valid
            + self.lambda_target_guard * target_guard_loss
            + self.lambda_keep * keep_loss
            + self.lambda_suppress * suppress_loss
        )

        validity_prob = torch.sigmoid(validity_logit.detach())
        candidate_pixels = neg_candidate.sum()
        target_pixels = target.sum()
        return {
            'total': total,
            'evidence_loss': loss_evidence.detach(),
            'valid_loss': loss_valid.detach(),
            'valid_pos_loss': valid_pos_loss.detach(),
            'valid_neg_loss': valid_neg_loss.detach(),
            'target_guard_loss': target_guard_loss.detach(),
            'keep_loss': keep_loss.detach(),
            'suppress_loss': suppress_loss.detach(),
            'candidate_pixels': candidate_pixels.detach(),
            'candidate_images': (neg_candidate.flatten(1).sum(dim=1) > 0).float().sum().detach(),
            'target_leakage_pixels': (neg_candidate * target_near).sum().detach(),
            'validity_prob_target_mean': ((validity_prob * target).sum() / (target_pixels + 1e-6)).detach(),
            'validity_prob_candidate_mean': ((validity_prob * neg_candidate).sum() / (candidate_pixels + 1e-6)).detach(),
            'p_std_candidate_mean': ((p_std.detach() * neg_candidate).sum() / (candidate_pixels + 1e-6)).detach(),
            'mscv_beta': output['beta'].detach() if torch.is_tensor(output.get('beta')) else torch.tensor(0.0, device=final_logit.device),
            'suppression_mean': output['suppression_map'].detach().mean(),
        }


class BCVLoss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        ohem_ratio=0.01,
        lambda_bg=1.0,
        lambda_smooth=0.05,
        lambda_valid=1.0,
        lambda_keep=1.0,
        lambda_suppress=0.2,
        far_radius=7,
        candidate_prob_thr=0.3,
    ):
        super().__init__()
        self.base_loss = MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            ohem_ratio=ohem_ratio,
        )
        self.lambda_bg = float(lambda_bg)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_valid = float(lambda_valid)
        self.lambda_keep = float(lambda_keep)
        self.lambda_suppress = float(lambda_suppress)
        self.far_radius = int(far_radius)
        self.candidate_prob_thr = float(candidate_prob_thr)

    @staticmethod
    def _resize_mask(mask, size):
        mask = mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.shape[-2:] != size:
            mask = F.interpolate(mask, size=size, mode='nearest')
        return mask

    @staticmethod
    def _masked_bce(logit, label_value, mask):
        mask = mask.float()
        denom = mask.sum()
        if float(denom.detach().cpu()) < 1.0:
            return logit.sum() * 0.0
        target = torch.full_like(logit, float(label_value))
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        return (loss * mask).sum() / denom.clamp_min(1.0)

    @staticmethod
    def _gradient_loss(bg, mask):
        grad_x = torch.abs(bg[:, :, :, 1:] - bg[:, :, :, :-1])
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        grad_y = torch.abs(bg[:, :, 1:, :] - bg[:, :, :-1, :])
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss_x = (grad_x * mask_x).sum() / (mask_x.sum() + 1e-6)
        loss_y = (grad_y * mask_y).sum() / (mask_y.sum() + 1e-6)
        return loss_x + loss_y

    def forward(self, output, img, gt, epoch=0, stage='bg_only'):
        masks = output['masks']
        evidence_logit = output['evidence_logit']
        final_logit = output['final_logit']
        bg = output['background']
        residual = output['residual']
        validity_logit = output['validity_logit']

        target = self._resize_mask(gt, evidence_logit.shape[-2:])
        img = img[:, :1].float()
        if img.shape[-2:] != bg.shape[-2:]:
            img = F.interpolate(img, size=bg.shape[-2:], mode='bilinear', align_corners=False)
        target_near = dilate_mask(target, radius=self.far_radius).float()
        far_mask = 1.0 - target_near

        base_out = self.base_loss(masks, evidence_logit, target, epoch=epoch)
        loss_evidence = base_out['total'] if isinstance(base_out, dict) else base_out

        loss_bg_map = torch.abs(bg - img)
        loss_bg = (loss_bg_map * far_mask).sum() / (far_mask.sum() + 1e-6)
        loss_smooth = self._gradient_loss(bg, far_mask)

        residual_norm = residual / (residual.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        with torch.no_grad():
            p_e = torch.sigmoid(evidence_logit.detach())
            low_residual = residual_norm.detach() < 1.0
            neg_candidate = (
                (far_mask > 0.5)
                & (p_e > self.candidate_prob_thr)
                & low_residual
            ).float()

        valid_pos = self._masked_bce(validity_logit, 1.0, target)
        valid_neg = self._masked_bce(validity_logit, 0.0, neg_candidate)
        loss_valid = valid_pos + valid_neg

        keep_weight = 1.0 + 5.0 * target
        keep_map = F.smooth_l1_loss(final_logit, evidence_logit.detach(), reduction='none')
        loss_keep = (keep_map * keep_weight).sum() / (keep_weight.sum() + 1e-6)

        if stage == 'calibration':
            suppress_map = F.softplus(final_logit)
            loss_suppress = (suppress_map * neg_candidate).sum() / (neg_candidate.sum() + 1e-6)
        elif stage in ('bg_only', 'validity_only'):
            loss_suppress = final_logit.sum() * 0.0
        else:
            raise ValueError("stage must be 'bg_only', 'validity_only', or 'calibration'")

        total = (
            loss_evidence
            + self.lambda_bg * loss_bg
            + self.lambda_smooth * loss_smooth
            + self.lambda_valid * loss_valid
            + self.lambda_keep * loss_keep
            + self.lambda_suppress * loss_suppress
        )

        validity_prob = torch.sigmoid(validity_logit.detach())
        candidate_pixels = neg_candidate.sum()
        target_pixels = target.sum()
        return {
            'total': total,
            'evidence_loss': loss_evidence.detach(),
            'bg_loss': loss_bg.detach(),
            'smooth_loss': loss_smooth.detach(),
            'valid_loss': loss_valid.detach(),
            'valid_pos_loss': valid_pos.detach(),
            'valid_neg_loss': valid_neg.detach(),
            'keep_loss': loss_keep.detach(),
            'suppress_loss': loss_suppress.detach(),
            'candidate_pixels': candidate_pixels.detach(),
            'candidate_images': (neg_candidate.flatten(1).sum(dim=1) > 0).float().sum().detach(),
            'target_leakage_pixels': (neg_candidate * target_near).sum().detach(),
            'validity_prob_target_mean': ((validity_prob * target).sum() / (target_pixels + 1e-6)).detach(),
            'validity_prob_candidate_mean': ((validity_prob * neg_candidate).sum() / (candidate_pixels + 1e-6)).detach(),
            'residual_target_mean': ((residual.detach() * target).sum() / (target_pixels + 1e-6)).detach(),
            'residual_candidate_mean': ((residual.detach() * neg_candidate).sum() / (candidate_pixels + 1e-6)).detach(),
            'bcv_beta': output['beta'].detach() if torch.is_tensor(output.get('beta')) else torch.tensor(0.0, device=final_logit.device),
            'suppression_mean': output['suppression_map'].detach().mean(),
        }


class EACFMSHNetLoss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        lambda_ohem=0.2,
        ohem_ratio=0.01,
        lambda_anchor=0.5,
        lambda_scale_bg=0.05,
        lambda_scale_target=0.02,
        target_dilate_radius=3,
        anchor_easy_bg_thr=0.05,
    ):
        super().__init__()
        self.mshnet_warm_epoch = int(mshnet_warm_epoch)
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_scale_bg = float(lambda_scale_bg)
        self.lambda_scale_target = float(lambda_scale_target)
        self.target_dilate_radius = int(target_dilate_radius)
        self.anchor_easy_bg_thr = float(anchor_easy_bg_thr)
        self.base_loss = MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            lambda_variant=lambda_ohem,
            ohem_ratio=ohem_ratio,
        )

    @staticmethod
    def _dilate(mask, radius):
        if radius <= 0:
            return mask.float()
        kernel = 2 * int(radius) + 1
        return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(radius))

    def forward(self, out, gt_mask, epoch=0):
        final_logit = out['final_logit']
        base_logit = out['base_logit'].detach()
        masks = out.get('masks', [])
        base_out = self.base_loss(masks, final_logit, gt_mask, epoch=epoch)

        with torch.no_grad():
            gt = gt_mask.float()
            target_dilated = self._dilate(gt, self.target_dilate_radius)
            far_bg = (target_dilated < 0.5).float()
            base_prob = torch.sigmoid(base_logit)
            easy_bg = (base_prob < self.anchor_easy_bg_thr).float() * far_bg
            anchor_mask = torch.clamp(gt + easy_bg, 0, 1).bool()

        if anchor_mask.sum() > 0:
            loss_anchor = F.binary_cross_entropy_with_logits(
                final_logit[anchor_mask],
                base_prob[anchor_mask],
            )
        else:
            loss_anchor = final_logit.sum() * 0.0

        if 'scale_var' in out:
            scale_var = out['scale_var']
            final_prob = torch.sigmoid(final_logit)
            if gt.sum() > 0:
                loss_scale_target = (scale_var * gt).sum() / gt.sum().clamp_min(1.0)
            else:
                loss_scale_target = final_logit.sum() * 0.0
            loss_scale_bg = (scale_var.detach() * final_prob * far_bg).sum() / far_bg.sum().clamp_min(1.0)
        else:
            loss_scale_target = final_logit.sum() * 0.0
            loss_scale_bg = final_logit.sum() * 0.0

        total = (
            base_out['total']
            + self.lambda_anchor * loss_anchor
            + self.lambda_scale_target * loss_scale_target
            + self.lambda_scale_bg * loss_scale_bg
        )
        return {
            **base_out,
            'total': total,
            'main': base_out['total'].detach(),
            'anchor': loss_anchor.detach(),
            'scale_target': loss_scale_target.detach(),
            'scale_bg': loss_scale_bg.detach(),
            'eta': out.get('eta', total.detach() * 0.0).detach(),
        }


class SACFMSHNetLoss(nn.Module):
    def __init__(
        self,
        base_ohem_loss=None,
        mshnet_warm_epoch=5,
        lambda_ohem=0.2,
        ohem_ratio=0.01,
        lambda_anchor=0.05,
        lambda_scale=0.20,
        lambda_disagree_bg=0.10,
        far_dilate=7,
        isolated_high_thr=0.5,
        isolated_range_thr=0.25,
    ):
        super().__init__()
        self.base_ohem_loss = base_ohem_loss or MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            lambda_variant=lambda_ohem,
            ohem_ratio=ohem_ratio,
        )
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_scale = float(lambda_scale)
        self.lambda_disagree_bg = float(lambda_disagree_bg)
        self.far_dilate = int(far_dilate)
        self.isolated_high_thr = float(isolated_high_thr)
        self.isolated_range_thr = float(isolated_range_thr)

    @staticmethod
    def _dilate(mask, radius):
        if radius <= 0:
            return mask.float()
        kernel = 2 * int(radius) + 1
        return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(radius))

    def _far_background(self, target):
        return (self._dilate(target.float(), self.far_dilate) < 0.5).float()

    def _scale_loss(self, scale_logits, target):
        if not scale_logits:
            return target.sum() * 0.0
        losses = [F.binary_cross_entropy_with_logits(logit, target.float()) for logit in scale_logits]
        return torch.stack(losses).mean()

    def _disagreement_bg_loss(self, out, target):
        scale_logits = out.get('scale_logits', [])
        if not scale_logits:
            return out['final_logits'].sum() * 0.0
        probs = torch.sigmoid(torch.cat(scale_logits, dim=1))
        p_max = probs.max(dim=1, keepdim=True).values
        p_min = probs.min(dim=1, keepdim=True).values
        p_mean = probs.mean(dim=1, keepdim=True)
        p_range = p_max - p_min
        far_bg = self._far_background(target).bool()
        isolated = (
            far_bg
            & (p_max > self.isolated_high_thr)
            & (p_range > self.isolated_range_thr)
            & (p_mean < p_max)
        )
        if isolated.sum() == 0:
            return out['final_logits'].sum() * 0.0
        return torch.sigmoid(out['final_logits'])[isolated].mean()

    def forward(self, out, target, epoch=0):
        final_logits = out['final_logits']
        base_logits = out['base_logits'].detach()
        masks = out.get('masks', [])
        main_out = self.base_ohem_loss(masks, final_logits, target, epoch=epoch)
        loss_main = main_out['total']
        loss_anchor = F.mse_loss(torch.sigmoid(final_logits), torch.sigmoid(base_logits))
        loss_scale = self._scale_loss(out.get('scale_logits', []), target)
        loss_disagree = self._disagreement_bg_loss(out, target)
        total = (
            loss_main
            + self.lambda_anchor * loss_anchor
            + self.lambda_scale * loss_scale
            + self.lambda_disagree_bg * loss_disagree
        )
        return {
            **main_out,
            'total': total,
            'main': loss_main.detach(),
            'anchor': loss_anchor.detach(),
            'scale': loss_scale.detach(),
            'disagree_bg': loss_disagree.detach(),
            'fusion_gate_mean': out['fusion_gate'].detach().mean(),
            'fusion_delta_abs_mean': out['fusion_delta'].detach().abs().mean(),
        }


class CGAMSHNetLoss(nn.Module):
    def __init__(
        self,
        base_loss=None,
        mshnet_warm_epoch=5,
        lambda_ohem=0.2,
        ohem_ratio=0.01,
        lambda_center=0.2,
        lambda_scale=0.1,
        lambda_core=0.1,
        lambda_boundary=0.05,
        lambda_peak_bg=0.1,
        lambda_anchor_easy=0.05,
        peak_topk_ratio=0.001,
        peak_min_k=8,
        peak_max_k=256,
        peak_dilate_radius=3,
    ):
        super().__init__()
        self.base_loss = base_loss or MSHNetVariantLoss(
            variant='ohem',
            mshnet_warm_epoch=mshnet_warm_epoch,
            lambda_variant=lambda_ohem,
            ohem_ratio=ohem_ratio,
        )
        self.lambda_center = float(lambda_center)
        self.lambda_scale = float(lambda_scale)
        self.lambda_core = float(lambda_core)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_peak_bg = float(lambda_peak_bg)
        self.lambda_anchor_easy = float(lambda_anchor_easy)
        self.peak_topk_ratio = float(peak_topk_ratio)
        self.peak_min_k = int(peak_min_k)
        self.peak_max_k = int(peak_max_k)
        self.peak_dilate_radius = int(peak_dilate_radius)

    @staticmethod
    def _masked_cross_entropy(logits, target, valid):
        if valid.sum() == 0:
            return logits.sum() * 0.0
        loss = F.cross_entropy(logits, target.long(), reduction='none')
        return loss[valid[:, 0]].mean()

    def _easy_anchor_loss(self, final, target, anchor_prob):
        if anchor_prob is None:
            return final.sum() * 0.0
        final_prob = torch.sigmoid(final)
        easy_target = (target > 0.5) & (anchor_prob > 0.85)
        easy_bg = (target < 0.5) & (anchor_prob < 0.05)
        easy = easy_target | easy_bg
        if easy.sum() == 0:
            return final.sum() * 0.0
        return F.mse_loss(final_prob[easy], anchor_prob[easy])

    def forward(self, out, target, epoch=0, anchor_prob=None):
        final = out['final_logits']
        base_out = self.base_loss(out.get('masks', []), final, target, epoch=epoch)
        loss_mask = base_out['total']

        center_target, _scale_map, _scale_valid_float = build_center_heatmap(target.float())
        scale_target, scale_valid, _counts = component_area_bins(target.float())
        core_target, boundary_target, _ignore = build_core_boundary_maps(target.float())

        loss_center = F.binary_cross_entropy_with_logits(out['center_logits'], center_target)
        loss_scale = self._masked_cross_entropy(out['geometry_scale_logits'], scale_target, scale_valid)
        loss_core = F.binary_cross_entropy_with_logits(out['core_logits'], core_target)
        loss_boundary = F.binary_cross_entropy_with_logits(out['boundary_logits'], boundary_target)

        peak_mask = select_background_peaks(
            final,
            target.float(),
            topk_ratio=self.peak_topk_ratio,
            min_k=self.peak_min_k,
            max_k=self.peak_max_k,
            dilate_radius=self.peak_dilate_radius,
        )
        if peak_mask.any():
            loss_peak_bg = F.binary_cross_entropy_with_logits(final[peak_mask], torch.zeros_like(final[peak_mask]))
        else:
            loss_peak_bg = final.sum() * 0.0
        loss_anchor = self._easy_anchor_loss(final, target.float(), anchor_prob)

        total = (
            loss_mask
            + self.lambda_center * loss_center
            + self.lambda_scale * loss_scale
            + self.lambda_core * loss_core
            + self.lambda_boundary * loss_boundary
            + self.lambda_peak_bg * loss_peak_bg
            + self.lambda_anchor_easy * loss_anchor
        )
        return {
            **base_out,
            'total': total,
            'mask': loss_mask.detach(),
            'center': loss_center.detach(),
            'scale': loss_scale.detach(),
            'core': loss_core.detach(),
            'boundary': loss_boundary.detach(),
            'peak_bg': loss_peak_bg.detach(),
            'anchor_easy': loss_anchor.detach(),
            'peak_count': peak_mask.float().sum().detach(),
        }


class OHCMMSHNetLoss(nn.Module):
    def __init__(
        self,
        mshnet_warm_epoch=5,
        ohcm_warm_epoch=60,
        tau=0.5,
        dilate_radius=5,
        topk=3,
        hard_area_min=0.0,
        hard_area_max=0.0,
        mining_min_score=0.0,
        gt_area_median=20.0,
        margin_m=0.1,
        margin_delta=0.5,
        lambda_clu=0.2,
        lambda_sup=0.5,
        lambda_margin=0.1,
        lambda_proto=0.0,
        inhibition_start_epoch=None,
        proto_start_epoch=80,
        proto_momentum=0.9,
        proto_temperature=0.1,
        mining_mode='cc_area_lc_ms',
        use_clutter_head=True,
        use_inhibition=True,
        use_margin=True,
    ):
        super().__init__()
        self.mshnet_warm_epoch = int(mshnet_warm_epoch)
        self.ohcm_warm_epoch = int(ohcm_warm_epoch)
        self.tau = float(tau)
        self.dilate_radius = int(dilate_radius)
        self.topk = int(topk)
        self.hard_area_min = float(hard_area_min)
        self.hard_area_max = float(hard_area_max)
        self.mining_min_score = float(mining_min_score)
        self.gt_area_median = float(gt_area_median)
        self.margin_m = float(margin_m)
        self.margin_delta = float(margin_delta)
        self.lambda_clu = float(lambda_clu)
        self.lambda_sup = float(lambda_sup)
        self.lambda_margin = float(lambda_margin)
        self.lambda_proto = float(lambda_proto)
        self.inhibition_start_epoch = (
            self.ohcm_warm_epoch if inhibition_start_epoch is None else int(inhibition_start_epoch)
        )
        self.proto_start_epoch = int(proto_start_epoch)
        self.proto_momentum = float(proto_momentum)
        self.proto_temperature = float(proto_temperature)
        self.mining_mode = mining_mode
        self.use_clutter_head = bool(use_clutter_head)
        self.use_inhibition = bool(use_inhibition)
        self.use_margin = bool(use_margin)
        self.sls = HCSLSIoULoss()
        self.down = nn.MaxPool2d(2, 2)
        self.register_buffer('target_proto', torch.zeros(16))
        self.register_buffer('clutter_proto', torch.zeros(16))
        self.register_buffer('proto_ready', torch.tensor(0.0))

    def _dilate(self, mask, radius):
        if radius <= 0:
            return mask.float()
        kernel = radius * 2 + 1
        return (F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0).float()

    def _resize_mask(self, mask, size):
        if mask.shape[-2:] == size:
            return mask.float()
        return F.interpolate(mask.float(), size=size, mode='nearest')

    def _local_contrast_score(self, image_np, component_mask):
        from skimage import morphology
        radius = max(2, self.dilate_radius)
        ring = morphology.dilation(component_mask, morphology.disk(radius))
        ring = np.logical_and(ring, ~component_mask)
        inside = image_np[component_mask]
        outside = image_np[ring]
        if inside.size == 0 or outside.size == 0:
            return 0.0
        z = (float(inside.mean()) - float(outside.mean())) / (float(outside.std()) + 1e-6)
        return max(0.0, min(1.0, z / 3.0))

    def _multiscale_score(self, masks, batch_idx, coords, final_prob_np, out_hw):
        if 'ms' not in self.mining_mode or not masks:
            return 1.0
        scores = [float(final_prob_np[coords[:, 0], coords[:, 1]].mean())]
        for mask in masks:
            prob = torch.sigmoid(mask[batch_idx:batch_idx + 1])
            prob = F.interpolate(prob, size=out_hw, mode='bilinear', align_corners=True)
            prob_np = prob[0, 0].detach().cpu().numpy()
            scores.append(float(prob_np[coords[:, 0], coords[:, 1]].mean()))
        return max(0.0, min(1.0, min(scores) / (self.tau + 1e-6)))

    def _mine_hard_clutter(self, target_logit, masks, gt_mask, img=None):
        from skimage import measure
        gt_mask = self._resize_mask(gt_mask, target_logit.shape[-2:])
        safe_bg = 1.0 - self._dilate(gt_mask, self.dilate_radius)
        prob = torch.sigmoid(target_logit.detach())
        candidate = (prob > self.tau) & (safe_bg > 0)
        hard_mask = torch.zeros_like(target_logit)
        component_counts = []
        score_values = []
        area_values = []

        prob_cpu = prob[:, 0].detach().cpu().numpy()
        candidate_cpu = candidate[:, 0].detach().cpu().numpy().astype(bool)
        if img is not None:
            img_cpu = self._resize_mask(img[:, :1], target_logit.shape[-2:])[:, 0].detach().cpu().numpy()
        else:
            img_cpu = prob_cpu

        for b in range(target_logit.shape[0]):
            label = measure.label(candidate_cpu[b].astype(np.uint8), connectivity=2)
            regions = measure.regionprops(label)
            scored_regions = []
            for region in regions:
                area = int(region.area)
                if area <= 0:
                    continue
                if self.hard_area_min > 0 and area < self.hard_area_min:
                    continue
                if self.hard_area_max > 0 and area > self.hard_area_max:
                    continue
                coords = region.coords
                mean_prob = float(prob_cpu[b][coords[:, 0], coords[:, 1]].mean())
                if self.mining_mode == 'pixel':
                    score = mean_prob
                else:
                    area_prior = 1.0
                    if 'area' in self.mining_mode:
                        area_prior = math.exp(-abs(math.log((area + 1e-6) / (self.gt_area_median + 1e-6))))
                    local_contrast = 1.0
                    if 'lc' in self.mining_mode:
                        local_contrast = self._local_contrast_score(img_cpu[b], label == region.label)
                    ms_score = self._multiscale_score(masks, b, coords, prob_cpu[b], target_logit.shape[-2:])
                    score = mean_prob * area_prior * local_contrast * ms_score
                if self.mining_min_score > 0 and score < self.mining_min_score:
                    continue
                scored_regions.append((score, region))

            scored_regions.sort(key=lambda item: item[0], reverse=True)
            for score, region in scored_regions[:max(0, self.topk)]:
                coords = region.coords
                hard_mask[b, 0, coords[:, 0], coords[:, 1]] = 1.0
                score_values.append(float(score))
                area_values.append(float(region.area))
            component_counts.append(min(len(scored_regions), max(0, self.topk)))

        empty_count = sum(1 for count in component_counts if count == 0)
        stats = {
            'hard_pixels': hard_mask.sum().detach(),
            'hard_ratio': hard_mask.mean().detach(),
            'hard_components': torch.tensor(float(sum(component_counts)), device=target_logit.device),
            'empty_mining_ratio': torch.tensor(float(empty_count) / max(1, len(component_counts)), device=target_logit.device),
            'hard_area_mean': torch.tensor(float(np.mean(area_values)) if area_values else 0.0, device=target_logit.device),
            'hard_score_mean': torch.tensor(float(np.mean(score_values)) if score_values else 0.0, device=target_logit.device),
        }
        return hard_mask, stats

    def _masked_tensor_mean(self, tensor, mask):
        mask = mask.float()
        denom = mask.sum()
        if denom <= 0:
            return tensor.sum().detach() * 0.0
        return ((tensor.detach() * mask).sum() / (denom + 1e-6)).detach()

    def _masked_mean_features(self, feature, mask):
        mask = self._resize_mask(mask, feature.shape[-2:])
        vectors = []
        for b in range(feature.shape[0]):
            weight = mask[b:b + 1]
            denom = weight.sum()
            if denom <= 0:
                continue
            vec = (feature[b:b + 1] * weight).sum(dim=(2, 3)) / (denom + 1e-6)
            vectors.append(vec.squeeze(0))
        if not vectors:
            return None
        return torch.stack(vectors, dim=0)

    def _update_proto(self, name, vectors):
        if vectors is None or vectors.numel() == 0:
            return
        mean_vec = F.normalize(vectors.detach().mean(dim=0), dim=0)
        proto = getattr(self, name)
        if self.proto_ready.item() <= 0:
            proto.copy_(mean_vec)
        else:
            proto.mul_(self.proto_momentum).add_(mean_vec * (1.0 - self.proto_momentum))
            proto.copy_(F.normalize(proto, dim=0))

    def _proto_loss(self, feature, gt_mask, hard_mask):
        target_vecs = self._masked_mean_features(feature, gt_mask)
        clutter_vecs = self._masked_mean_features(feature, hard_mask)
        if target_vecs is None or clutter_vecs is None:
            return feature.sum() * 0.0

        if self.proto_ready.item() <= 0:
            with torch.no_grad():
                self._update_proto('target_proto', target_vecs)
                self._update_proto('clutter_proto', clutter_vecs)
                self.proto_ready.fill_(1.0)
            return feature.sum() * 0.0

        target_vecs = F.normalize(target_vecs, dim=1)
        clutter_vecs = F.normalize(clutter_vecs, dim=1)
        target_proto = F.normalize(self.target_proto, dim=0)
        clutter_proto = F.normalize(self.clutter_proto, dim=0)
        proto_bank = torch.stack([target_proto, clutter_proto], dim=0)
        logits_t = torch.matmul(target_vecs, proto_bank.t()) / self.proto_temperature
        logits_c = torch.matmul(clutter_vecs, proto_bank.t()) / self.proto_temperature
        labels_t = torch.zeros(logits_t.shape[0], dtype=torch.long, device=feature.device)
        labels_c = torch.ones(logits_c.shape[0], dtype=torch.long, device=feature.device)
        loss = 0.5 * (F.cross_entropy(logits_t, labels_t) + F.cross_entropy(logits_c, labels_c))

        with torch.no_grad():
            self._update_proto('target_proto', target_vecs)
            self._update_proto('clutter_proto', clutter_vecs)
        return loss

    def forward(self, output, gt_mask, img=None, epoch=0):
        masks = output['masks']
        target_logit = output['target_logit']
        clutter_logit = output['clutter_logit']
        final_logit = output['final_logit']
        feature = output['feature']

        labels = gt_mask
        loss_sls_main = self.sls(final_logit, gt_mask, warm_epoch=self.mshnet_warm_epoch, epoch=epoch)
        loss_sls = loss_sls_main
        aux_count = 0
        for idx, mask_pred in enumerate(masks):
            if idx > 0:
                labels = self.down(labels)
            loss_sls = loss_sls + self.sls(mask_pred, labels, warm_epoch=self.mshnet_warm_epoch, epoch=epoch)
            aux_count += 1
        loss_sls = loss_sls / (aux_count + 1)

        zero = final_logit.sum() * 0.0
        if epoch <= self.ohcm_warm_epoch:
            return {
                'total': loss_sls,
                'sls': loss_sls.detach(),
                'clu': zero.detach(),
                'sup': zero.detach(),
                'margin': zero.detach(),
                'proto': zero.detach(),
                'hard_pixels': torch.tensor(0.0, device=final_logit.device),
                'hard_ratio': torch.tensor(0.0, device=final_logit.device),
                'hard_components': torch.tensor(0.0, device=final_logit.device),
                'empty_mining_ratio': torch.tensor(1.0, device=final_logit.device),
                'hard_area_mean': torch.tensor(0.0, device=final_logit.device),
                'hard_score_mean': torch.tensor(0.0, device=final_logit.device),
                'target_prob_mean': torch.tensor(0.0, device=final_logit.device),
                'hard_prob_mean': torch.tensor(0.0, device=final_logit.device),
                'target_clutter_prob_mean': torch.tensor(0.0, device=final_logit.device),
                'hard_clutter_prob_mean': torch.tensor(0.0, device=final_logit.device),
                'target_prob_drop': torch.tensor(0.0, device=final_logit.device),
                'hard_prob_drop': torch.tensor(0.0, device=final_logit.device),
                'gamma': output['gamma'].detach(),
            }

        hard_mask, mining_stats = self._mine_hard_clutter(target_logit, masks, gt_mask, img=img)
        gt_resized = self._resize_mask(gt_mask, final_logit.shape[-2:])
        hard_pixels = hard_mask.sum()
        target_prob = torch.sigmoid(target_logit.detach())
        final_prob_detached = torch.sigmoid(final_logit.detach())
        clutter_prob = torch.sigmoid(clutter_logit.detach())
        target_prob_mean = self._masked_tensor_mean(target_prob, gt_resized)
        hard_prob_mean = self._masked_tensor_mean(target_prob, hard_mask)
        target_clutter_prob_mean = self._masked_tensor_mean(clutter_prob, gt_resized)
        hard_clutter_prob_mean = self._masked_tensor_mean(clutter_prob, hard_mask)
        target_prob_drop = self._masked_tensor_mean(target_prob - final_prob_detached, gt_resized)
        hard_prob_drop = self._masked_tensor_mean(target_prob - final_prob_detached, hard_mask)

        if self.use_clutter_head and self.lambda_clu > 0:
            clu_target = torch.zeros_like(clutter_logit)
            clu_target = torch.where(hard_mask > 0, torch.ones_like(clu_target), clu_target)
            valid = ((hard_mask > 0) | (gt_resized > 0)).float()
            clu_loss_map = F.binary_cross_entropy_with_logits(clutter_logit, clu_target, reduction='none')
            loss_clu = (clu_loss_map * valid).sum() / (valid.sum() + 1e-6)
        else:
            loss_clu = zero

        if (
            epoch > self.inhibition_start_epoch
            and self.use_inhibition
            and self.lambda_sup > 0
            and hard_pixels > 0
        ):
            final_prob = torch.sigmoid(final_logit)
            loss_sup = (F.relu(final_prob - self.margin_m) * hard_mask).sum() / (hard_pixels + 1e-6)
        else:
            loss_sup = zero

        if self.use_margin and self.lambda_margin > 0:
            target_region = gt_resized > 0
            margin_terms = []
            if target_region.sum() > 0:
                margin_terms.append(F.relu(self.margin_delta - (target_logit - clutter_logit))[target_region].mean())
            if hard_pixels > 0:
                margin_terms.append(F.relu(self.margin_delta - (clutter_logit - target_logit))[hard_mask > 0].mean())
            loss_margin = torch.stack(margin_terms).mean() if margin_terms else zero
        else:
            loss_margin = zero

        if self.lambda_proto > 0 and epoch >= self.proto_start_epoch:
            loss_proto = self._proto_loss(feature, gt_resized, hard_mask)
        else:
            loss_proto = zero

        loss_total = (
            loss_sls
            + self.lambda_clu * loss_clu
            + self.lambda_sup * loss_sup
            + self.lambda_margin * loss_margin
            + self.lambda_proto * loss_proto
        )

        return {
            'total': loss_total,
            'sls': loss_sls.detach(),
            'clu': loss_clu.detach(),
            'sup': loss_sup.detach(),
            'margin': loss_margin.detach(),
            'proto': loss_proto.detach(),
            'hard_pixels': mining_stats['hard_pixels'],
            'hard_ratio': mining_stats['hard_ratio'],
            'hard_components': mining_stats['hard_components'],
            'empty_mining_ratio': mining_stats['empty_mining_ratio'],
            'hard_area_mean': mining_stats['hard_area_mean'],
            'hard_score_mean': mining_stats['hard_score_mean'],
            'target_prob_mean': target_prob_mean,
            'hard_prob_mean': hard_prob_mean,
            'target_clutter_prob_mean': target_clutter_prob_mean,
            'hard_clutter_prob_mean': hard_clutter_prob_mean,
            'target_prob_drop': target_prob_drop,
            'hard_prob_drop': hard_prob_drop,
            'gamma': output['gamma'].detach(),
        }


class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge
