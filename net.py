from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from probability import foreground_probability

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

MODEL_ALIASES = {
    'UNet': 'U-Net',
    'U_Net': 'U-Net',
}

SUPPORTED_MODEL_NAMES = (
    'ACM',
    'ALCNet',
    'DNANet',
    'DNANet_BY',
    'ISNet',
    'RISTDnet',
    'UIUNet',
    'U-Net',
    'ISTDU-Net',
    'RDIAN',
    'ResUNet',
    'HCNet',
    'MSHNet',
    'MSHNetFocal',
    'MSHNetOHEM',
    'EACFMSHNet',
    'SACFMSHNet',
    'CGAMSHNet',
    'MSHNetTopKNeg',
    'MSHNetSPSOHEM',
    'ERDMSHNet',
    'ERDMSHNetV3',
    'PFRMSHNet',
    'ECDVMSHNet',
    'MSCVMSHNet',
    'BCVMSHNet',
    'OHCMMSHNet',
    'OHCMMSHNetFull',
)

MSHNET_VARIANT_NAMES = (
    'MSHNet',
    'MSHNetFocal',
    'MSHNetOHEM',
    'MSHNetTopKNeg',
    'MSHNetSPSOHEM',
)


class Net(nn.Module):
    def __init__(self, model_name, mode, loss_cfg=None):
        super(Net, self).__init__()
        if loss_cfg is None:
            loss_cfg = {}

        model_name = MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ResUNet':
            self.model = ResUNet()
        elif model_name == 'HCNet':
            self.model = HCNet()
            print(
                '[HCNetLoss Config] '
                f"sls_warm_epoch={loss_cfg.get('sls_warm_epoch', 10)}, "
                f"hc_warm_epoch={loss_cfg.get('hc_warm_epoch', 10)}, "
                f"lambda_hc={loss_cfg.get('lambda_hc', 0.0)}, "
                f"topk_ratio={loss_cfg.get('hc_topk_ratio', 0.01)}, "
                f"dilate_kernel={loss_cfg.get('hc_dilate_kernel', 7)}, "
                f"gamma={loss_cfg.get('hc_gamma', 2.0)}"
            )
            self.cal_loss = HCNetLoss(
                sls_warm_epoch=loss_cfg.get('sls_warm_epoch', 10),
                hc_warm_epoch=loss_cfg.get('hc_warm_epoch', 10),
                lambda_hc=loss_cfg.get('lambda_hc', 0.0),
                topk_ratio=loss_cfg.get('hc_topk_ratio', 0.01),
                dilate_kernel=loss_cfg.get('hc_dilate_kernel', 7),
                gamma=loss_cfg.get('hc_gamma', 2.0),
            )
        elif model_name == 'ERDMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.erd_gate_start_epoch = int(loss_cfg.get('erd_gate_start_epoch', 20))
            self.erd_gate_ramp_epochs = int(loss_cfg.get('erd_gate_ramp_epochs', 30))
            self.erd_gamma_max = float(loss_cfg.get('erd_gamma_max', 1.0))
            self.model = ERDMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                feature_channels=int(loss_cfg.get('erd_feature_channels', 16)),
                hidden_channels=int(loss_cfg.get('erd_hidden_channels', 32)),
                rho=float(loss_cfg.get('erd_rho', 0.25)),
            )
            self.cal_loss = ERDMSHNetLoss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                lambda_evidence=float(loss_cfg.get('erd_lambda_evidence', 0.2)),
                lambda_gate_pos=float(loss_cfg.get('erd_lambda_gate_pos', 0.05)),
                lambda_gate_neg=float(loss_cfg.get('erd_lambda_gate_neg', 0.20)),
                gate_start_epoch=self.erd_gate_start_epoch,
                gate_ramp_epochs=self.erd_gate_ramp_epochs,
                gate_target_radius=int(loss_cfg.get('erd_gate_target_radius', 2)),
                gate_far_radius=int(loss_cfg.get('erd_gate_far_radius', 5)),
                gate_neg_q=float(loss_cfg.get('erd_gate_neg_q', 0.01)),
                gate_neg_min_k=int(loss_cfg.get('erd_gate_neg_min_k', 16)),
                gate_neg_max_k=int(loss_cfg.get('erd_gate_neg_max_k', 512)),
            )
            print(
                '[ERDMSHNet Config] '
                f"rho={loss_cfg.get('erd_rho', 0.25)}, "
                f"gamma_max={self.erd_gamma_max}, "
                f"gate_start={self.erd_gate_start_epoch}, "
                f"gate_ramp={self.erd_gate_ramp_epochs}, "
                f"gate_neg_q={loss_cfg.get('erd_gate_neg_q', 0.01)}"
            )
        elif model_name == 'ERDMSHNetV3':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.model = ERDMSHNetV3(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                aux_in_channels=int(loss_cfg.get('erd_aux_in_channels', 16)),
                hidden_channels=int(loss_cfg.get('erd_hidden_channels', 32)),
                s_max=float(loss_cfg.get('erd_smax', 4.0)),
            )
            self.cal_loss = ERDMSHNetV3Loss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                lambda_evidence=float(loss_cfg.get('erd_lambda_evidence', 0.2)),
                far_radius=int(loss_cfg.get('erd_far_radius', 7)),
                target_protect_radius=int(loss_cfg.get('erd_target_protect_radius', 2)),
                neg_topk_ratio=float(loss_cfg.get('erd_neg_topk_ratio', 0.01)),
                lambda_protect_pos=float(loss_cfg.get('erd_lambda_protect_pos', 0.5)),
                lambda_protect_neg=float(loss_cfg.get('erd_lambda_protect_neg', 0.25)),
                lambda_clutter_pos=float(loss_cfg.get('erd_lambda_clutter_pos', 0.5)),
                lambda_clutter_neg=float(loss_cfg.get('erd_lambda_clutter_neg', 0.25)),
                lambda_preserve=float(loss_cfg.get('erd_lambda_preserve', 0.5)),
                preserve_margin=float(loss_cfg.get('erd_preserve_margin', 0.02)),
            )
            print(
                '[ERDMSHNetV3 Config] '
                f"aux_in_channels={loss_cfg.get('erd_aux_in_channels', 16)}, "
                f"hidden_channels={loss_cfg.get('erd_hidden_channels', 32)}, "
                f"s_max={loss_cfg.get('erd_smax', 4.0)}, "
                f"far_radius={loss_cfg.get('erd_far_radius', 7)}, "
                f"target_protect_radius={loss_cfg.get('erd_target_protect_radius', 2)}, "
                f"neg_topk_ratio={loss_cfg.get('erd_neg_topk_ratio', 0.01)}"
            )
        elif model_name == 'PFRMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.model = PFRMSHNet(
                evidence_net=MSHNet(int(loss_cfg.get('mshnet_in_channels', 1))),
                feature_channels=int(loss_cfg.get('pfr_feature_channels', 16)),
                beta=float(loss_cfg.get('pfr_beta', 0.5)),
            )
            self.cal_loss = PFRLoss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                lambda_far_neg=float(loss_cfg.get('pfr_lambda_far_neg', 0.5)),
                lambda_target_protect=float(loss_cfg.get('pfr_lambda_target_protect', 1.0)),
                lambda_boundary_protect=float(loss_cfg.get('pfr_lambda_boundary_protect', 0.5)),
                lambda_residual_sparse=float(loss_cfg.get('pfr_lambda_residual_sparse', 0.01)),
                far_topk_ratio=float(loss_cfg.get('pfr_far_topk_ratio', 0.005)),
                target_dilate=int(loss_cfg.get('pfr_target_dilate', 3)),
                far_dilate=int(loss_cfg.get('pfr_far_dilate', 9)),
            )
            print(
                '[PFRMSHNet Config] '
                f"beta={loss_cfg.get('pfr_beta', 0.5)}, "
                f"feature_channels={loss_cfg.get('pfr_feature_channels', 16)}, "
                f"far_topk_ratio={loss_cfg.get('pfr_far_topk_ratio', 0.005)}, "
                f"target_dilate={loss_cfg.get('pfr_target_dilate', 3)}, "
                f"far_dilate={loss_cfg.get('pfr_far_dilate', 9)}"
            )
        elif model_name == 'EACFMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.model = EACFMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                eta_max=float(loss_cfg.get('eacf_eta_max', 0.5)),
            )
            if bool(loss_cfg.get('eacf_freeze_backbone', True)):
                self.model.freeze_backbone()
            self.cal_loss = EACFMSHNetLoss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                lambda_ohem=float(loss_cfg.get('lambda_variant', 0.2)),
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                lambda_anchor=float(loss_cfg.get('eacf_lambda_anchor', 0.5)),
                lambda_scale_bg=float(loss_cfg.get('eacf_lambda_scale_bg', 0.05)),
                lambda_scale_target=float(loss_cfg.get('eacf_lambda_scale_target', 0.02)),
                target_dilate_radius=int(loss_cfg.get('eacf_target_dilate_radius', 3)),
                anchor_easy_bg_thr=float(loss_cfg.get('eacf_anchor_easy_bg_thr', 0.05)),
            )
            print(
                '[EACFMSHNet Config] '
                f"eta_max={loss_cfg.get('eacf_eta_max', 0.5)}, "
                f"freeze_backbone={loss_cfg.get('eacf_freeze_backbone', True)}, "
                f"lambda_anchor={loss_cfg.get('eacf_lambda_anchor', 0.5)}, "
                f"lambda_scale_bg={loss_cfg.get('eacf_lambda_scale_bg', 0.05)}, "
                f"lambda_scale_target={loss_cfg.get('eacf_lambda_scale_target', 0.02)}"
            )
        elif model_name == 'SACFMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.model = SACFMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                hidden_channels=int(loss_cfg.get('sacf_hidden_channels', 16)),
                delta_max=float(loss_cfg.get('sacf_delta_max', 1.0)),
            )
            if bool(loss_cfg.get('freeze_evidence', loss_cfg.get('sacf_freeze_evidence', True))):
                self.model.freeze_evidence()
            base_ohem_loss = MSHNetVariantLoss(
                variant='ohem',
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                lambda_variant=float(loss_cfg.get('lambda_variant', 0.2)),
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
            )
            self.cal_loss = SACFMSHNetLoss(
                base_ohem_loss=base_ohem_loss,
                lambda_anchor=float(loss_cfg.get('sacf_lambda_anchor', 0.05)),
                lambda_scale=float(loss_cfg.get('sacf_lambda_scale', 0.20)),
                lambda_disagree_bg=float(loss_cfg.get('sacf_lambda_disagree_bg', 0.10)),
                far_dilate=int(loss_cfg.get('sacf_far_dilate', 7)),
            )
            print(
                '[SACFMSHNet Config] '
                f"hidden={loss_cfg.get('sacf_hidden_channels', 16)}, "
                f"delta_max={loss_cfg.get('sacf_delta_max', 1.0)}, "
                f"freeze_evidence={loss_cfg.get('freeze_evidence', loss_cfg.get('sacf_freeze_evidence', True))}, "
                f"lambda_anchor={loss_cfg.get('sacf_lambda_anchor', 0.05)}, "
                f"lambda_scale={loss_cfg.get('sacf_lambda_scale', 0.20)}, "
                f"lambda_disagree_bg={loss_cfg.get('sacf_lambda_disagree_bg', 0.10)}"
            )
        elif model_name == 'CGAMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.model = CGAMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                num_scale_bins=int(loss_cfg.get('cga_num_scale_bins', 4)),
            )
            base_ohem_loss = MSHNetVariantLoss(
                variant='ohem',
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                lambda_variant=float(loss_cfg.get('lambda_variant', 0.2)),
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
            )
            self.cal_loss = CGAMSHNetLoss(
                base_loss=base_ohem_loss,
                lambda_center=float(loss_cfg.get('cga_lambda_center', 0.2)),
                lambda_scale=float(loss_cfg.get('cga_lambda_scale', 0.1)),
                lambda_core=float(loss_cfg.get('cga_lambda_core', 0.1)),
                lambda_boundary=float(loss_cfg.get('cga_lambda_boundary', 0.05)),
                lambda_peak_bg=float(loss_cfg.get('cga_lambda_peak_bg', 0.1)),
                lambda_anchor_easy=float(loss_cfg.get('cga_lambda_anchor_easy', 0.05)),
            )
            print(
                '[CGAMSHNet Config] '
                f"scale_bins={loss_cfg.get('cga_num_scale_bins', 4)}, "
                f"lambda_center={loss_cfg.get('cga_lambda_center', 0.2)}, "
                f"lambda_scale={loss_cfg.get('cga_lambda_scale', 0.1)}, "
                f"lambda_peak_bg={loss_cfg.get('cga_lambda_peak_bg', 0.1)}"
            )
        elif model_name == 'ECDVMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.ecdv_beta_max = float(loss_cfg.get('ecdv_beta_max', 0.1))
            self.ecdv_beta_start_epoch = int(loss_cfg.get('ecdv_beta_start_epoch', 999999))
            self.ecdv_beta_ramp_epochs = int(loss_cfg.get('ecdv_beta_ramp_epochs', 50))
            self.ecdv_eval_beta = loss_cfg.get('ecdv_eval_beta', None)
            self.model = ECDVMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                hidden_channels=int(loss_cfg.get('ecdv_hidden_channels', 32)),
                beta_max=self.ecdv_beta_max,
                evidence_threshold=float(loss_cfg.get('ecdv_evidence_threshold', 0.0)),
                detach_verifier_input=bool(loss_cfg.get('ecdv_detach_verifier_input', True)),
                contrast_kernel=int(loss_cfg.get('ecdv_contrast_kernel', 9)),
                highpass_kernel=int(loss_cfg.get('ecdv_highpass_kernel', 9)),
            )
            self.cal_loss = ECDVLoss(
                lambda_risk=float(loss_cfg.get('ecdv_lambda_risk', 1.0)),
                lambda_target_guard=float(loss_cfg.get('ecdv_lambda_target_guard', 1.0)),
                lambda_keep=float(loss_cfg.get('ecdv_lambda_keep', 1.0)),
                lambda_suppress=float(loss_cfg.get('ecdv_lambda_suppress', 0.2)),
                target_dilate=int(loss_cfg.get('ecdv_target_dilate', 5)),
            )
            print(
                '[ECDVMSHNet Config] '
                f"input_channels={loss_cfg.get('mshnet_in_channels', 1)}, "
                f"warm_epoch={self.mshnet_warm_epoch}, "
                f"beta_max={self.ecdv_beta_max}, "
                f"beta_start={self.ecdv_beta_start_epoch}, "
                f"beta_ramp={self.ecdv_beta_ramp_epochs}, "
                f"hidden={loss_cfg.get('ecdv_hidden_channels', 32)}, "
                f"detach_verifier_input={loss_cfg.get('ecdv_detach_verifier_input', True)}"
            )
        elif model_name == 'MSCVMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.mscv_beta_max = float(loss_cfg.get('mscv_beta_max', 0.1))
            self.mscv_beta_start_epoch = int(loss_cfg.get('mscv_beta_start_epoch', 999999))
            self.mscv_beta_ramp_epochs = int(loss_cfg.get('mscv_beta_ramp_epochs', 50))
            self.mscv_eval_beta = loss_cfg.get('mscv_eval_beta', None)
            self.model = MSCVMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                hidden_channels=int(loss_cfg.get('mscv_hidden_channels', 32)),
                beta_max=self.mscv_beta_max,
                evidence_threshold=float(loss_cfg.get('mscv_evidence_threshold', 0.0)),
                detach_verifier_input=bool(loss_cfg.get('mscv_detach_verifier_input', True)),
                contrast_kernel=int(loss_cfg.get('mscv_contrast_kernel', 9)),
            )
            self.cal_loss = MSCVLoss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                lambda_valid=float(loss_cfg.get('mscv_lambda_valid', 1.0)),
                lambda_target_guard=float(loss_cfg.get('mscv_lambda_target_guard', 1.0)),
                lambda_keep=float(loss_cfg.get('mscv_lambda_keep', 1.0)),
                lambda_suppress=float(loss_cfg.get('mscv_lambda_suppress', 0.2)),
                far_radius=int(loss_cfg.get('mscv_far_radius', 7)),
                candidate_prob_thr=float(loss_cfg.get('mscv_candidate_prob_thr', 0.2)),
                candidate_std_thr=float(loss_cfg.get('mscv_candidate_std_thr', 0.05)),
                nonflat_thr=float(loss_cfg.get('mscv_nonflat_thr', 0.05)),
            )
            print(
                '[MSCVMSHNet Config] '
                f"input_channels={loss_cfg.get('mshnet_in_channels', 1)}, "
                f"warm_epoch={self.mshnet_warm_epoch}, "
                f"beta_max={self.mscv_beta_max}, "
                f"beta_start={self.mscv_beta_start_epoch}, "
                f"beta_ramp={self.mscv_beta_ramp_epochs}, "
                f"hidden={loss_cfg.get('mscv_hidden_channels', 32)}, "
                f"prob_thr={loss_cfg.get('mscv_candidate_prob_thr', 0.2)}, "
                f"std_thr={loss_cfg.get('mscv_candidate_std_thr', 0.05)}"
            )
        elif model_name == 'BCVMSHNet':
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.bcv_beta_max = float(loss_cfg.get('bcv_beta_max', 0.1))
            self.bcv_beta_start_epoch = int(loss_cfg.get('bcv_beta_start_epoch', 999999))
            self.bcv_beta_ramp_epochs = int(loss_cfg.get('bcv_beta_ramp_epochs', 50))
            self.bcv_eval_beta = loss_cfg.get('bcv_eval_beta', None)
            self.model = BCVMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                hidden_channels=int(loss_cfg.get('bcv_hidden_channels', 32)),
                beta_max=self.bcv_beta_max,
                evidence_threshold=float(loss_cfg.get('bcv_evidence_threshold', 0.0)),
                detach_verifier_input=bool(loss_cfg.get('bcv_detach_verifier_input', True)),
                contrast_kernel=int(loss_cfg.get('bcv_contrast_kernel', 9)),
                validity_mode=str(loss_cfg.get('bcv_validity_mode', 'learned')),
                residual_theta=float(loss_cfg.get('bcv_residual_theta', 1.0)),
                residual_temp=float(loss_cfg.get('bcv_residual_temp', 0.2)),
                shape_theta=float(loss_cfg.get('bcv_shape_theta', 0.0)),
                shape_temp=float(loss_cfg.get('bcv_shape_temp', 0.2)),
            )
            self.cal_loss = BCVLoss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                lambda_bg=float(loss_cfg.get('bcv_lambda_bg', 1.0)),
                lambda_smooth=float(loss_cfg.get('bcv_lambda_smooth', 0.05)),
                lambda_valid=float(loss_cfg.get('bcv_lambda_valid', 1.0)),
                lambda_keep=float(loss_cfg.get('bcv_lambda_keep', 1.0)),
                lambda_suppress=float(loss_cfg.get('bcv_lambda_suppress', 0.2)),
                far_radius=int(loss_cfg.get('bcv_far_radius', 7)),
                candidate_prob_thr=float(loss_cfg.get('bcv_candidate_prob_thr', 0.3)),
            )
            print(
                '[BCVMSHNet Config] '
                f"input_channels={loss_cfg.get('mshnet_in_channels', 1)}, "
                f"warm_epoch={self.mshnet_warm_epoch}, "
                f"beta_max={self.bcv_beta_max}, "
                f"beta_start={self.bcv_beta_start_epoch}, "
                f"beta_ramp={self.bcv_beta_ramp_epochs}, "
                f"hidden={loss_cfg.get('bcv_hidden_channels', 32)}, "
                f"prob_thr={loss_cfg.get('bcv_candidate_prob_thr', 0.3)}, "
                f"far_radius={loss_cfg.get('bcv_far_radius', 7)}, "
                f"validity_mode={loss_cfg.get('bcv_validity_mode', 'learned')}, "
                f"residual_theta={loss_cfg.get('bcv_residual_theta', 1.0)}, "
                f"residual_temp={loss_cfg.get('bcv_residual_temp', 0.2)}, "
                f"shape_theta={loss_cfg.get('bcv_shape_theta', 0.0)}, "
                f"shape_temp={loss_cfg.get('bcv_shape_temp', 0.2)}"
            )
        elif model_name in MSHNET_VARIANT_NAMES:
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.model = MSHNet(int(loss_cfg.get('mshnet_in_channels', 1)))
            variant = {
                'MSHNet': 'baseline',
                'MSHNetFocal': 'focal',
                'MSHNetOHEM': 'ohem',
                'MSHNetTopKNeg': 'topk_neg',
                'MSHNetSPSOHEM': 'sps_ohem',
            }[model_name]
            self.cal_loss = MSHNetVariantLoss(
                variant=variant,
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                lambda_variant=float(loss_cfg.get('lambda_variant', 0.2)),
                focal_alpha=float(loss_cfg.get('focal_alpha', 0.25)),
                focal_gamma=float(loss_cfg.get('focal_gamma', 2.0)),
                ohem_ratio=float(loss_cfg.get('ohem_ratio', 0.01)),
                topk_ratio=float(loss_cfg.get('topk_ratio', 0.01)),
                topk_dilate_kernel=int(loss_cfg.get('topk_dilate_kernel', 7)),
                tsr_lambda_region=float(loss_cfg.get('tsr_lambda_region', 0.0)),
                tsr_region_start_epoch=int(loss_cfg.get('tsr_region_start_epoch', 60)),
                tsr_region_end_epoch=int(loss_cfg.get('tsr_region_end_epoch', 100)),
                tsr_target_scales=str(loss_cfg.get('tsr_target_scales', '3,5,7')),
                tsr_region_loss_mode=str(loss_cfg.get('tsr_region_loss_mode', 'rank')),
                tsr_beta=float(loss_cfg.get('tsr_beta', 0.5)),
                tsr_topk=int(loss_cfg.get('tsr_topk', 3)),
                tsr_nms_iou=float(loss_cfg.get('tsr_nms_iou', 0.3)),
                tsr_weight_temp=float(loss_cfg.get('tsr_weight_temp', 0.2)),
                tsr_target_temp=float(loss_cfg.get('tsr_target_temp', 0.25)),
                tsr_hard_temp=float(loss_cfg.get('tsr_hard_temp', 0.25)),
                tsr_rank_temp=float(loss_cfg.get('tsr_rank_temp', 0.5)),
                tsr_margin=float(loss_cfg.get('tsr_margin', 0.5)),
                tsr_topq=float(loss_cfg.get('tsr_topq', 0.25)),
                tsr_dilate_radius=int(loss_cfg.get('tsr_dilate_radius', 0)),
                tsr_use_consensus=bool(loss_cfg.get('tsr_use_consensus', True)),
                tsr_bank_path=loss_cfg.get('tsr_bank_path', None),
                tsr_bank_max_regions=int(loss_cfg.get('tsr_bank_max_regions', 3)),
                sps_lambda=float(loss_cfg.get('sps_lambda', 0.0)),
                sps_start_epoch=int(loss_cfg.get('sps_start_epoch', 60)),
                sps_end_epoch=int(loss_cfg.get('sps_end_epoch', 120)),
                sps_mode=str(loss_cfg.get('sps_mode', 'sps')),
                sps_objective=str(loss_cfg.get('sps_objective', 'additive')),
                sps_two_view_base=bool(loss_cfg.get('sps_two_view_base', True)),
                sps_dilate_radius=int(loss_cfg.get('sps_dilate_radius', 5)),
                sps_disable_far_mask=bool(loss_cfg.get('sps_disable_far_mask', False)),
                sps_candidate_tau=float(loss_cfg.get('sps_candidate_tau', 0.3)),
                sps_candidate_topk_ratio=float(loss_cfg.get('sps_candidate_topk_ratio', 0.0)),
                sps_candidate_topk_metric=str(loss_cfg.get('sps_candidate_topk_metric', 'confidence')),
                sps_candidate_min_metric=loss_cfg.get('sps_candidate_min_metric', None),
                sps_candidate_min_confidence=float(loss_cfg.get('sps_candidate_min_confidence', 0.0)),
                sps_candidate_fallback_topk_ratio=float(loss_cfg.get('sps_candidate_fallback_topk_ratio', 0.0)),
                sps_candidate_expand_radius=int(loss_cfg.get('sps_candidate_expand_radius', 0)),
                sps_candidate_expand_min_confidence=float(loss_cfg.get('sps_candidate_expand_min_confidence', 0.0)),
                sps_target_margin_quantile=float(loss_cfg.get('sps_target_margin_quantile', 0.85)),
                sps_target_margin_temp=float(loss_cfg.get('sps_target_margin_temp', 0.01)),
                sps_target_margin_min=float(loss_cfg.get('sps_target_margin_min', 0.0)),
                sps_rerank_strict_fallback=bool(loss_cfg.get('sps_rerank_strict_fallback', True)),
                sps_budget_q=float(loss_cfg.get('sps_budget_q', 0.1)),
                sps_kmax=int(loss_cfg.get('sps_kmax', 256)),
                sps_eta=float(loss_cfg.get('sps_eta', 1.0)),
                sps_adaptive_radius=bool(loss_cfg.get('sps_adaptive_radius', True)),
                sps_radius_kappa=float(loss_cfg.get('sps_radius_kappa', 1.0)),
                sps_radius_r0=float(loss_cfg.get('sps_radius_r0', 2.0)),
                sps_radius_min=int(loss_cfg.get('sps_radius_min', 3)),
                sps_radius_max=int(loss_cfg.get('sps_radius_max', 9)),
                sps_target_safe=bool(loss_cfg.get('sps_target_safe', False)),
                sps_target_safe_u_low=float(loss_cfg.get('sps_target_safe_u_low', 0.02)),
                sps_target_safe_u_high=float(loss_cfg.get('sps_target_safe_u_high', 0.08)),
                sps_target_safe_conf_min=float(loss_cfg.get('sps_target_safe_conf_min', 0.55)),
                sps_target_safe_conf_floor=float(loss_cfg.get('sps_target_safe_conf_floor', 0.35)),
                sps_target_safe_alpha_floor=float(loss_cfg.get('sps_target_safe_alpha_floor', 0.0)),
            )
            print(
                '[MSHNet Config] '
                f"variant={variant}, "
                f"input_channels={loss_cfg.get('mshnet_in_channels', 1)}, "
                f"warm_epoch={self.mshnet_warm_epoch}, "
                f"tsr_lambda_region={loss_cfg.get('tsr_lambda_region', 0.0)}, "
                f"tsr_scales={loss_cfg.get('tsr_target_scales', '3,5,7')}, "
                f"tsr_loss_mode={loss_cfg.get('tsr_region_loss_mode', 'rank')}, "
                f"tsr_bank={bool(loss_cfg.get('tsr_bank_path', None))}, "
                f"sps_lambda={loss_cfg.get('sps_lambda', 0.0)}, "
                f"sps_start={loss_cfg.get('sps_start_epoch', 60)}, "
                f"sps_end={loss_cfg.get('sps_end_epoch', 120)}, "
                f"sps_mode={loss_cfg.get('sps_mode', 'sps')}, "
                f"sps_objective={loss_cfg.get('sps_objective', 'additive')}, "
                f"sps_tau={loss_cfg.get('sps_candidate_tau', 0.3)}, "
                f"sps_candidate_topk_ratio={loss_cfg.get('sps_candidate_topk_ratio', 0.0)}, "
                f"sps_candidate_topk_metric={loss_cfg.get('sps_candidate_topk_metric', 'confidence')}, "
                f"sps_candidate_min_metric={loss_cfg.get('sps_candidate_min_metric', None)}, "
                f"sps_candidate_min_confidence={loss_cfg.get('sps_candidate_min_confidence', 0.0)}, "
                f"sps_candidate_fallback_topk_ratio={loss_cfg.get('sps_candidate_fallback_topk_ratio', 0.0)}, "
                f"sps_candidate_expand_radius={loss_cfg.get('sps_candidate_expand_radius', 0)}, "
                f"sps_candidate_expand_min_confidence={loss_cfg.get('sps_candidate_expand_min_confidence', 0.0)}, "
                f"sps_target_margin_quantile={loss_cfg.get('sps_target_margin_quantile', 0.85)}, "
                f"sps_target_margin_temp={loss_cfg.get('sps_target_margin_temp', 0.01)}, "
                f"sps_target_margin_min={loss_cfg.get('sps_target_margin_min', 0.0)}, "
                f"sps_rerank_strict_fallback={loss_cfg.get('sps_rerank_strict_fallback', True)}, "
                f"sps_q={loss_cfg.get('sps_budget_q', 0.1)}, "
                f"sps_kmax={loss_cfg.get('sps_kmax', 256)}, "
                f"sps_disable_far_mask={loss_cfg.get('sps_disable_far_mask', False)}, "
                f"sps_target_safe={loss_cfg.get('sps_target_safe', False)}, "
                f"sps_target_safe_floor={loss_cfg.get('sps_target_safe_alpha_floor', 0.0)}"
            )
        elif model_name in ('OHCMMSHNet', 'OHCMMSHNetFull'):
            self.mshnet_warm_epoch = int(loss_cfg.get('mshnet_warm_epoch', 5))
            self.ohcm_warm_epoch = int(loss_cfg.get('ohcm_warm_epoch', 60))
            self.ohcm_gamma_max = float(loss_cfg.get('ohcm_gamma_max', 0.3))
            self.ohcm_gamma_ramp_epochs = int(loss_cfg.get('ohcm_gamma_ramp_epochs', 60))
            inhibition_start = loss_cfg.get('ohcm_inhibition_start_epoch', None)
            self.ohcm_inhibition_start_epoch = (
                self.ohcm_warm_epoch if inhibition_start is None else int(inhibition_start)
            )
            force_no_proto = bool(loss_cfg.get('ohcm_force_no_proto', False))
            self.ohcm_use_proto = (model_name == 'OHCMMSHNetFull' or bool(loss_cfg.get('ohcm_use_proto', False))) and not force_no_proto
            lambda_proto = float(loss_cfg.get('lambda_proto', 0.05)) if self.ohcm_use_proto else 0.0
            self.model = OHCMMSHNet(
                input_channels=int(loss_cfg.get('mshnet_in_channels', 1)),
                gamma_max=self.ohcm_gamma_max,
                use_clutter_head=bool(loss_cfg.get('ohcm_use_clutter_head', True)),
                use_inhibition=bool(loss_cfg.get('ohcm_use_inhibition', True)),
            )
            self.cal_loss = OHCMMSHNetLoss(
                mshnet_warm_epoch=self.mshnet_warm_epoch,
                ohcm_warm_epoch=self.ohcm_warm_epoch,
                tau=float(loss_cfg.get('ohcm_tau', 0.5)),
                dilate_radius=int(loss_cfg.get('ohcm_dilate_radius', 5)),
                topk=int(loss_cfg.get('ohcm_topk', 3)),
                hard_area_min=float(loss_cfg.get('ohcm_hard_area_min', 0.0)),
                hard_area_max=float(loss_cfg.get('ohcm_hard_area_max', 0.0)),
                mining_min_score=float(loss_cfg.get('ohcm_mining_min_score', 0.0)),
                gt_area_median=float(loss_cfg.get('ohcm_gt_area_median', 20.0)),
                margin_m=float(loss_cfg.get('ohcm_margin_m', 0.1)),
                margin_delta=float(loss_cfg.get('ohcm_margin_delta', 0.5)),
                lambda_clu=float(loss_cfg.get('lambda_clu', 0.2)),
                lambda_sup=float(loss_cfg.get('lambda_sup', 0.5)),
                lambda_margin=float(loss_cfg.get('lambda_margin', 0.1)),
                lambda_proto=lambda_proto,
                inhibition_start_epoch=self.ohcm_inhibition_start_epoch,
                proto_start_epoch=int(loss_cfg.get('ohcm_proto_start_epoch', 80)),
                proto_momentum=float(loss_cfg.get('ohcm_proto_momentum', 0.9)),
                proto_temperature=float(loss_cfg.get('ohcm_proto_temperature', 0.1)),
                mining_mode=str(loss_cfg.get('ohcm_mining_mode', 'cc_area_lc_ms')),
                use_clutter_head=bool(loss_cfg.get('ohcm_use_clutter_head', True)),
                use_inhibition=bool(loss_cfg.get('ohcm_use_inhibition', True)),
                use_margin=bool(loss_cfg.get('ohcm_use_margin', True)),
            )
            print(
                '[OHCMMSHNet Config] '
                f"input_channels={loss_cfg.get('mshnet_in_channels', 1)}, "
                f"mshnet_warm_epoch={self.mshnet_warm_epoch}, "
                f"ohcm_warm_epoch={self.ohcm_warm_epoch}, "
                f"inhibition_start={self.ohcm_inhibition_start_epoch}, "
                f"gamma_max={self.ohcm_gamma_max}, "
                f"use_proto={self.ohcm_use_proto}"
            )
        else:
            raise ValueError(
                "Unknown model_name '{}'. Supported model names: {}".format(
                    model_name, ', '.join(SUPPORTED_MODEL_NAMES)
                )
            )
        
    def forward(self, img, epoch=0, return_dict: bool = False, output_head: str = "final"):
        if self.model_name == 'ERDMSHNet':
            gamma = self._erd_gamma(epoch) if self.training else self.erd_gamma_max
            output = self.model(
                img,
                warm_flag=True,
                gamma=gamma,
                return_feature=True,
            )
            if self.training:
                return output
            return foreground_probability(output['final_logit'])

        if self.model_name == 'ERDMSHNetV3':
            output = self.model(
                img,
                warm_flag=True,
                return_aux=True,
                return_feature=True,
            )
            if self.training:
                return output
            return foreground_probability(output['logits'])

        if self.model_name == 'PFRMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            output = self.model(img, warm_flag=warm_flag, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head == "evidence":
                return foreground_probability(output['evidence_logits'])
            if output_head == "residual":
                return output['residual_delta']
            if output_head != "final":
                raise ValueError("output_head must be one of: final, evidence, residual")
            return foreground_probability(output['logits'])

        if self.model_name == 'EACFMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            output = self.model(img, warm_flag=warm_flag, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head == 'base':
                return foreground_probability(output['base_logit'])
            if output_head == 'consensus':
                return foreground_probability(output['consensus_logit'])
            if output_head != 'final':
                raise ValueError("output_head must be one of: final, base, consensus")
            return foreground_probability(output['final_logit'])

        if self.model_name == 'SACFMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            output = self.model(img, warm_flag=warm_flag, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head == 'base':
                return foreground_probability(output['base_logits'])
            if output_head != 'final':
                raise ValueError("output_head must be one of: final, base")
            return foreground_probability(output['final_logits'])

        if self.model_name == 'CGAMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            output = self.model(img, warm_flag=warm_flag, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head != 'final':
                raise ValueError("output_head must be final for CGAMSHNet")
            return foreground_probability(output['final_logits'])

        if self.model_name == 'ECDVMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            beta = self._ecdv_beta(epoch) if self.training else self._ecdv_eval_beta()
            output = self.model(img, warm_flag=warm_flag, beta=beta, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head != 'final':
                raise ValueError("output_head must be final for ECDVMSHNet")
            return foreground_probability(output['final_logit'])

        if self.model_name == 'MSCVMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            beta = self._mscv_beta(epoch) if self.training else self._mscv_eval_beta()
            output = self.model(img, warm_flag=warm_flag, beta=beta, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head != 'final':
                raise ValueError("output_head must be final for MSCVMSHNet")
            return foreground_probability(output['final_logit'])

        if self.model_name == 'BCVMSHNet':
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            beta = self._bcv_beta(epoch) if self.training else self._bcv_eval_beta()
            output = self.model(img, warm_flag=warm_flag, beta=beta, return_dict=True)
            if return_dict:
                return output
            if self.training:
                return output
            if output_head != 'final':
                raise ValueError("output_head must be final for BCVMSHNet")
            return foreground_probability(output['final_logit'])

        if self.model_name in MSHNET_VARIANT_NAMES:
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            masks, pred = self.model(img, warm_flag)
            if self.training:
                return masks, pred
            return foreground_probability(pred)

        if self.model_name in ('OHCMMSHNet', 'OHCMMSHNetFull'):
            warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
            gamma = self._ohcm_gamma(epoch) if self.training else self.ohcm_gamma_max
            output = self.model(img, warm_flag=warm_flag, gamma=gamma, return_feature=True)
            if self.training:
                return output
            return foreground_probability(output['final_logit'])

        pred = self.model(img)

        if self.model_name == 'HCNet':
            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[-1]

            # 训练阶段返回 logits，loss 内部 sigmoid
            # 测试阶段返回 probability，方便 threshold=0.5
            if not self.training:
                pred = foreground_probability(pred)

        return pred

    def loss(
        self,
        pred,
        gt_mask,
        epoch=0,
        image_ids=None,
        aug_ops=None,
        sps_pred=None,
        sps_gt_mask=None,
        sps_op='hflip',
        ecdv_pseudo_fp_mask=None,
        ecdv_stage='risk_only',
        mscv_stage='validity_only',
        bcv_stage='bg_only',
    ):
        if self.model_name == 'HCNet':
            return self.cal_loss(pred, gt_mask, epoch)
        elif self.model_name == 'ERDMSHNet':
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        elif self.model_name == 'ERDMSHNetV3':
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        elif self.model_name == 'PFRMSHNet':
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        elif self.model_name == 'EACFMSHNet':
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        elif self.model_name == 'SACFMSHNet':
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        elif self.model_name == 'CGAMSHNet':
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        elif self.model_name == 'ECDVMSHNet':
            return self.cal_loss(
                pred,
                gt_mask,
                epoch=epoch,
                pseudo_fp_mask=ecdv_pseudo_fp_mask,
                stage=ecdv_stage,
            )
        elif self.model_name == 'MSCVMSHNet':
            return self.cal_loss(
                pred,
                gt_mask,
                epoch=epoch,
                stage=mscv_stage,
            )
        elif self.model_name == 'BCVMSHNet':
            raise ValueError("BCVMSHNet loss requires img; call loss_with_image(...).")
        elif self.model_name in MSHNET_VARIANT_NAMES:
            masks, final_pred = pred
            return self.cal_loss(
                masks,
                final_pred,
                gt_mask,
                epoch=epoch,
                image_ids=image_ids,
                aug_ops=aug_ops,
                sps_pred=sps_pred,
                sps_gt_mask=sps_gt_mask,
                sps_op=sps_op,
            )
        elif self.model_name in ('OHCMMSHNet', 'OHCMMSHNetFull'):
            return self.cal_loss(pred, gt_mask, epoch=epoch)
        else:
            return self.cal_loss(pred, gt_mask)

    def loss_with_image(self, pred, gt_mask, img, epoch=0, bcv_stage='bg_only'):
        if self.model_name in ('OHCMMSHNet', 'OHCMMSHNetFull'):
            return self.cal_loss(pred, gt_mask, img=img, epoch=epoch)
        if self.model_name == 'BCVMSHNet':
            return self.cal_loss(pred, img, gt_mask, epoch=epoch, stage=bcv_stage)
        return self.loss(pred, gt_mask, epoch=epoch)

    def _ohcm_gamma(self, epoch):
        if epoch <= self.ohcm_inhibition_start_epoch:
            return 0.0
        ramp = max(1, self.ohcm_gamma_ramp_epochs)
        progress = min(1.0, float(epoch - self.ohcm_inhibition_start_epoch) / float(ramp))
        return self.ohcm_gamma_max * progress

    def _erd_gamma(self, epoch):
        if epoch <= self.erd_gate_start_epoch:
            return 0.0
        ramp = max(1, self.erd_gate_ramp_epochs)
        progress = min(1.0, float(epoch - self.erd_gate_start_epoch) / float(ramp))
        return self.erd_gamma_max * progress

    def _ecdv_beta(self, epoch):
        if epoch <= self.ecdv_beta_start_epoch:
            return 0.0
        ramp = max(1, self.ecdv_beta_ramp_epochs)
        progress = min(1.0, float(epoch - self.ecdv_beta_start_epoch) / float(ramp))
        return self.ecdv_beta_max * progress

    def _ecdv_eval_beta(self):
        if self.ecdv_eval_beta is None:
            return self.ecdv_beta_max
        return min(max(float(self.ecdv_eval_beta), 0.0), self.ecdv_beta_max)

    def _mscv_beta(self, epoch):
        if epoch <= self.mscv_beta_start_epoch:
            return 0.0
        ramp = max(1, self.mscv_beta_ramp_epochs)
        progress = min(1.0, float(epoch - self.mscv_beta_start_epoch) / float(ramp))
        return self.mscv_beta_max * progress

    def _mscv_eval_beta(self):
        if self.mscv_eval_beta is None:
            return self.mscv_beta_max
        return min(max(float(self.mscv_eval_beta), 0.0), self.mscv_beta_max)

    def _bcv_beta(self, epoch):
        if epoch <= self.bcv_beta_start_epoch:
            return 0.0
        ramp = max(1, self.bcv_beta_ramp_epochs)
        progress = min(1.0, float(epoch - self.bcv_beta_start_epoch) / float(ramp))
        return self.bcv_beta_max * progress

    def _bcv_eval_beta(self):
        if self.bcv_eval_beta is None:
            return self.bcv_beta_max
        return min(max(float(self.bcv_eval_beta), 0.0), self.bcv_beta_max)

    def export_logits_features(self, img):
        if self.model_name == 'ERDMSHNet':
            output = self.model(img, warm_flag=True, gamma=self.erd_gamma_max, return_feature=True)
            return {
                'logit': output['final_logit'],
                'target_logit': output['evidence_logit'],
                'clutter_logit': torch.zeros_like(output['final_logit']),
                'reliability_logit': output['reliability_logit'],
                'feature': output['feature'],
                'masks': output['masks'],
                'gate': output['gate'],
            }
        if self.model_name == 'ERDMSHNetV3':
            output = self.model(img, warm_flag=True, return_aux=True, return_feature=True)
            return {
                'logit': output['logits'],
                'target_logit': output['evidence_logits'],
                'clutter_logit': output['clutter_logits'],
                'protection_logit': output['protection_logits'],
                'feature': output['feature'],
                'masks': output['masks'],
                'gate': output['gate'],
                'suppression': output['suppression'],
                'protection': output['protection'],
                'clutter': output['clutter'],
            }
        if self.model_name == 'PFRMSHNet':
            output = self.model(img, warm_flag=True, return_dict=True)
            return {
                'logit': output['logits'],
                'final_logit': output['logits'],
                'target_logit': output['evidence_logits'],
                'evidence_logit': output['evidence_logits'],
                'clutter_logit': torch.zeros_like(output['logits']),
                'delta_logit': output['delta_logits'],
                'residual_delta': output['residual_delta'],
                'raw_delta': output['raw_delta'],
                'feature': output['feature'],
                'masks': output['masks'],
            }
        if self.model_name == 'EACFMSHNet':
            output = self.model(img, warm_flag=True, return_dict=True)
            return {
                'logit': output['final_logit'],
                'final_logit': output['final_logit'],
                'target_logit': output['base_logit'],
                'base_logit': output['base_logit'],
                'consensus_logit': output['consensus_logit'],
                'clutter_logit': torch.zeros_like(output['final_logit']),
                'feature': output['decoder_feature'],
                'scale_logits': output['scale_logits'],
                'scale_probs': output['scale_probs'],
                'scale_var': output['scale_var'],
                'scale_range': output['scale_range'],
                'scale_weights': output['scale_weights'],
                'eta': output['eta'],
                'masks': output['masks'],
            }
        if self.model_name == 'SACFMSHNet':
            output = self.model(img, warm_flag=True, return_dict=True)
            return {
                'logit': output['final_logits'],
                'final_logit': output['final_logits'],
                'target_logit': output['base_logits'],
                'base_logit': output['base_logits'],
                'clutter_logit': torch.zeros_like(output['final_logits']),
                'consensus_logit': output['consensus_logits'],
                'feature': output['decoder_feature'],
                'scale_logits': output['scale_logits'],
                'fusion_weights': output['fusion_weights'],
                'fusion_gate': output['fusion_gate'],
                'fusion_delta': output['fusion_delta'],
                'scale_prob_var': output['scale_prob_var'],
                'scale_prob_range': output['scale_prob_range'],
                'masks': output['masks'],
            }
        if self.model_name == 'CGAMSHNet':
            output = self.model(img, warm_flag=True, return_dict=True)
            return {
                'logit': output['final_logits'],
                'final_logit': output['final_logits'],
                'target_logit': output['base_logits'],
                'base_logit': output['base_logits'],
                'clutter_logit': torch.zeros_like(output['final_logits']),
                'center_logit': output['center_logits'],
                'geometry_scale_logits': output['geometry_scale_logits'],
                'core_logit': output['core_logits'],
                'boundary_logit': output['boundary_logits'],
                'scale_logits_up': output['scale_logits_up'],
                'feature': output['decoder_feature'],
                'masks': output['masks'],
            }
        if self.model_name == 'ECDVMSHNet':
            output = self.model(img, warm_flag=True, beta=self._ecdv_eval_beta(), return_dict=True)
            return {
                'logit': output['final_logit'],
                'final_logit': output['final_logit'],
                'target_logit': output['evidence_logit'],
                'evidence_logit': output['evidence_logit'],
                'risk_logit': output['risk_logit'],
                'risk_prob': output['risk_prob'],
                'suppression_map': output['suppression_map'],
                'clutter_logit': output['risk_logit'],
                'feature': output['feature'],
                'masks': output['masks'],
            }
        if self.model_name == 'MSCVMSHNet':
            output = self.model(img, warm_flag=True, beta=self._mscv_eval_beta(), return_dict=True)
            return {
                'logit': output['final_logit'],
                'final_logit': output['final_logit'],
                'target_logit': output['evidence_logit'],
                'evidence_logit': output['evidence_logit'],
                'validity_logit': output['validity_logit'],
                'validity_prob': output['validity_prob'],
                'suppression_map': output['suppression_map'],
                'clutter_logit': -output['validity_logit'],
                'p_mean': output['p_mean'],
                'p_std': output['p_std'],
                'p_min': output['p_min'],
                'p_max': output['p_max'],
                'local_contrast': output['local_contrast'],
                'verifier_input': output['verifier_input'],
                'aux_logits': output['aux_logits'],
                'feature': output['feature'],
                'masks': output['masks'],
            }
        if self.model_name == 'BCVMSHNet':
            output = self.model(img, warm_flag=True, beta=self._bcv_eval_beta(), return_dict=True)
            return {
                'logit': output['final_logit'],
                'final_logit': output['final_logit'],
                'target_logit': output['evidence_logit'],
                'evidence_logit': output['evidence_logit'],
                'background': output['background'],
                'residual': output['residual'],
                'residual_norm': output['residual_norm'],
                'shape_score_map': output['shape_score_map'],
                'validity_logit': output['validity_logit'],
                'validity_prob': output['validity_prob'],
                'suppression_map': output['suppression_map'],
                'clutter_logit': -output['validity_logit'],
                'local_contrast': output['local_contrast'],
                'background_gradient': output['background_gradient'],
                'verifier_input': output['verifier_input'],
                'feature': output['feature'],
                'masks': output['masks'],
            }
        if self.model_name in MSHNET_VARIANT_NAMES:
            masks, logit, feature = self.model(img, True, return_feature=True)
            return {
                'logit': logit,
                'target_logit': logit,
                'clutter_logit': torch.zeros_like(logit),
                'feature': feature,
                'masks': masks,
            }
        if self.model_name in ('OHCMMSHNet', 'OHCMMSHNetFull'):
            output = self.model(img, warm_flag=True, gamma=self.ohcm_gamma_max, return_feature=True)
            return {
                'logit': output['final_logit'],
                'target_logit': output['target_logit'],
                'clutter_logit': output['clutter_logit'],
                'feature': output['feature'],
                'masks': output['masks'],
            }
        pred = self.model(img)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        return {
            'logit': pred,
            'target_logit': pred,
            'clutter_logit': torch.zeros_like(pred),
            'feature': pred,
            'masks': [],
        }
