import argparse
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
import random
import subprocess
import hashlib
from utils.branch_status import assert_branch_allowed
from utils.ecdv_decoy_bank import ECDVDecoyBank, apply_decoy_batch, check_ecdv_gate_b_summary
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
DEFAULT_SAVE_DIR = os.path.join(PROJECT_DIR, 'log')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['HCNet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet', 'HCNet', 'MSHNet', 'MSHNetFocal', 'MSHNetOHEM', 'EACFMSHNet', 'SACFMSHNet', 'CGAMSHNet', 'ECDVMSHNet', 'MSHNetTopKNeg', 'MSHNetSPSOHEM', 'ERDMSHNet', 'ERDMSHNetV3', 'PFRMSHNet', 'OHCMMSHNet', 'OHCMMSHNetFull'")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR, type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default=DEFAULT_SAVE_DIR, type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, nargs='+', help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=800, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--learning_rate", type=float, default=None,
                    help="Override the optimizer default lr. Useful for checkpoint fine-tuning.")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--scheduler_min_lr", type=float, default=None,
                    help="Override CosineAnnealingLR eta_min when applicable.")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--intervals", type=int, default=10, help="Intervals for print loss")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--use_parallel", action="store_true", help="Use DataParallel when multiple CUDA devices are available")
parser.add_argument("--allow_stopped_branch", action="store_true", default=False,
                    help="Allow training a stopped diagnostic branch only for reproduction.")
parser.add_argument("--eval_during_train", action="store_true", default=False,
                    help="Run the legacy test-set evaluation after checkpoints. Disabled by default to avoid test leakage.")
parser.add_argument("--lambda_hc", type=float, default=0.0)
parser.add_argument("--hc_topk_ratio", type=float, default=0.01)
parser.add_argument("--hc_dilate_kernel", type=int, default=7)
parser.add_argument("--hc_gamma", type=float, default=2.0)
parser.add_argument("--sls_warm_epoch", type=int, default=10)
parser.add_argument("--hc_warm_epoch", type=int, default=10)
parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
parser.add_argument("--mshnet_in_channels", type=int, default=1)
parser.add_argument("--lambda_variant", type=float, default=0.2)
parser.add_argument("--focal_alpha", type=float, default=0.25)
parser.add_argument("--focal_gamma", type=float, default=2.0)
parser.add_argument("--ohem_ratio", type=float, default=0.01)
parser.add_argument("--topk_ratio", type=float, default=0.01)
parser.add_argument("--topk_dilate_kernel", type=int, default=7)
parser.add_argument("--tsr_lambda_region", type=float, default=0.0)
parser.add_argument("--tsr_region_start_epoch", type=int, default=60)
parser.add_argument("--tsr_region_end_epoch", type=int, default=100)
parser.add_argument("--tsr_target_scales", type=str, default='3,5,7')
parser.add_argument("--tsr_region_loss_mode", type=str, default='rank', choices=['rank', 'neg_bce', 'asym_rank'])
parser.add_argument("--tsr_beta", type=float, default=0.5)
parser.add_argument("--tsr_topk", type=int, default=3)
parser.add_argument("--tsr_nms_iou", type=float, default=0.3)
parser.add_argument("--tsr_weight_temp", type=float, default=0.2)
parser.add_argument("--tsr_target_temp", type=float, default=0.25)
parser.add_argument("--tsr_hard_temp", type=float, default=0.25)
parser.add_argument("--tsr_rank_temp", type=float, default=0.5)
parser.add_argument("--tsr_margin", type=float, default=0.5)
parser.add_argument("--tsr_topq", type=float, default=0.25)
parser.add_argument("--tsr_dilate_radius", type=int, default=0,
                    help="0 means auto radius max(3, ceil(median target scale / 2)).")
parser.add_argument("--tsr_use_consensus", action="store_true", default=True)
parser.add_argument("--tsr_no_consensus", dest="tsr_use_consensus", action="store_false")
parser.add_argument("--tsr_bank_path", type=str, default=None)
parser.add_argument("--tsr_bank_max_regions", type=int, default=3)
parser.add_argument("--sps_lambda", type=float, default=0.0)
parser.add_argument("--sps_start_epoch", type=int, default=60)
parser.add_argument("--sps_end_epoch", type=int, default=120)
parser.add_argument("--sps_mode", type=str, default='sps',
                    choices=['sps', 'confidence_only', 'instability_only', 'target_margin', 'global_consistency', 'two_view_ohem', 'none'])
parser.add_argument("--sps_objective", type=str, default='additive', choices=['additive', 'rerank'],
                    help="additive keeps legacy loss_ohem + lambda_sps * loss_sps; rerank uses stability to reorder the OHEM negative budget.")
parser.add_argument("--sps_two_view_base", action="store_true", default=True)
parser.add_argument("--sps_no_two_view_base", dest="sps_two_view_base", action="store_false")
parser.add_argument("--sps_perturbation", type=str, default='hflip',
                    choices=['hflip', 'vflip', 'hvflip', 'transpose', 'gain_offset', 'gaussian_noise', 'mixed'])
parser.add_argument("--sps_gain_min", type=float, default=0.9)
parser.add_argument("--sps_gain_max", type=float, default=1.1)
parser.add_argument("--sps_offset_abs", type=float, default=0.03)
parser.add_argument("--sps_noise_std", type=float, default=0.02)
parser.add_argument("--sps_dilate_radius", type=int, default=5)
parser.add_argument("--sps_disable_far_mask", action="store_true", default=False,
                    help="Ablation: mine SPS candidates from all background pixels instead of the far-background safety mask.")
parser.add_argument("--sps_candidate_tau", type=float, default=0.3)
parser.add_argument("--sps_candidate_topk_ratio", type=float, default=0.0,
                    help="If >0, choose this top ratio of far-background pixels as SPS candidates.")
parser.add_argument("--sps_candidate_topk_metric", type=str, default='confidence',
                    choices=['confidence', 'instability', 'sps_score', 'target_margin_instability', 'target_margin_sps_score', 'target_contrast_instability', 'target_contrast_sps_score'],
                    help="Ranking metric for top-ratio SPS candidates.")
parser.add_argument("--sps_candidate_min_metric", type=float, default=None,
                    help="If set, top-ratio SPS candidates must have candidate metric greater than this value.")
parser.add_argument("--sps_candidate_min_confidence", type=float, default=0.0,
                    help="If >0, top-ratio SPS candidates must have at least this foreground confidence.")
parser.add_argument("--sps_candidate_fallback_topk_ratio", type=float, default=0.0,
                    help="If >0, fill empty threshold candidates with this top ratio of far-background pixels.")
parser.add_argument("--sps_candidate_expand_radius", type=int, default=0,
                    help="Expand SPS candidate seed pixels by this radius inside the far-background mask.")
parser.add_argument("--sps_candidate_expand_min_confidence", type=float, default=0.0,
                    help="If >0, expanded SPS candidates must also have at least this foreground confidence.")
parser.add_argument("--sps_target_margin_quantile", type=float, default=0.85,
                    help="Target instability quantile used by target_margin_* SPS candidate metrics.")
parser.add_argument("--sps_target_margin_temp", type=float, default=0.01,
                    help="Temperature for continuous target_contrast_* SPS candidate metrics.")
parser.add_argument("--sps_target_margin_min", type=float, default=0.0,
                    help="Extra absolute instability margin above the target quantile for target_margin_* SPS metrics.")
parser.add_argument("--sps_rerank_strict_fallback", dest="sps_rerank_strict_fallback", action="store_true", default=True,
                    help="For rerank mode, replace all-far fallback with a metric-ranked shortlist of at least the OHEM budget.")
parser.add_argument("--sps_no_rerank_strict_fallback", dest="sps_rerank_strict_fallback", action="store_false",
                    help="Legacy ablation: allow fallback to all valid negatives when SPS candidates are empty.")
parser.add_argument("--sps_budget_q", type=float, default=0.1)
parser.add_argument("--sps_kmax", type=int, default=256)
parser.add_argument("--sps_eta", type=float, default=1.0)
parser.add_argument("--sps_adaptive_radius", action="store_true", default=True)
parser.add_argument("--sps_fixed_radius", dest="sps_adaptive_radius", action="store_false")
parser.add_argument("--sps_radius_kappa", type=float, default=1.0)
parser.add_argument("--sps_radius_r0", type=float, default=2.0)
parser.add_argument("--sps_radius_min", type=int, default=3)
parser.add_argument("--sps_radius_max", type=int, default=9)
parser.add_argument("--sps_target_safe", action="store_true", default=False,
                    help="Reduce per-image SPS rerank strength when target pixels are unstable or weak under perturbation.")
parser.add_argument("--sps_target_safe_u_low", type=float, default=0.02)
parser.add_argument("--sps_target_safe_u_high", type=float, default=0.08)
parser.add_argument("--sps_target_safe_conf_min", type=float, default=0.55)
parser.add_argument("--sps_target_safe_conf_floor", type=float, default=0.35)
parser.add_argument("--sps_target_safe_alpha_floor", type=float, default=0.0)
parser.add_argument("--erd_rho", type=float, default=0.25)
parser.add_argument("--erd_gamma_max", type=float, default=1.0)
parser.add_argument("--erd_gate_start_epoch", type=int, default=20)
parser.add_argument("--erd_gate_ramp_epochs", type=int, default=30)
parser.add_argument("--erd_feature_channels", type=int, default=16)
parser.add_argument("--erd_hidden_channels", type=int, default=32)
parser.add_argument("--erd_lambda_evidence", type=float, default=0.2)
parser.add_argument("--erd_lambda_gate_pos", type=float, default=0.05)
parser.add_argument("--erd_lambda_gate_neg", type=float, default=0.20)
parser.add_argument("--erd_gate_target_radius", type=int, default=2)
parser.add_argument("--erd_gate_far_radius", type=int, default=5)
parser.add_argument("--erd_gate_neg_q", type=float, default=0.01)
parser.add_argument("--erd_gate_neg_min_k", type=int, default=16)
parser.add_argument("--erd_gate_neg_max_k", type=int, default=512)
parser.add_argument("--erd_require_gate_audit", action="store_true", default=True)
parser.add_argument("--erd_gate_audit_json", type=str, default=None)
parser.add_argument("--erd_no_gate_audit_guard", dest="erd_require_gate_audit", action="store_false")
parser.add_argument("--erd_pretrained_evidence", type=str, default=None,
                    help="Load MSHNet/MSHNetOHEM checkpoint into ERD evidence backbone.")
parser.add_argument("--erd_freeze_evidence_until", type=int, default=0,
                    help="Reserved. First ERD version keeps evidence trainable by default.")
parser.add_argument("--erd_version", type=str, default="v3_tpcs")
parser.add_argument("--erd_pretrained_ohem", type=str, default="",
                    help="Load MSHNetOHEM checkpoint into ERD-v3 evidence branch only.")
parser.add_argument("--erd_aux_in_channels", type=int, default=16)
parser.add_argument("--erd_smax", type=float, default=4.0)
parser.add_argument("--erd_far_radius", type=int, default=7)
parser.add_argument("--erd_target_protect_radius", type=int, default=2)
parser.add_argument("--erd_neg_topk_ratio", type=float, default=0.01)
parser.add_argument("--erd_lambda_protect_pos", type=float, default=0.5)
parser.add_argument("--erd_lambda_protect_neg", type=float, default=0.25)
parser.add_argument("--erd_lambda_clutter_pos", type=float, default=0.5)
parser.add_argument("--erd_lambda_clutter_neg", type=float, default=0.25)
parser.add_argument("--erd_lambda_preserve", type=float, default=0.5)
parser.add_argument("--erd_preserve_margin", type=float, default=0.02)
parser.add_argument("--erd_require_candidate_audit", action="store_true")
parser.add_argument("--erd_candidate_audit_json", type=str,
                    default=os.path.join(PROJECT_DIR, "docs/internal/erd_v3_candidate_audit_train/gate_pass.json"))
parser.add_argument("--pfr_beta", type=float, default=0.5)
parser.add_argument("--pfr_feature_channels", type=int, default=16)
parser.add_argument("--pfr_lambda_far_neg", type=float, default=0.5)
parser.add_argument("--pfr_lambda_target_protect", type=float, default=1.0)
parser.add_argument("--pfr_lambda_boundary_protect", type=float, default=0.5)
parser.add_argument("--pfr_lambda_residual_sparse", type=float, default=0.01)
parser.add_argument("--pfr_far_topk_ratio", type=float, default=0.005)
parser.add_argument("--pfr_target_dilate", type=int, default=3)
parser.add_argument("--pfr_far_dilate", type=int, default=9)
parser.add_argument("--pfr_ready_summary", type=str, default="")
parser.add_argument("--pfr_pretrained_ohem", type=str, default="",
                    help="Load MSHNetOHEM checkpoint into the PFR evidence branch.")
parser.add_argument("--pretrained_ohem_checkpoint", type=str, default="",
                    help="Load MSHNetOHEM checkpoint into the EACF backbone.")
parser.add_argument("--eacf_freeze_backbone", action="store_true",
                    help="Freeze the EACF MSHNetOHEM backbone for the official stage-1 fusion run.")
parser.add_argument("--eacf_eta_max", type=float, default=0.5)
parser.add_argument("--eacf_lambda_anchor", type=float, default=0.5)
parser.add_argument("--eacf_lambda_scale_bg", type=float, default=0.05)
parser.add_argument("--eacf_lambda_scale_target", type=float, default=0.02)
parser.add_argument("--eacf_target_dilate_radius", type=int, default=3)
parser.add_argument("--eacf_anchor_easy_bg_thr", type=float, default=0.05)
parser.add_argument("--ohem_checkpoint", type=str, default="",
                    help="Load MSHNetOHEM checkpoint into the SACF evidence branch.")
parser.add_argument("--freeze_evidence", type=str2bool, default=True)
parser.add_argument("--output_head", type=str, default="final", choices=["final", "base"])
parser.add_argument("--sacf_hidden_channels", type=int, default=16)
parser.add_argument("--sacf_delta_max", type=float, default=1.0)
parser.add_argument("--sacf_lambda_anchor", type=float, default=0.05)
parser.add_argument("--sacf_lambda_scale", type=float, default=0.20)
parser.add_argument("--sacf_lambda_disagree_bg", type=float, default=0.10)
parser.add_argument("--sacf_far_dilate", type=int, default=7)
parser.add_argument("--load_ohem_checkpoint", type=str, default="",
                    help="Load MSHNetOHEM checkpoint into the CGA evidence branch.")
parser.add_argument("--cga_train_mode", type=str, default="decoder_aux", choices=["decoder_aux", "aux_only"])
parser.add_argument("--cga_num_scale_bins", type=int, default=4)
parser.add_argument("--cga_lambda_center", type=float, default=0.2)
parser.add_argument("--cga_lambda_scale", type=float, default=0.1)
parser.add_argument("--cga_lambda_core", type=float, default=0.1)
parser.add_argument("--cga_lambda_boundary", type=float, default=0.05)
parser.add_argument("--cga_lambda_peak_bg", type=float, default=0.1)
parser.add_argument("--cga_lambda_anchor_easy", type=float, default=0.05)
parser.add_argument("--ohcm_warm_epoch", type=int, default=60)
parser.add_argument("--ohcm_tau", type=float, default=0.5)
parser.add_argument("--ohcm_dilate_radius", type=int, default=5)
parser.add_argument("--ohcm_topk", type=int, default=3)
parser.add_argument("--ohcm_hard_area_min", type=float, default=0.0)
parser.add_argument("--ohcm_hard_area_max", type=float, default=0.0)
parser.add_argument("--ohcm_mining_min_score", type=float, default=0.0)
parser.add_argument("--ohcm_gamma_max", type=float, default=0.3)
parser.add_argument("--ohcm_gamma_ramp_epochs", type=int, default=60)
parser.add_argument("--ohcm_inhibition_start_epoch", type=int, default=None)
parser.add_argument("--ohcm_margin_m", type=float, default=0.1)
parser.add_argument("--ohcm_margin_delta", type=float, default=0.5)
parser.add_argument("--ohcm_gt_area_median", type=float, default=20.0)
parser.add_argument("--ohcm_mining_mode", type=str, default='cc_area_lc_ms')
parser.add_argument("--ohcm_use_proto", action="store_true")
parser.add_argument("--ohcm_force_no_proto", action="store_true")
parser.add_argument("--ohcm_proto_start_epoch", type=int, default=80)
parser.add_argument("--ohcm_proto_momentum", type=float, default=0.9)
parser.add_argument("--ohcm_proto_temperature", type=float, default=0.1)
parser.add_argument("--ohcm_use_clutter_head", action="store_true", default=True)
parser.add_argument("--ohcm_no_clutter_head", dest="ohcm_use_clutter_head", action="store_false")
parser.add_argument("--ohcm_use_inhibition", action="store_true", default=True)
parser.add_argument("--ohcm_no_inhibition", dest="ohcm_use_inhibition", action="store_false")
parser.add_argument("--ohcm_use_margin", action="store_true", default=True)
parser.add_argument("--ohcm_no_margin", dest="ohcm_use_margin", action="store_false")
parser.add_argument("--ohcm_pretrained_backbone", default=None, type=str,
                    help="Load an MSHNet/OHCM checkpoint into OHCM backbone before training.")
parser.add_argument("--ohcm_freeze_backbone", action="store_true",
                    help="Freeze OHCM backbone and train only the added OHCM heads.")
parser.add_argument("--lambda_clu", type=float, default=0.2)
parser.add_argument("--lambda_sup", type=float, default=0.5)
parser.add_argument("--lambda_margin", type=float, default=0.1)
parser.add_argument("--lambda_proto", type=float, default=0.05)
parser.add_argument("--ecdv_pretrained_evidence", default=None, type=str,
                    help="Load an MSHNetOHEM checkpoint into the protected ECDV evidence branch.")
parser.add_argument("--ecdv_freeze_evidence", action="store_true",
                    help="Freeze the ECDV evidence branch and train only the verifier/calibration head.")
parser.add_argument("--ecdv_stage", choices=["risk_only", "calibration"], default="risk_only")
parser.add_argument("--ecdv_decoy_bank", default=None, type=str,
                    help="Gate-B-passed evidence-conditioned decoy bank directory.")
parser.add_argument("--ecdv_decoy_bank_summary", default=None, type=str,
                    help="Gate-B summary.json. Defaults to <ecdv_decoy_bank>/summary.json.")
parser.add_argument("--ecdv_beta_max", type=float, default=0.1)
parser.add_argument("--ecdv_beta_start_epoch", type=int, default=999999)
parser.add_argument("--ecdv_beta_ramp_epochs", type=int, default=50)
parser.add_argument("--ecdv_eval_beta", type=float, default=None)
parser.add_argument("--ecdv_hidden_channels", type=int, default=32)
parser.add_argument("--ecdv_evidence_threshold", type=float, default=0.0)
parser.add_argument("--ecdv_detach_verifier_input", action="store_true", default=True)
parser.add_argument("--ecdv_no_detach_verifier_input", dest="ecdv_detach_verifier_input", action="store_false")
parser.add_argument("--ecdv_contrast_kernel", type=int, default=9)
parser.add_argument("--ecdv_highpass_kernel", type=int, default=9)
parser.add_argument("--ecdv_lambda_risk", type=float, default=1.0)
parser.add_argument("--ecdv_lambda_target_guard", type=float, default=1.0)
parser.add_argument("--ecdv_lambda_keep", type=float, default=1.0)
parser.add_argument("--ecdv_lambda_suppress", type=float, default=0.2)
parser.add_argument("--ecdv_target_dilate", type=int, default=5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwrap_model(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def configure_optimizer_schedule():
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4 if opt.learning_rate is None else opt.learning_rate}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs': 400, 'step': [200, 300], 'gamma': 0.1}
        opt.scheduler_settings['epochs'] = opt.nEpochs
    elif opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05 if opt.learning_rate is None else opt.learning_rate}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': 1500, 'min_lr': 1e-5 if opt.scheduler_min_lr is None else opt.scheduler_min_lr}
        opt.scheduler_settings['epochs'] = opt.nEpochs
    else:
        opt.scheduler_settings.setdefault('epochs', opt.nEpochs)
    opt.nEpochs = opt.scheduler_settings['epochs']


def find_matching_checkpoint(paths, model_name, dataset_name, device):
    if not paths:
        return None, None
    for checkpoint_path in paths:
        if checkpoint_matches_model(checkpoint_path, model_name, dataset_name):
            return checkpoint_path, torch_load_checkpoint(checkpoint_path, device)
    return None, None


def torch_load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def load_model_checkpoint(net, checkpoint):
    unwrap_model(net).load_state_dict(checkpoint_state_dict(checkpoint))


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_split_hash():
    split_path = os.path.join(
        opt.dataset_dir,
        opt.dataset_name,
        'img_idx',
        'train_%s.txt' % opt.dataset_name,
    )
    if not os.path.exists(split_path):
        return {
            'dataset': opt.dataset_name,
            'train_split': os.path.abspath(split_path),
            'sha256': None,
            'status': 'missing',
        }
    return {
        'dataset': opt.dataset_name,
        'train_split': os.path.abspath(split_path),
        'sha256': file_sha256(split_path),
        'status': 'ok',
    }


def checkpoint_config():
    return {key: value for key, value in vars(opt).items() if key != 'f'}


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(checkpoint):
    rng_state = checkpoint.get('rng_state') if isinstance(checkpoint, dict) else None
    if not rng_state:
        return

    def cpu_byte_state(state):
        if torch.is_tensor(state):
            return state.detach().cpu().to(torch.uint8)
        return state

    if 'python' in rng_state:
        random.setstate(rng_state['python'])
    if 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])
    if 'torch' in rng_state:
        torch.set_rng_state(cpu_byte_state(rng_state['torch']))
    if torch.cuda.is_available() and rng_state.get('cuda'):
        cuda_states = [cpu_byte_state(state) for state in rng_state['cuda']]
        cuda_states = cuda_states[:torch.cuda.device_count()]
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def load_ohcm_pretrained_backbone(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name not in ('OHCMMSHNet', 'OHCMMSHNetFull'):
        raise ValueError("--ohcm_pretrained_backbone is only valid for OHCM models.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    mapped = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("model.backbone."):
            mapped[key] = value
        elif key.startswith("model."):
            mapped["model.backbone." + key[len("model."):]] = value

    current_keys = set(base_net.state_dict().keys())
    mapped = {key: value for key, value in mapped.items() if key in current_keys}
    missing, unexpected = base_net.load_state_dict(mapped, strict=False)
    missing_backbone = [key for key in missing if key.startswith("model.backbone.")]
    print(
        "[OHCM FrozenCalib] loaded backbone checkpoint=%s mapped_keys=%d missing_backbone=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_backbone), len(unexpected)),
        flush=True,
    )
    if missing_backbone:
        print("[OHCM FrozenCalib] missing backbone keys sample: %s" % missing_backbone[:8], flush=True)


def load_erd_pretrained_evidence(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'ERDMSHNet':
        raise ValueError("--erd_pretrained_evidence is only valid for ERDMSHNet.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.evidence.'):
            mapped[clean_key] = value
        elif clean_key.startswith('model.backbone.'):
            mapped['model.evidence.' + clean_key[len('model.backbone.'):]] = value
        elif clean_key.startswith('model.'):
            mapped['model.evidence.' + clean_key[len('model.'):]] = value
        else:
            mapped['model.evidence.' + clean_key] = value

    current = base_net.state_dict()
    mapped = {
        key: value
        for key, value in mapped.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing, unexpected = base_net.load_state_dict(mapped, strict=False)
    print(
        "[ERD pretrained evidence] checkpoint=%s mapped_keys=%d missing=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing), len(unexpected)),
        flush=True,
    )


def load_erd_v3_pretrained_ohem(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'ERDMSHNetV3':
        raise ValueError("--erd_pretrained_ohem is only valid for ERDMSHNetV3.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.evidence_net.'):
            mapped[clean_key] = value
        elif clean_key.startswith('model.evidence.'):
            mapped['model.evidence_net.' + clean_key[len('model.evidence.'):]] = value
        elif clean_key.startswith('model.backbone.'):
            mapped['model.evidence_net.' + clean_key[len('model.backbone.'):]] = value
        elif clean_key.startswith('model.'):
            mapped['model.evidence_net.' + clean_key[len('model.'):]] = value
        else:
            mapped['model.evidence_net.' + clean_key] = value

    current = base_net.state_dict()
    mapped = {
        key: value
        for key, value in mapped.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing, unexpected = base_net.load_state_dict(mapped, strict=False)
    missing_evidence = [key for key in missing if key.startswith('model.evidence_net.')]
    print(
        "[ERD-v3 pretrained OHEM] checkpoint=%s mapped_keys=%d missing_evidence=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_evidence), len(unexpected)),
        flush=True,
    )
    if missing_evidence:
        print("[ERD-v3 pretrained OHEM] missing evidence keys sample: %s" % missing_evidence[:8], flush=True)


def load_pfr_pretrained_ohem(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'PFRMSHNet':
        raise ValueError("--pfr_pretrained_ohem is only valid for PFRMSHNet.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.evidence_net.'):
            mapped[clean_key] = value
        elif clean_key.startswith('model.'):
            mapped['model.evidence_net.' + clean_key[len('model.'):]] = value
        else:
            mapped['model.evidence_net.' + clean_key] = value

    current = base_net.state_dict()
    mapped = {
        key: value
        for key, value in mapped.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing, unexpected = base_net.load_state_dict(mapped, strict=False)
    missing_evidence = [key for key in missing if key.startswith('model.evidence_net.')]
    print(
        "[PFR pretrained OHEM] checkpoint=%s mapped_keys=%d missing_evidence=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_evidence), len(unexpected)),
        flush=True,
    )
    if missing_evidence:
        print("[PFR pretrained OHEM] missing evidence keys sample: %s" % missing_evidence[:8], flush=True)


def load_eacf_ohem_checkpoint(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'EACFMSHNet':
        raise ValueError("--pretrained_ohem_checkpoint is only valid for EACFMSHNet.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    current = base_net.model.state_dict()
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.backbone.'):
            mapped_key = 'backbone.' + clean_key[len('model.backbone.'):]
        elif clean_key.startswith('model.'):
            mapped_key = 'backbone.' + clean_key[len('model.'):]
        else:
            mapped_key = 'backbone.' + clean_key
        if mapped_key in current and tuple(current[mapped_key].shape) == tuple(value.shape):
            mapped[mapped_key] = value

    missing, unexpected = base_net.model.load_state_dict(mapped, strict=False)
    missing_backbone = [key for key in missing if key.startswith('backbone.')]
    unexpected = [key for key in unexpected if not key.startswith('fusion.')]
    print(
        "[EACF pretrained OHEM] checkpoint=%s mapped_keys=%d missing_backbone=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_backbone), len(unexpected)),
        flush=True,
    )
    if missing_backbone:
        raise RuntimeError("[EACF pretrained OHEM] missing backbone keys sample: %s" % missing_backbone[:8])
    if unexpected:
        raise RuntimeError("[EACF pretrained OHEM] unexpected keys sample: %s" % unexpected[:8])


def load_sacf_ohem_checkpoint(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'SACFMSHNet':
        raise ValueError("--ohem_checkpoint is only valid for SACFMSHNet.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    current = base_net.model.state_dict()
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.evidence_net.'):
            mapped_key = 'evidence_net.' + clean_key[len('model.evidence_net.'):]
        elif clean_key.startswith('model.backbone.'):
            mapped_key = 'evidence_net.' + clean_key[len('model.backbone.'):]
        elif clean_key.startswith('model.'):
            mapped_key = 'evidence_net.' + clean_key[len('model.'):]
        else:
            mapped_key = 'evidence_net.' + clean_key
        if mapped_key in current and tuple(current[mapped_key].shape) == tuple(value.shape):
            mapped[mapped_key] = value

    missing, unexpected = base_net.model.load_state_dict(mapped, strict=False)
    missing_evidence = [key for key in missing if key.startswith('evidence_net.')]
    unexpected = [key for key in unexpected if not key.startswith('fusion.')]
    print(
        "[SACF pretrained OHEM] checkpoint=%s mapped_keys=%d missing_evidence=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_evidence), len(unexpected)),
        flush=True,
    )
    if missing_evidence:
        raise RuntimeError("[SACF pretrained OHEM] missing evidence keys sample: %s" % missing_evidence[:8])
    if unexpected:
        raise RuntimeError("[SACF pretrained OHEM] unexpected keys sample: %s" % unexpected[:8])


def load_cga_ohem_checkpoint(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'CGAMSHNet':
        raise ValueError("--load_ohem_checkpoint is only valid for CGAMSHNet.")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    current = base_net.model.state_dict()
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.evidence_net.'):
            mapped_key = 'evidence_net.' + clean_key[len('model.evidence_net.'):]
        elif clean_key.startswith('model.backbone.'):
            mapped_key = 'evidence_net.' + clean_key[len('model.backbone.'):]
        elif clean_key.startswith('model.'):
            mapped_key = 'evidence_net.' + clean_key[len('model.'):]
        else:
            mapped_key = 'evidence_net.' + clean_key
        if mapped_key in current and tuple(current[mapped_key].shape) == tuple(value.shape):
            mapped[mapped_key] = value

    missing, unexpected = base_net.model.load_state_dict(mapped, strict=False)
    missing_evidence = [key for key in missing if key.startswith('evidence_net.')]
    unexpected = [key for key in unexpected if not key.startswith('geometry_heads.')]
    print(
        "[CGA pretrained OHEM] checkpoint=%s mapped_keys=%d missing_evidence=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_evidence), len(unexpected)),
        flush=True,
    )
    if missing_evidence:
        raise RuntimeError("[CGA pretrained OHEM] missing evidence keys sample: %s" % missing_evidence[:8])
    if unexpected:
        raise RuntimeError("[CGA pretrained OHEM] unexpected keys sample: %s" % unexpected[:8])


def load_ecdv_pretrained_evidence(net, checkpoint_path, device):
    base_net = unwrap_model(net)
    if base_net.model_name != 'ECDVMSHNet':
        raise ValueError("--ecdv_pretrained_evidence is only valid for ECDVMSHNet.")
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint_state_dict(checkpoint)
    mapped = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if clean_key.startswith('model.evidence_net.'):
            mapped[clean_key] = value
        elif clean_key.startswith('model.'):
            mapped['model.evidence_net.' + clean_key[len('model.'):]] = value
        else:
            mapped['model.evidence_net.' + clean_key] = value
    current = base_net.state_dict()
    mapped = {
        key: value
        for key, value in mapped.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing, unexpected = base_net.load_state_dict(mapped, strict=False)
    missing_evidence = [key for key in missing if key.startswith('model.evidence_net.')]
    print(
        "[ECDV pretrained OHEM] checkpoint=%s mapped_keys=%d missing_evidence=%d unexpected=%d"
        % (checkpoint_path, len(mapped), len(missing_evidence), len(unexpected)),
        flush=True,
    )
    if missing_evidence:
        print("[ECDV pretrained OHEM] missing evidence keys sample: %s" % missing_evidence[:8], flush=True)


def freeze_ecdv_evidence(net):
    base_net = unwrap_model(net)
    if base_net.model_name != 'ECDVMSHNet':
        raise ValueError("--ecdv_freeze_evidence is only valid for ECDVMSHNet.")
    for param in base_net.model.evidence_net.parameters():
        param.requires_grad = False
    base_net.model.evidence_net.eval()
    trainable = sum(param.numel() for param in base_net.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in base_net.parameters() if not param.requires_grad)
    print(
        "[ECDV Stage1] freeze_evidence=True trainable_params=%d frozen_params=%d beta_start_epoch=%d"
        % (trainable, frozen, opt.ecdv_beta_start_epoch),
        flush=True,
    )


def assert_erd_gate_audit_ready(opt):
    if opt.model_name != 'ERDMSHNet':
        return
    if not opt.erd_require_gate_audit:
        print('[ERD Warning] Gate audit guard disabled by --erd_no_gate_audit_guard', flush=True)
        return
    if opt.erd_gate_audit_json is None:
        raise RuntimeError(
            'ERDMSHNet training requires --erd_gate_audit_json. '
            'Run tools/official/audit_online_gate_candidates.py first.'
        )
    import json
    with open(opt.erd_gate_audit_json, 'r', encoding='utf-8') as f:
        audit = json.load(f)
    if not audit.get('gate_pass', False):
        raise RuntimeError('ERD gate audit failed: %s' % audit)


def assert_erd_v3_candidate_audit_ready(opt):
    if opt.model_name != 'ERDMSHNetV3':
        return
    if not opt.erd_require_candidate_audit:
        print('[ERD-v3 Warning] Candidate audit guard disabled; official training should pass --erd_require_candidate_audit', flush=True)
        return
    audit_json = opt.erd_candidate_audit_json
    if not audit_json or not os.path.exists(audit_json):
        raise RuntimeError(
            'ERDMSHNetV3 training requires a passing candidate audit. '
            'Run tools/official/audit_erd_v3_candidates.py first. Missing: %s' % audit_json
        )
    import json
    with open(audit_json, 'r', encoding='utf-8') as f:
        audit = json.load(f)
    if not audit.get('gate_pass', False):
        raise RuntimeError('ERD-v3 candidate audit failed: %s' % audit)


def assert_pfr_ready(opt):
    if opt.model_name != 'PFRMSHNet':
        return
    if not opt.pfr_ready_summary or not os.path.exists(opt.pfr_ready_summary):
        raise RuntimeError(
            'PFRMSHNet training requires --pfr_ready_summary from a passed candidate audit. Missing: %s'
            % opt.pfr_ready_summary
        )
    import json
    with open(opt.pfr_ready_summary, 'r', encoding='utf-8') as f:
        audit = json.load(f)
    if not audit.get('gate_pass', False):
        raise RuntimeError('PFR candidate audit failed: %s' % audit)


def assert_eacf_ready(opt):
    if opt.model_name != 'EACFMSHNet':
        return
    if not opt.pretrained_ohem_checkpoint:
        raise RuntimeError('EACFMSHNet requires --pretrained_ohem_checkpoint.')
    if not os.path.exists(opt.pretrained_ohem_checkpoint):
        raise RuntimeError('Missing EACF OHEM checkpoint: %s' % opt.pretrained_ohem_checkpoint)
    if not opt.eacf_freeze_backbone:
        raise RuntimeError('Gate-F2 EACF stage-1 requires --eacf_freeze_backbone.')


def assert_sacf_ready_for_training(opt):
    if opt.model_name != 'SACFMSHNet':
        return
    if not opt.ohem_checkpoint:
        raise RuntimeError('SACFMSHNet requires --ohem_checkpoint.')
    if not os.path.exists(opt.ohem_checkpoint):
        raise RuntimeError('Missing SACF OHEM checkpoint: %s' % opt.ohem_checkpoint)
    if not opt.freeze_evidence:
        raise RuntimeError('Current SACF Gate-S2 stage requires --freeze_evidence true.')


def assert_cga_ready_for_training(opt):
    if opt.model_name != 'CGAMSHNet':
        return
    if not opt.load_ohem_checkpoint:
        raise RuntimeError('CGAMSHNet requires --load_ohem_checkpoint.')
    if not os.path.exists(opt.load_ohem_checkpoint):
        raise RuntimeError('Missing CGA OHEM checkpoint: %s' % opt.load_ohem_checkpoint)


def assert_ecdv_ready_for_training(opt):
    if opt.model_name != 'ECDVMSHNet':
        return
    if not opt.ecdv_decoy_bank:
        raise RuntimeError('ECDVMSHNet requires --ecdv_decoy_bank.')
    summary_path = opt.ecdv_decoy_bank_summary or os.path.join(opt.ecdv_decoy_bank, 'summary.json')
    check_ecdv_gate_b_summary(summary_path)
    if not opt.ecdv_pretrained_evidence:
        raise RuntimeError('ECDVMSHNet requires --ecdv_pretrained_evidence.')
    if not os.path.exists(opt.ecdv_pretrained_evidence):
        raise RuntimeError('Missing ECDV pretrained evidence checkpoint: %s' % opt.ecdv_pretrained_evidence)


def freeze_sacf_evidence_train_fusion(net):
    base_net = unwrap_model(net)
    if base_net.model_name != 'SACFMSHNet':
        raise ValueError("--freeze_evidence is only valid for SACFMSHNet.")
    base_net.model.freeze_evidence()
    trainable = [name for name, param in base_net.named_parameters() if param.requires_grad]
    if not any('fusion' in name for name in trainable):
        raise RuntimeError('SACF freeze error: fusion parameters are not trainable: %s' % trainable[:20])
    if any('encoder' in name or 'evidence_net' in name for name in trainable):
        raise RuntimeError('SACF freeze error: evidence parameters remain trainable: %s' % trainable[:20])
    print("[SACF] trainable params after freeze: %s" % trainable[:20], flush=True)


def configure_cga_trainable(net, mode):
    base_net = unwrap_model(net)
    if base_net.model_name != 'CGAMSHNet':
        raise ValueError("--cga_train_mode is only valid for CGAMSHNet.")
    for name, param in base_net.model.named_parameters():
        param.requires_grad = False
        if mode == 'decoder_aux':
            if (
                'geometry_heads' in name
                or 'evidence_net.decoder_0' in name
                or 'evidence_net.output_0' in name
                or 'evidence_net.final' in name
            ):
                param.requires_grad = True
        elif mode == 'aux_only':
            if 'geometry_heads' in name:
                param.requires_grad = True
        else:
            raise ValueError(mode)
    trainable = [name for name, param in base_net.named_parameters() if param.requires_grad]
    if not any('geometry_heads' in name for name in trainable):
        raise RuntimeError('CGA trainable error: geometry_heads not trainable: %s' % trainable[:20])
    if any('encoder_0' in name or 'encoder_1' in name or 'encoder_2' in name or 'encoder_3' in name for name in trainable):
        raise RuntimeError('CGA trainable error: encoder remains trainable: %s' % trainable[:20])
    print("[CGA] train_mode=%s trainable params sample: %s" % (mode, trainable[:20]), flush=True)


def freeze_ohcm_backbone(net):
    base_net = unwrap_model(net)
    if base_net.model_name not in ('OHCMMSHNet', 'OHCMMSHNetFull'):
        raise ValueError("--ohcm_freeze_backbone is only valid for OHCM models.")
    for param in base_net.model.backbone.parameters():
        param.requires_grad = False
    base_net.model.backbone.eval()
    trainable = sum(param.numel() for param in base_net.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in base_net.parameters() if not param.requires_grad)
    print(
        "[OHCM FrozenCalib] freeze_backbone=True trainable_params=%d frozen_params=%d"
        % (trainable, frozen),
        flush=True,
    )


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def seed_worker(worker_id):
    worker_seed = opt.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_mshnet_family(model_name):
    return model_name in ('MSHNet', 'MSHNetFocal', 'MSHNetOHEM', 'MSHNetTopKNeg', 'MSHNetSPSOHEM')


def is_ecdv_model(model_name):
    return model_name == 'ECDVMSHNet'


def checkpoint_matches_model(path, model_name, dataset_name):
    if dataset_name not in path:
        return False
    if model_name in path:
        return True
    return model_name == 'MSHNetSPSOHEM' and 'MSHNetOHEM' in path


def build_sps_view(img, gt_mask):
    op = opt.sps_perturbation
    if op == 'mixed':
        op = random.choice(['hflip', 'vflip', 'transpose', 'gain_offset', 'gaussian_noise'])

    view = img
    view_gt = gt_mask
    inverse_op = op

    if op == 'hflip':
        view = torch.flip(img, dims=[-1])
        view_gt = torch.flip(gt_mask, dims=[-1])
    elif op == 'vflip':
        view = torch.flip(img, dims=[-2])
        view_gt = torch.flip(gt_mask, dims=[-2])
    elif op == 'hvflip':
        view = torch.flip(img, dims=[-2, -1])
        view_gt = torch.flip(gt_mask, dims=[-2, -1])
    elif op == 'transpose':
        view = img.transpose(-1, -2).contiguous()
        view_gt = gt_mask.transpose(-1, -2).contiguous()
    elif op == 'gain_offset':
        gain = random.uniform(opt.sps_gain_min, opt.sps_gain_max)
        offset = random.uniform(-opt.sps_offset_abs, opt.sps_offset_abs)
        view = img * gain + offset
        inverse_op = 'identity'
    elif op == 'gaussian_noise':
        view = img + torch.randn_like(img) * opt.sps_noise_std
        inverse_op = 'identity'
    else:
        inverse_op = 'identity'

    return view.contiguous(), view_gt.contiguous(), inverse_op


def trainable_parameter_names(net):
    return [name for name, param in unwrap_model(net).named_parameters() if param.requires_grad]


def make_checkpoint_state(epoch, net, optimizer, scheduler, total_loss_list, generator):
    return {
        'epoch': epoch,
        'state_dict': unwrap_model(net).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'total_loss': total_loss_list,
        'config': checkpoint_config(),
        'seed': opt.seed,
        'rng_state': capture_rng_state(),
        'dataloader_generator_state': generator.get_state(),
        'git_commit': get_git_commit(),
        'best_metric': None,
        'dataset_split_hash': dataset_split_hash(),
        'trainable_parameter_names': trainable_parameter_names(net),
        'checkpoint_schema_version': 2,
    }


def train():
    device = get_device()
    configure_optimizer_schedule()
    assert_branch_allowed(opt.model_name, allow_stopped_branch=opt.allow_stopped_branch)
    assert_erd_gate_audit_ready(opt)
    assert_erd_v3_candidate_audit_ready(opt)
    assert_pfr_ready(opt)
    assert_eacf_ready(opt)
    assert_sacf_ready_for_training(opt)
    assert_cga_ready_for_training(opt)
    assert_ecdv_ready_for_training(opt)
    resume_path, resume_ckpt = find_matching_checkpoint(opt.resume, opt.model_name, opt.dataset_name, device)
    if opt.model_name == 'ERDMSHNet' and opt.erd_freeze_evidence_until > 0:
        raise ValueError("--erd_freeze_evidence_until is reserved; keep it 0 for the first ERD version.")
    ecdv_decoy_bank = None
    if is_ecdv_model(opt.model_name):
        ecdv_decoy_bank = ECDVDecoyBank(opt.ecdv_decoy_bank)
    return_train_meta = bool(opt.tsr_bank_path) or is_ecdv_model(opt.model_name)
    train_set = TrainSetLoader(
        dataset_dir=opt.dataset_dir,
        dataset_name=opt.dataset_name,
        patch_size=opt.patchSize,
        img_norm_cfg=opt.img_norm_cfg,
        return_meta=return_train_meta,
    )
    generator = torch.Generator()
    generator.manual_seed(opt.seed)
    if isinstance(resume_ckpt, dict) and resume_ckpt.get('dataloader_generator_state') is not None:
        dataloader_generator_state = resume_ckpt['dataloader_generator_state']
        if torch.is_tensor(dataloader_generator_state):
            dataloader_generator_state = dataloader_generator_state.detach().cpu().to(torch.uint8)
        generator.set_state(dataloader_generator_state)
    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    net = Net(
        model_name=opt.model_name,
        mode='train',
        loss_cfg=vars(opt),
    ).to(device)
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if resume_ckpt is not None:
        load_model_checkpoint(net, resume_ckpt)
        if isinstance(resume_ckpt, dict):
            epoch_state = int(resume_ckpt.get('epoch', 0))
            total_loss_list = list(resume_ckpt.get('total_loss', []))
        print(f"[Resume] loaded checkpoint={resume_path} epoch={epoch_state}", flush=True)
    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if checkpoint_matches_model(pretrained_pth, opt.model_name, opt.dataset_name):
                ckpt = torch_load_checkpoint(pretrained_pth, device)
                load_model_checkpoint(net, ckpt)

    if opt.ohcm_pretrained_backbone:
        load_ohcm_pretrained_backbone(net, opt.ohcm_pretrained_backbone, device)
    if opt.ohcm_freeze_backbone:
        freeze_ohcm_backbone(net)
    if opt.erd_pretrained_evidence:
        load_erd_pretrained_evidence(net, opt.erd_pretrained_evidence, device)
    if opt.erd_pretrained_ohem:
        load_erd_v3_pretrained_ohem(net, opt.erd_pretrained_ohem, device)
    if opt.pfr_pretrained_ohem:
        load_pfr_pretrained_ohem(net, opt.pfr_pretrained_ohem, device)
    if opt.model_name == 'EACFMSHNet':
        load_eacf_ohem_checkpoint(net, opt.pretrained_ohem_checkpoint, device)
        if opt.eacf_freeze_backbone:
            unwrap_model(net).model.freeze_backbone()
    if opt.model_name == 'SACFMSHNet':
        load_sacf_ohem_checkpoint(net, opt.ohem_checkpoint, device)
        if opt.freeze_evidence:
            freeze_sacf_evidence_train_fusion(net)
    if opt.model_name == 'CGAMSHNet':
        if resume_ckpt is None:
            load_cga_ohem_checkpoint(net, opt.load_ohem_checkpoint, device)
        else:
            print("[CGA Resume] keeping checkpoint weights; skip OHEM re-initialization.", flush=True)
        configure_cga_trainable(net, opt.cga_train_mode)
    if opt.model_name == 'ECDVMSHNet':
        load_ecdv_pretrained_evidence(net, opt.ecdv_pretrained_evidence, device)
        if opt.ecdv_freeze_evidence:
            freeze_ecdv_evidence(net)

    if opt.use_parallel and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)
    if isinstance(resume_ckpt, dict):
        if resume_ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
        if resume_ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(resume_ckpt['scheduler'])
        restore_rng_state(resume_ckpt)
    loss_model = unwrap_model(net)

    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, batch in enumerate(train_loader):
            if return_train_meta:
                img, gt_mask, image_ids, aug_ops = batch
            else:
                img, gt_mask = batch
                image_ids, aug_ops = None, None
            img, gt_mask = Variable(img).to(device), Variable(gt_mask).to(device)
            if aug_ops is not None:
                aug_ops = aug_ops.to(device)
            if img.shape[0] == 1:
                continue
            if opt.model_name == 'HCNet':
                pred = net(img)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'HCNet loss detail | '
                        'sls: %.6f | hc: %.6f | hard_ratio: %.6f | hard_prob: %.6f'
                        % (
                            float(loss_out['sls'].detach().cpu()),
                            float(loss_out['hc'].detach().cpu()),
                            float(loss_out['hard_ratio'].detach().cpu()),
                            float(loss_out['hard_prob_mean'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'ERDMSHNet':
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'ERD loss detail | '
                        'total: %.6f | det: %.6f | evidence: %.6f | gate_pos: %.6f | gate_neg: %.6f | '
                        'gate_w: %.4f | neg_px: %.1f | neg_per_img_mean: %.1f | neg_per_img_min: %.1f'
                        % (
                            float(loss_out['total'].detach().cpu()),
                            float(loss_out['sls'].detach().cpu()),
                            float(loss_out['evidence'].detach().cpu()),
                            float(loss_out['gate_pos'].detach().cpu()),
                            float(loss_out['gate_neg'].detach().cpu()),
                            float(loss_out['gate_w'].detach().cpu()),
                            float(loss_out['gate_neg_pixels'].detach().cpu()),
                            float(loss_out['gate_neg_per_image_mean'].detach().cpu()),
                            float(loss_out['gate_neg_per_image_min'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'ERDMSHNetV3':
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'ERD-v3 loss detail | '
                        'total: %.6f | final: %.6f | evidence: %.6f | '
                        'protect_pos: %.6f | protect_neg: %.6f | '
                        'clutter_pos: %.6f | clutter_neg: %.6f | preserve: %.6f | '
                        'neg_px_mean: %.1f | pt: %.4f | pbg: %.4f | ct: %.4f | cbg: %.4f | '
                        'st: %.4f | sbg: %.4f'
                        % (
                            float(loss_out['total'].detach().cpu()),
                            float(loss_out['loss_final'].detach().cpu()),
                            float(loss_out['loss_evidence'].detach().cpu()),
                            float(loss_out['loss_protect_pos'].detach().cpu()),
                            float(loss_out['loss_protect_neg'].detach().cpu()),
                            float(loss_out['loss_clutter_pos'].detach().cpu()),
                            float(loss_out['loss_clutter_neg'].detach().cpu()),
                            float(loss_out['loss_preserve'].detach().cpu()),
                            float(loss_out['mean_online_neg_pixels'].detach().cpu()),
                            float(loss_out['mean_protection_target'].detach().cpu()),
                            float(loss_out['mean_protection_far_bg'].detach().cpu()),
                            float(loss_out['mean_clutter_target'].detach().cpu()),
                            float(loss_out['mean_clutter_far_bg'].detach().cpu()),
                            float(loss_out['mean_suppression_target'].detach().cpu()),
                            float(loss_out['mean_suppression_far_bg'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'PFRMSHNet':
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'PFR loss detail | '
                        'total: %.6f | seg: %.6f | far_neg: %.6f | target_protect: %.6f | '
                        'boundary_protect: %.6f | residual: %.6f | far_hard_px: %.1f | '
                        'target_px: %.1f | boundary_px: %.1f | delta_abs: %.6f'
                        % (
                            float(loss_out['total'].detach().cpu()),
                            float(loss_out['seg'].detach().cpu()),
                            float(loss_out['far_neg'].detach().cpu()),
                            float(loss_out['target_protect'].detach().cpu()),
                            float(loss_out['boundary_protect'].detach().cpu()),
                            float(loss_out['residual_sparse'].detach().cpu()),
                            float(loss_out['far_hard_pixels'].detach().cpu()),
                            float(loss_out['target_pixels'].detach().cpu()),
                            float(loss_out['boundary_pixels'].detach().cpu()),
                            float(loss_out['delta_abs_mean'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'EACFMSHNet':
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'EACF loss detail | '
                        'total: %.6f | main: %.6f | anchor: %.6f | '
                        'scale_tgt: %.6f | scale_bg: %.6f | eta: %.6f'
                        % (
                            float(loss_out['total'].detach().cpu()),
                            float(loss_out['main'].detach().cpu()),
                            float(loss_out['anchor'].detach().cpu()),
                            float(loss_out['scale_target'].detach().cpu()),
                            float(loss_out['scale_bg'].detach().cpu()),
                            float(loss_out['eta'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'SACFMSHNet':
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'SACF loss detail | '
                        'total: %.6f | main: %.6f | anchor: %.6f | scale: %.6f | '
                        'disagree_bg: %.6f | gate: %.6f | delta_abs: %.6f'
                        % (
                            float(loss_out['total'].detach().cpu()),
                            float(loss_out['main'].detach().cpu()),
                            float(loss_out['anchor'].detach().cpu()),
                            float(loss_out['scale'].detach().cpu()),
                            float(loss_out['disagree_bg'].detach().cpu()),
                            float(loss_out['fusion_gate_mean'].detach().cpu()),
                            float(loss_out['fusion_delta_abs_mean'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'CGAMSHNet':
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(pred, gt_mask, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'CGA loss detail | '
                        'total: %.6f | mask: %.6f | center: %.6f | scale: %.6f | '
                        'core: %.6f | boundary: %.6f | peak_bg: %.6f | peak_count: %.1f'
                        % (
                            float(loss_out['total'].detach().cpu()),
                            float(loss_out['mask'].detach().cpu()),
                            float(loss_out['center'].detach().cpu()),
                            float(loss_out['scale'].detach().cpu()),
                            float(loss_out['core'].detach().cpu()),
                            float(loss_out['boundary'].detach().cpu()),
                            float(loss_out['peak_bg'].detach().cpu()),
                            float(loss_out['peak_count'].detach().cpu()),
                        )
                    )
            elif opt.model_name == 'ECDVMSHNet':
                residual, pseudo_fp_mask, _decoy_rows = ecdv_decoy_bank.sample_batch(image_ids, img)
                img_aug = apply_decoy_batch(img, residual, pseudo_fp_mask)
                pred = net(img_aug, epoch=idx_epoch + 1)
                loss_out = loss_model.loss(
                    pred,
                    gt_mask,
                    idx_epoch + 1,
                    ecdv_pseudo_fp_mask=pseudo_fp_mask,
                    ecdv_stage=opt.ecdv_stage,
                )
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'ECDV loss detail | '
                        'stage: %s | risk: %.6f | guard: %.6f | keep: %.6f | '
                        'suppress: %.6f | beta: %.4f | pseudo_px: %.1f | '
                        'pseudo_img: %.1f | risk_pseudo: %.6f | risk_target: %.6f | suppression: %.6f'
                        % (
                            opt.ecdv_stage,
                            float(loss_out['risk_loss'].detach().cpu()),
                            float(loss_out['target_guard_loss'].detach().cpu()),
                            float(loss_out['keep_loss'].detach().cpu()),
                            float(loss_out['suppress_loss'].detach().cpu()),
                            float(loss_out['ecdv_beta'].detach().cpu()),
                            float(loss_out['pseudo_fp_pixels'].detach().cpu()),
                            float(loss_out['pseudo_fp_images'].detach().cpu()),
                            float(loss_out['risk_prob_pseudo_mean'].detach().cpu()),
                            float(loss_out['risk_prob_target_mean'].detach().cpu()),
                            float(loss_out['suppression_mean'].detach().cpu()),
                        )
                    )
            elif is_mshnet_family(opt.model_name):
                if opt.model_name == 'MSHNetSPSOHEM':
                    sps_active = (
                        opt.sps_mode != 'none'
                        and opt.sps_lambda > 0
                        and (idx_epoch + 1) > opt.sps_start_epoch
                    )
                    if sps_active:
                        sps_img, sps_gt_mask, sps_op = build_sps_view(img, gt_mask)
                        both_pred = net(torch.cat([img, sps_img], dim=0), epoch=idx_epoch + 1)
                        masks_all, final_all = both_pred
                        batch_size = img.shape[0]
                        pred = (
                            [mask[:batch_size] for mask in masks_all],
                            final_all[:batch_size],
                        )
                        sps_pred = (
                            [mask[batch_size:] for mask in masks_all],
                            final_all[batch_size:],
                        )
                    else:
                        pred = net(img, epoch=idx_epoch + 1)
                        sps_pred, sps_gt_mask, sps_op = None, None, 'identity'
                else:
                    pred = net(img, epoch=idx_epoch + 1)
                    sps_pred, sps_gt_mask, sps_op = None, None, 'identity'
                loss_out = loss_model.loss(
                    pred,
                    gt_mask,
                    idx_epoch + 1,
                    image_ids=image_ids,
                    aug_ops=aug_ops,
                    sps_pred=sps_pred,
                    sps_gt_mask=sps_gt_mask,
                    sps_op=sps_op,
                )
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'MSHNet loss detail | '
                        'sls: %.6f | variant: %.6f | region: %.6f | lambda_reg: %.4f | '
                        'sps: %.6f | lambda_sps: %.4f | sps_hard_px: %.1f | sps_cand_px: %.1f | '
                        'sps_u: %.6f | sps_score: %.6f | sps_tgt_scale: %.4f | '
                        'sps_tgt_u: %.6f | sps_tgt_conf: %.6f | '
                        'hard_regions: %.1f | empty_region: %.3f | hard_score: %.6f | '
                        'target_regions: %.1f | gap: %.6f | aux_count: %.0f'
                        % (
                            float(loss_out['sls'].detach().cpu()),
                            float(loss_out['variant_loss'].detach().cpu()),
                            float(loss_out['region_loss'].detach().cpu()),
                            float(loss_out['lambda_region'].detach().cpu()),
                            float(loss_out['sps_loss'].detach().cpu()),
                            float(loss_out['lambda_sps'].detach().cpu()),
                            float(loss_out['sps_hard_pixels'].detach().cpu()),
                            float(loss_out['sps_candidate_pixels'].detach().cpu()),
                            float(loss_out['sps_instability_mean'].detach().cpu()),
                            float(loss_out['sps_score_mean'].detach().cpu()),
                            float(loss_out['sps_target_alpha_scale'].detach().cpu()),
                            float(loss_out['sps_target_instability_mean'].detach().cpu()),
                            float(loss_out['sps_target_conf_mean'].detach().cpu()),
                            float(loss_out['hard_regions'].detach().cpu()),
                            float(loss_out['empty_region_ratio'].detach().cpu()),
                            float(loss_out['hard_region_score_mean'].detach().cpu()),
                            float(loss_out['target_regions'].detach().cpu()),
                            float(loss_out['region_logit_gap'].detach().cpu()),
                            float(loss_out['aux_count'].detach().cpu()),
                        )
                    )
            elif opt.model_name in ('OHCMMSHNet', 'OHCMMSHNetFull'):
                pred = net(img, epoch=idx_epoch + 1)
                loss_out = loss_model.loss_with_image(pred, gt_mask, img, idx_epoch + 1)
                loss = loss_out['total']

                if idx_iter == 0 and (idx_epoch + 1) % opt.intervals == 0:
                    print(
                        'OHCM loss detail | '
                        'sls: %.6f | clu: %.6f | sup: %.6f | margin: %.6f | proto: %.6f | '
                        'hard_px: %.1f | hard_comp: %.1f | empty: %.3f | area: %.2f | hard_score: %.6f | '
                        'pt_tgt: %.6f | pt_hard: %.6f | pc_tgt: %.6f | pc_hard: %.6f | '
                        'drop_tgt: %.6f | drop_hard: %.6f | gamma: %.4f'
                        % (
                            float(loss_out['sls'].detach().cpu()),
                            float(loss_out['clu'].detach().cpu()),
                            float(loss_out['sup'].detach().cpu()),
                            float(loss_out['margin'].detach().cpu()),
                            float(loss_out['proto'].detach().cpu()),
                            float(loss_out['hard_pixels'].detach().cpu()),
                            float(loss_out['hard_components'].detach().cpu()),
                            float(loss_out['empty_mining_ratio'].detach().cpu()),
                            float(loss_out['hard_area_mean'].detach().cpu()),
                            float(loss_out['hard_score_mean'].detach().cpu()),
                            float(loss_out['target_prob_mean'].detach().cpu()),
                            float(loss_out['hard_prob_mean'].detach().cpu()),
                            float(loss_out['target_clutter_prob_mean'].detach().cpu()),
                            float(loss_out['hard_clutter_prob_mean'].detach().cpu()),
                            float(loss_out['target_prob_drop'].detach().cpu()),
                            float(loss_out['hard_prob_drop'].detach().cpu()),
                            float(loss_out['gamma'].detach().cpu()),
                        )
                    )
            else:
                pred = net(img)
                loss = loss_model.loss(pred, gt_mask)

            total_loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每 10 个 batch 输出一次进度
            if (idx_iter + 1) % 10 == 0:
                print(
                    f"Epoch [{idx_epoch + 1}/{opt.nEpochs}], Batch [{idx_iter + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        if (idx_epoch + 1) % opt.intervals == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,'
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []

        if (idx_epoch + 1) % 50 == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint(
                make_checkpoint_state(idx_epoch + 1, net, optimizer, scheduler, total_loss_list, generator),
                save_pth,
            )
            if opt.eval_during_train:
                test(save_pth)

        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint(
                make_checkpoint_state(idx_epoch + 1, net, optimizer, scheduler, total_loss_list, generator),
                save_pth,
            )
            if opt.eval_during_train:
                test(save_pth)


def test(save_pth):
    device = get_device()
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    #net = Net(model_name=opt.model_name, mode='test').cuda()
    net = Net(
        model_name=opt.model_name,
        mode='test',
        loss_cfg=vars(opt),
    ).to(device)
    ckpt = torch_load_checkpoint(save_pth, device)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()

    eval_metrics = BinaryMetricsGPU(threshold=opt.threshold, device=device)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            #img = Variable(img).cuda()
            img = Variable(img).to(device)
            gt_mask = gt_mask.to(device)
            pred = net(img)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            pred = pred[:, :, :h, :w]
            gt_mask = gt_mask[:, :, :h, :w]
            eval_metrics.update(pred, gt_mask)
            if (idx_iter + 1) % 100 == 0:
                print(f"Test [{idx_iter + 1}/{len(test_loader)}]", flush=True)
    
    metric_results = eval_metrics.get()
    params_count = Params(net)

    result_lines = [
        "pixAcc:\t" + str(metric_results['pixAcc']),
        "mIoU:\t" + str(metric_results['mIoU']),
        "nIoU:\t" + str(metric_results['nIoU']),
        "PixelRecall:\t" + str(metric_results['PixelRecall']),
        "Fa:\t" + str(metric_results['Fa']),
        "Precision:\t" + str(metric_results['Precision']),
        "Recall:\t" + str(metric_results['Recall']),
        "F1:\t" + str(metric_results['F1']),
        "Params:\t%s (%d)" % (format_params(params_count), params_count),
    ]
    for line in result_lines:
        print(line, flush=True)
        opt.f.write(line + '\n')
    opt.f.flush()

    try:
        flops_count = FLOPs(net, input_size=(1, 1, opt.patchSize, opt.patchSize), device=device)
        flops_line = "FLOPs:\t%s (%d)" % (format_flops(flops_count), flops_count)
    except Exception as exc:
        flops_line = "FLOPs:\tUnavailable (%s)" % exc
    print(flops_line, flush=True)
    opt.f.write(flops_line + '\n')
    opt.f.flush()
    
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
