# SACF Activation Sanity Protocol

Run only seed42 and at most 1 epoch:

```bash
CUDA_VISIBLE_DEVICES=1 bash tools/official/train_sacf_seed.sh 42 --max_epochs 1 --freeze_evidence true
```

Then audit:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/official/audit_sacf_activation.py \
  --dataset_dir /home/AAAI/OHCM-MSHNet/datasets \
  --dataset_name NUDT-SIRST \
  --split train \
  --model_name SACFMSHNet \
  --checkpoint /home/AAAI/OHCM-MSHNet/results/official/SACFMSHNet/seed42/NUDT-SIRST/SACFMSHNet_1.pth.tar \
  --output_dir docs/internal/sacf_activation_seed42_train_epoch1
```

Go requires:

```text
mean_abs_final_minus_base_prob > 1e-4
fusion_gate_mean > 1e-3
fusion_delta_abs_mean > 1e-5
changed_pixel_ratio_at_0p5 > 0
checkpoint_has_fusion_keys = true
optimizer_has_fusion_params = true
```
