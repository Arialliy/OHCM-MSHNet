# CGA Component Target Audit Protocol

Gate-C1 validates GT-derived component geometry targets before any CGA training.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/official/audit_cga_component_targets.py \
  --dataset_dir /home/AAAI/OHCM-MSHNet/datasets \
  --dataset_name NUDT-SIRST \
  --split train \
  --output_dir docs/internal/cga_component_target_audit_seed42_train
```

Go requires:

```text
center targets non-empty
images_with_target_but_no_center = 0
local_bg_peak_count_mean >= min_k
scale_bin_max_ratio < 0.85
boundary_pixels_mean > 0
target_leakage_pixels = 0
```
