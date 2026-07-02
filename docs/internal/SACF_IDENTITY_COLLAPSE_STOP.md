# SACF-v1 Identity Collapse Stop

## Checkpoint

```text
/home/ly/AAAI/OHCM-MSHNet/results/official/SACFMSHNet/seed42/NUDT-SIRST/SACFMSHNet_1.pth.tar
```

## Gate-S2a Result

- gate_pass = false
- fail_reasons = ["final_equals_base_identity_collapse"]
- mean_abs_final_minus_base_prob = 1.020973338e-06
- changed_pixel_ratio_at_0p5 = 8.756837339e-07
- fusion_gate_mean = 0.113682778
- fusion_delta_abs_mean = 0.991195551
- checkpoint_has_fusion_keys = true
- optimizer_has_fusion_params = true

## Interpretation

SACF-v1 has trainable fusion parameters and saved fusion keys, but the final probability remains effectively identical to the MSHNetOHEM base path. The failure mode is identity collapse, not evidence pollution.

## Decision

Stop SACF-v1.

Do not run SACF 80 epoch.
Do not tune SACF gate, lambda, learning rate, or epoch count.
Do not use HC-Test, blind, or external evaluation.
Move to CGA-MSHNet.
