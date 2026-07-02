# PFR Head Decomposition Audit Stop

## Scope

This audit is failure analysis only. It does not authorize PFR seed43/44,
HC-Test, blind, external evaluation, or PFR hyperparameter tuning.

## Input

Checkpoint:

```text
/home/ly/AAAI/OHCM-MSHNet/results/official/PFRMSHNet/seed42/NUDT-SIRST/PFRMSHNet_400.pth.tar
```

Audit output:

```text
docs/internal/pfr_head_audit_seed42_full/summary.json
```

## Head Metrics

| Metric | Evidence-only | Final |
|---|---:|---:|
| mIoU | 0.778985 | 0.780196 |
| Pd | 0.985185 | 0.986243 |
| Precision | 0.872860 | 0.865911 |
| FA ppm | 83.280 | 89.416 |
| FP components | 111 | 114 |

## Residual Findings

```text
residual_new_fp_pixels = 267
residual_new_fp_components = 234
residual_removed_fp_pixels = 0
residual_removed_fp_components = 0
residual_lost_target_pixels = 0
residual_boundary_excess_pixels = 158
delta_mean_target = 0.478759
delta_mean_boundary = 0.166256
delta_mean_far_bg = 0.001330
delta_positive_far_bg_ratio = 0.576573
```

## Decision Tree Result

Case A is triggered:

```text
evidence_mIoU < OHEM_mIoU - 0.005
evidence_Precision < OHEM_Precision - 0.005
evidence_FA_ppm > OHEM_FA_ppm + 5
```

Conclusion:

```text
PFR training polluted the MSHNetOHEM evidence branch.
All PFR-style end-to-end trainable head routes remain stopped.
Do not add PFR-v2 / PFR-v3 trainable correction heads.
Future candidates must preserve the MSHNetOHEM inference anchor.
```

## Next Allowed Step

Only APF-OHEM candidate audit is allowed.
No APF-OHEM training is allowed until the candidate audit passes.
PFR, ERD, TCD, and SPS failed-route training entry points are blocked by
`tools/official/check_failed_routes_blocked.py` by default.
