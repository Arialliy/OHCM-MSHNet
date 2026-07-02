# PFR-MSHNet Seed42 Full Gate Failure

## Checkpoint

```text
/home/ly/AAAI/OHCM-MSHNet/results/official/PFRMSHNet/seed42/NUDT-SIRST/PFRMSHNet_400.pth.tar
```

## Full Gate Result

| Metric | PFR | Required | Status |
|---|---:|---:|---|
| mIoU | 0.780196 | >= 0.833393 | FAIL |
| Pd | 0.986243 | >= 0.979894 | PASS |
| Precision | 0.865911 | >= 0.906277 | FAIL |
| FA ppm | 89.416 | <= 63.449 | FAIL |
| FP components | 114 | OHEM Full = 47 | FAIL |

## Failure Audit

```text
total_target_lost_count = 2
total_boundary_excess_delta = 652
failure_mode = structural_suppression_or_calibration_regression
```

## Interpretation

PFR preserves target-level detection probability but produces many new
background false-positive components. The dominant failure is output
calibration / residual regression, not target recall failure.

## Decision

```text
No seed43/44.
No HC-Test.
No blind / external evaluation.
No PFR hyperparameter tuning.
PFR remains only for failure analysis.
```
