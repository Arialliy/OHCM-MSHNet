# ERDMSHNetV3 Full Gate Stop

## Checkpoint

`results/official/ERDMSHNetV3/seed42/NUDT-SIRST/ERDMSHNetV3_400.pth.tar`

## Full metrics

| Metric | ERDMSHNetV3 seed42 |
|---|---:|
| mIoU | 0.808049 |
| Pd | 0.984127 |
| Precision | 0.889289 |
| FA ppm | 72.778 |
| FP components | 38 |

## Decision

ERDMSHNetV3 failed the Full split gate. The result is below the MSHNetOHEM
strong baseline on mIoU, Precision, and FA.

## No-Go actions

- Do not run seed43 / seed44.
- Do not tune ERD-v3 suppression coefficients.
- Do not evaluate HC-Test / blind / external.
- Do not use ERD-v3 as AAAI main method.

## Next allowed actions

- Run ERD-v3 failure-mode audit.
- Build dense TCE soft labels from OHEM checkpoints.
- Design TCE-guided residual calibration instead of suppression-only gating.
