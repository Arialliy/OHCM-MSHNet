# EACF-v1 Identity Collapse Stop

## Checkpoint

```text
/home/ly/AAAI/OHCM-MSHNet/results/official/EACFMSHNet/seed42/NUDT-SIRST/EACFMSHNet_80.pth.tar
```

## Full Result

- base == final
- eta = 0.0
- mIoU = 0.834392849
- Pd = 0.979894180
- Precision = 0.906277382
- FA = 61.448798 ppm
- FP components = 47

## HC-Val Result

- base == final
- eta = 0.0
- mIoU = 0.604790419
- Pd = 0.833333333
- Precision = 0.665934066
- FA = 386.555990 ppm
- FP components = 9

## Interpretation

EACF-v1 preserves MSHNetOHEM but does not activate its fusion branch. The failure mode is identity collapse, not evidence pollution.

## Decision

Stop EACF-v1.

Do not run seed43/44.
Do not tune eta or losses.
Do not use HC-Test, blind, or external evaluation.
Move to SACF-MSHNet / DASF-MSHNet.
