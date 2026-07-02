# ERD-MSHNet v2 seed42 Stop Record

## Status

ERD-MSHNet v2 online gate completed seed42 training and stopped at Gate-C.

## Results

### Full

| Metric | Delta vs OHEM |
|---|---:|
| mIoU | +0.0498 |
| Pd | +0.0011 |
| Precision | +0.0316 |
| FA | -20.93 ppm |
| FP components | -10 |

### HC-Val

| Metric | Delta vs OHEM |
|---|---:|
| mIoU | -0.0360 |
| Pd | +0.0000 |
| Precision | +0.0015 |
| FA | -35.60 ppm |
| FP components | -1 |

## Decision

No-Go for ERD-v2.

Reason:

- Full split passed.
- HC-Val did not pass because mIoU dropped and Precision gain was far below the roadmap threshold.
- The method suppresses false alarms, but may damage target mask quality in hard scenes.

## Allowed Next Work

- Run ERD-v2 failure audit.
- Design ERD-v3 target-preserving clutter suppression.
- Do not run ERD-v2 seed43/44.
- Do not tune HC-Test / blind / external.
