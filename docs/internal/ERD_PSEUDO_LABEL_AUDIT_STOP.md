# ERD Offline Pseudo-Label Audit Stop

Date: 2026-06-28

SPS HOLD and offline reliability-label ERD training are stopped for AAAI design
decisions.

## Stop Evidence

SPS-OHEM pixel / region / peak Gate0 failed:

- selected OHEM overlap stayed at 1.0;
- candidate-to-budget ratio was too low or empty;
- peak-region candidate count reached 0.0.

Offline TCE/OHEM reliability pseudo-label audit also failed:

```text
rel_neg_pixels_mean: 0.0574
num_images_without_neg: 695 / 697
images_without_neg_ratio: 0.9971
target_leakage_neg_pixels: 0
gate_pass: False
```

## Decision

Do not continue SPS alpha / lambda / topk sweeps.

Do not tune offline pseudo-label thresholds to rescue sparse reliability
negatives.

The next allowed step is ERD-MSHNet v2 online gate candidate audit on the train
split only.
