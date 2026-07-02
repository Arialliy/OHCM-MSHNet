# OHEM Error-Component Audit Stop

## Context

APF Gate-A failed because pixel-level candidates were dominated by flat
background. The next allowed step was a train-split OHEM error-component audit
to decide whether enough stable detached far false-positive components exist
to support AEC-OHEM.

## Build Result

```text
num_images = 697
num_written = 697
component_count_total = 1013
target_hit_components = 978
target_leakage_components = 0
target_leakage_pixels_total = 0
```

## Gate-ECA Result

```text
gate_pass = false
fail_reasons:
- total_detached_far_fp_components_too_low
- images_with_detached_far_fp_ratio_too_low
- train_candidate_to_budget_ratio_mean_too_low

total_detached_far_fp_components = 28
images_with_detached_far_fp = 3
images_with_detached_far_fp_ratio = 0.00430416068866571
nonflat_detached_far_fp_ratio = 1.0
target_like_area_detached_far_fp_ratio = 0.4642857142857143
mean_detached_far_fp_peak_prob = 0.8937530091830662
target_leakage_components = 0
boundary_excess_dominance_ratio = 0.17142857142857143
train_candidate_to_budget_ratio_mean = 0.0004544317043857767
flat_bg_ratio_mean = 0.0
```

## Ready Check

```text
error_component_ready = false
errors:
- error_component_audit_gate_failed
- total_detached_far_fp_components_too_low
```

## Decision

```text
Stop APF-OHEM.
Stop AEC-OHEM.
Stop component-mining OHEM.
Do not run APF/AEC seed42.
Do not run seed43/44.
Do not tune APF thresholds, top-k, quantiles, or loss lambdas.
Do not use HC-Test, blind, or external evaluation for these routes.
```

The train split does contain a few high-confidence, nonflat detached far-FP
components, but they are too sparse to support a reliable mining-based AAAI
method route.
