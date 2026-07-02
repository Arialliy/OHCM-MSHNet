# OHEM Error-Component Gate-E0 Failure

## Scope

This audit was run after APF Gate-A failed. It checks whether frozen
MSHNetOHEM train-split predictions contain enough reliable false-positive
connected components to support component-level mining.

## Inputs

```text
anchor_map_dir = docs/internal/ohem_anchor_maps/seed42_train
component_dir = docs/internal/ohem_error_components_seed42_train
audit_dir = docs/internal/ohem_error_component_audit_seed42_train
threshold = 0.5
gt_dilate_radius = 5
```

## Build Result

```text
num_images = 697
num_written = 697
component_count_total = 30
num_images_with_components = 5
target_leakage_pixels_total = 0
```

## Gate-E0 Result

```text
gate_pass = false
fail_reasons:
- num_images_with_components_ratio_too_low
- component_count_total_too_low

num_images_with_components_ratio = 0.007173601147776184
component_count_total = 30
flat_component_ratio = 0.0
mean_component_max_prob = 0.8963704427083333
detached_far_fp_component_ratio = 1.0
boundary_excess_only_ratio = 0.0
```

## Interpretation

The remaining OHEM train-split false-positive components are target-safe and
high confidence, but they are too sparse to support a reliable component-mining
training route. Only 5 of 697 training images contain such components, with 30
components in total.

## Decision

```text
Stop APF-OHEM.
Stop APF-v2.
Stop AEC-OHEM.
Stop component-mining OHEM routes.
Do not run seed42/43/44 for these routes.
Do not use HC-Test, blind, or external evaluation.
Do not tune APF thresholds or loss lambdas to force a pass.
```

The project should remain anchored on MSHNetOHEM strong baseline evidence and
failure-analysis diagnostics unless a different research question is defined.
