# APF-OHEM Gate-A Candidate Audit Failure

## Result

```text
anchor_maps_gate_pass = true
candidate_audit_gate_pass = false
candidate_to_budget_ratio_mean = 1.0
flat_bg_ratio_mean = 0.9993765
target_leakage_pixels_total = 0
ohem_fp_component_coverage_mean = 0.5
```

## Interpretation

APF candidates are safe but not useful. They are dominated by flat background
and do not provide a reliable hard false-positive supervision signal.

## Decision

- Stop APF training.
- Do not run seed42.
- Do not tune APF thresholds or lambdas.
- Move to error-component audit before any new method design.
