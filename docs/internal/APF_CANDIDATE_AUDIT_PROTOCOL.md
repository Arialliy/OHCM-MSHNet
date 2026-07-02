# APF Candidate Audit Protocol

## Purpose

APF candidate audit decides whether far-background candidates are safe enough to
permit APF seed42 training. It does not train a model.

## Candidate Rules

Candidates must be selected only from far background:

```text
far_bg = not dilate(gt_mask, target_dilation_radius)
```

The audit combines probability-window candidates and top-q OHEM-hard negatives:

```text
tau_low <= prob_ohem <= tau_high
or prob_ohem in top hard_top_q of far_bg
```

Candidates are finally intersected with `far_bg`, so target leakage should be
zero by construction.

## Gate-A

Minimum pass conditions:

```text
target_leakage_pixels_total == 0
num_images_with_candidate_empty / num_images <= 0.10
candidate_to_budget_ratio_mean >= 1.5
candidate_to_budget_ratio_positive_fraction >= 0.90
flat_bg_ratio_mean <= 0.35
ohem_fp_component_coverage_mean >= 0.40
```

If Gate-A fails, stop APF. Do not train APF, do not tune loss lambda, and do
not run seed42.
