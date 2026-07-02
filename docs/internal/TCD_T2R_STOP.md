# TCD Gate-T2R Stop Record

## Status

Gate-T2R teacher-information audit failed on train-only TCE soft labels.

## Inputs

- Labels: `docs/internal/tce_soft_labels/seed42_train`
- Audit summary: `docs/internal/tce_teacher_info_audit_seed42_train/summary.json`
- Container: Docker `3ca`
- GPU setting: `CUDA_VISIBLE_DEVICES=1`

## Result

```text
gate_pass = false
global_absdiff_mean = 1.637253e-05
topk_far_absdiff_mean = 0.000506739
teacher_lower_on_student_high_far_rate = 0.794674
teacher_preserves_target_rate = 0.954343
informative_image_ratio = 0.007174
fail_reasons = [
  "topk_far_teacher_student_diff_too_small",
  "too_few_informative_images"
]
```

## Decision

TCD dense/conditional teacher-distillation is stopped for the current code state.
The TCE teacher does not provide enough hard-region signal for seed42 training.

## Allowed Next Work

- Implement PFR-MSHNet.
- Run PFR train-only candidate audit.
- Do not train PFR unless the candidate audit passes.
- Do not run seed43/44, HC-Test, blind, or external.
