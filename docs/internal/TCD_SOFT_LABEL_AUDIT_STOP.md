# TCD Soft-Label Audit Stop Record

## Run context

- Container: Docker 3ca
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Code dir: `/home/ly/AAAI/OHCM-MSHNet-main`
- Original MSHNet reference: `/home/ly/AAAI/MSHNet`, not modified

## Inputs

- TCE soft labels: `docs/internal/tce_soft_labels/seed42_train`
- Count: `697 / 697`

## Gate-T2 result

```text
gate_pass = false
teacher_student_absdiff_mean = 1.637e-05
required_min_absdiff_mean = 0.001
```

## Decision

Naive dense TCE distillation is stopped. The next step is not training, but
Gate-T2R root-cause audit with conditional hard-region metrics.

## Stop constraints

```text
No TCD seed42.
No seed43/44.
No HC-Test / blind / external.
No ERD-v3 tuning.
```
