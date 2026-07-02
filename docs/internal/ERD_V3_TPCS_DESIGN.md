# ERD-MSHNet v3 TP-CS Design

## Context

ERD-MSHNet v2 seed42 stopped at Gate-C.

Full split improved over MSHNetOHEM, but HC-Val failed:

| Metric | HC-Val Delta vs OHEM |
|---|---:|
| mIoU | -0.0360 |
| Pd | +0.0000 |
| Precision | +0.0015 |
| FA | -35.60 ppm |
| FP components | -1 |

The follow-up failure audit reports:

```text
failure_mode = target_damage_or_support_shrinkage
mean_target_dilate_recall_delta = -0.0020
target_dilate_drop_gt_2pct_ratio = 0.3333
mean_suppression_selectivity = 0.4776
mean_gate_target_dilate = 0.9518
```

This means v2 can reduce false alarms, but the suppress-only reliability gate is not selective enough on HC-Val and can remove target support together with false positives.

## Goal

ERD-MSHNet v3 implements Target-Preserving Clutter Suppression (TP-CS):

```text
MSHNetOHEM evidence branch: where target-like evidence appears.
Protection head: which target-like responses should be protected.
Clutter head: which target-like responses are likely hard background clutter.
Constrained fusion: suppress unreliable evidence only; never create new target evidence.
```

## Fusion

For evidence logits `z_e`, protection logits `z_t`, and clutter logits `z_c`:

```text
T = sigmoid(z_t)
C = sigmoid(z_c)
suppression = s_max * C * (1 - T)
z_f = z_e - suppression
```

Required semantics:

```text
z_f <= z_e
s_max = 0 makes z_f == z_e
T -> 1 blocks suppression
C -> 0 blocks suppression
C -> 1 and T -> 0 strongly suppresses clutter
```

## Current Implementation Detail

Current repository MSHNet returns `x_d0` with 16 channels when called as:

```python
masks, logits, feature = MSHNet(...)(x, warm_flag=True, return_feature=True)
```

Therefore the default ERD-v3 `erd_aux_in_channels` is `16`, not the placeholder `32` in the planning sketch.

## Loss

ERD-v3 uses train-only online supervision:

```text
L_total =
  L_final_ohem(z_f, Y)
+ lambda_evidence * L_evidence_ohem(z_e, Y)
+ lambda_protect_pos * BCE(z_t, 1, Dilate(Y))
+ lambda_protect_neg * BCE(z_t, 0, high_evidence_far_bg)
+ lambda_clutter_pos * BCE(z_c, 1, high_evidence_far_bg)
+ lambda_clutter_neg * BCE(z_c, 0, Dilate(Y))
+ lambda_preserve * target_preserve(sigmoid(z_f), sigmoid(z_e), Dilate(Y))
```

Online negatives are selected from current-batch evidence:

```text
high_evidence_far_bg = top-k sigmoid(z_e.detach()) inside 1 - Dilate(Y, far_radius)
```

No offline pseudo labels, HC-Test, blind, or external data are used for the candidate audit or design decision.

## Gates

Proceed in order:

```text
Gate-D0: ERD-v2 failure audit complete.
Gate-D1: py_compile and pytest pass.
Gate-D2: ERD-v3 train split candidate audit passes.
Gate-D3: only then run ERD-v3 seed42.
```

Stop immediately if Gate-D1 or Gate-D2 fails. Do not run seed42 before `docs/internal/erd_v3_candidate_audit_train/gate_pass.json` reports `gate_pass=true`.
