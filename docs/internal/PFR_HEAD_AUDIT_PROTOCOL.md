# PFR Evidence / Final / Residual Head Audit Protocol

## Purpose

This audit is failure analysis only. It must not trigger training, seed
expansion, HC-Test, blind, or external evaluation.

The audit answers whether the trained PFR checkpoint degraded the MSHNetOHEM
evidence branch itself, or whether the residual fusion head is the main source
of the Full split regression.

## Heads

```text
evidence: MSHNet evidence logits inside the PFR checkpoint
final: evidence logits plus bounded residual delta
residual: bounded residual delta only, not a segmentation probability
```

## Required Outputs

```text
evidence_mIoU
final_mIoU
evidence_Pd
final_Pd
evidence_Precision
final_Precision
evidence_FA_ppm
final_FA_ppm
evidence_FP_components
final_FP_components
residual_new_fp_pixels
residual_new_fp_components
residual_removed_fp_pixels
residual_removed_fp_components
residual_lost_target_pixels
residual_boundary_excess_pixels
delta_mean_target
delta_mean_boundary
delta_mean_far_bg
delta_positive_far_bg_ratio
```

## Decision Rules

If evidence-only is clearly below MSHNetOHEM, PFR training polluted the evidence
branch and all PFR-style trainable head routes stay stopped.

If evidence-only is close to MSHNetOHEM but final is worse, the residual fusion
design is the main failure mode. The current PFR residual head remains stopped.

If both evidence-only and final are close to MSHNetOHEM despite the Full Gate
failure, inspect evaluation/export parity before making any new method decision.
