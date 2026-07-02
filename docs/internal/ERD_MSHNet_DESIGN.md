# ERD-MSHNet Design Note

## Status

SPS-OHEM reranking is stopped as the main route after Gate0 diagnostics:

- TC-v2 pixel-level Gate0 failed.
- `region_component` Gate0 failed.
- `peak_region` Gate0 failed.

The project is now in architecture design stage. The current candidate method is
`ERD-MSHNet`: Evidence-Reliability Decoupled MSHNet.

## Core Idea

`MSHNetOHEM` is retained as the target evidence branch. ERD-MSHNet adds a
lightweight reliability branch that predicts whether target-like evidence is
trustworthy.

The final prediction is suppress-only:

```text
P_e = sigmoid(z_e)
R   = sigmoid(z_r)
G   = rho + (1 - rho) * R
z_f = z_e + gamma * log(G)
```

Because `G <= 1`, the reliability gate cannot create new positives; it can only
calibrate or suppress evidence.

## Training Supervision

Reliability pseudo labels must be generated only from the train split:

- reliable positives: target/dilated target pixels;
- unreliable target-like negatives: far-background pixels where OHEM is high but
  TCE support is low.

HC-Test, blind, and external splits must not be used to build reliability labels
or choose thresholds.

## Gates

Gate-A: architecture semantics

- `gamma=0` output equals evidence output.
- `sigmoid(final_logit) <= sigmoid(evidence_logit)`.
- forward/export expose `final_logit`, `evidence_logit`, `reliability_logit`,
  `feature`, and multi-scale masks.
- reliability loss ignores invalid pixels.

Gate-B: pseudo-label audit

- target leakage is zero by construction;
- positive and negative reliability labels are non-empty;
- labels come from train split only.

Gate-C and later training gates are blocked until Gate-A and Gate-B pass.

## Prohibited Work

- Do not continue SPS alpha/lambda/tau/topk sweeps.
- Do not convert `region_component` or `peak_region` into training loss.
- Do not use HC-Test/blind/external labels for reliability supervision.
- Do not train seed42 before reliability label audit passes.
