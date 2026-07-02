# ERD-MSHNet v2 Design

ERD-MSHNet v2 treats the original MSHNet implementation in
`/home/ly/AAAI/MSHNet` as the upstream evidence-backbone reference. The current
design code is implemented only in `/home/ly/AAAI/OHCM-MSHNet-main`.

## Main Idea

MSHNet answers where target-like evidence appears. ERD-MSHNet v2 adds a
reliability head that estimates whether target-like evidence is trustworthy in
complex far-background regions.

The final prediction is suppress-only:

```text
R = sigmoid(z_r)
G = rho + (1 - rho) * R
z_f = z_e + gamma * log(G)
```

Because `G <= 1`, the reliability branch cannot create new target probability;
it can only suppress unreliable evidence.

## Supervision

Offline TCE/OHEM pseudo labels are stopped because negative reliability labels
were too sparse. ERD-MSHNet v2 instead uses online dense reliability supervision:

- positive gate pixels: GT target dilation;
- negative gate pixels: high-evidence far-background pixels selected online
  from the detached evidence logit;
- detection loss: OHEM-style MSHNet detection loss on final gated logits;
- auxiliary evidence loss: OHEM on evidence logits.

No memory bank, prototype queue, EMA teacher, TCE distillation, or region miner
is used in the first v2 design.

## Gate

Before any ERD training, run `tools/official/audit_online_gate_candidates.py`
on the train split only. Seed42 training is allowed only if the audit summary has
`gate_pass: true`.

No HC-Test, blind, or external split can be used for design-time gate decisions.
