# PFR-MSHNet Design

## Goal

PFR-MSHNet is the fallback after ERD-v3 and TCD Gate-T2R failures. It keeps
MSHNetOHEM as the evidence anchor and learns a bounded residual correction only
under explicit target-protection and far-background constraints.

## Forward

```text
z_e = MSHNetOHEM(I)
delta = beta * tanh(g(feature))
z_final = z_e + delta
```

The residual head is zero-initialized, so the initial model is identical to the
evidence branch.

## Training Constraints

- Far-background hard negatives are selected only outside target dilation.
- Negative residual on GT target pixels is penalized.
- Negative residual on near-target boundary support is penalized.
- Residual magnitude is regularized.
- Candidate audit must pass before any training.

## Stop Rules

- If the train-only PFR candidate audit fails, stop and do not train.
- If seed42 Full gate fails, stop and do not evaluate HC-Val.
- Do not run seed43/44 until seed42 and mechanism controls pass.
