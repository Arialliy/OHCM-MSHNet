# EACF-MSHNet Design

EACF-MSHNet is the active AAAI architecture candidate after SPS, ERD, TCD, PFR, APF, and AEC hard-clutter routes were stopped.

## Core Constraint

EACF must preserve MSHNetOHEM as the evidence anchor:

```text
eta = 0 -> final_logit == base_logit
```

The fusion output is:

```text
consensus_logit = sum_s softmax(weight_s) * scale_logit_s
final_logit = base_logit + eta * (consensus_logit - base_logit)
```

The scale weights are convex:

```text
weight_s >= 0
sum_s weight_s = 1
```

## Stage-1 Rule

The first allowed training stage loads the paired MSHNetOHEM checkpoint, freezes the backbone and native scale heads, and trains only the EACF fusion module for 80 epochs on seed42.

## Gates

- Gate-F0: scale-consensus audit must pass before any EACF training.
- Gate-F1: code semantics must pass py_compile, pytest, and check_eacf_ready.py.
- Gate-F2: seed42 Full audit must show base approximately equals paired OHEM and final does not regress.

If any gate fails, stop EACF and do not run HC-Val, seed43/44, blind, or external evaluation.
