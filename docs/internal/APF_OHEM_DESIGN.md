# APF-OHEM Design

APF-OHEM means Anchor-Preserved Far-background Online Hard Example Mining.

The route exists because PFR head decomposition showed that end-to-end
trainable correction heads can pollute the MSHNetOHEM evidence branch.

## Current Scope

The current stage is candidate audit only.

Allowed:

```text
build frozen MSHNetOHEM train anchor maps
audit APF far-background candidate masks
run check_apf_ready.py
```

Blocked:

```text
APF-OHEM training
seed42 APF training
seed43/44
HC-Test / blind / external
loss lambda tuning
new residual/gate/fusion heads
```

## Design Constraint

APF-OHEM must preserve the MSHNetOHEM inference graph. Any future training
implementation must start from the frozen OHEM anchor and must not add a new
inference head.
