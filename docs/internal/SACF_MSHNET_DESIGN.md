# SACF-MSHNet Design

SACF-MSHNet is the next allowed structure after EACF-v1 identity collapse.

It removes the global eta path and uses local scale-agreement fusion:

```text
W_s(x) = softmax(g(P, stats))
G(x) = sigmoid(h(P, stats))
z_csf(x) = sum_s W_s(x) * U_s(x)
delta(x) = clip(z_csf(x) - z_base(x), -delta_max, +delta_max)
z_final(x) = z_base(x) + G(x) * delta(x)
```

Gate-S2a must prove the fusion branch activates before any 80-epoch run.
