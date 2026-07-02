# EACF Scale-Consensus Audit Protocol

Gate-F0 is an audit-only step. It checks whether frozen MSHNetOHEM scale heads
contain dense cross-scale signal before EACF-MSHNet is implemented or trained.

The audit uses train split only and must not use HC-Test, blind, or external
sets.

Minimum pass conditions:

```text
multi_scale_target_support_mean >= 0.90
far_bg_high_conf_scale_var_mean > 1e-5
single_scale_high_bg_ratio > 0
easy_bg_scale_var_mean must not exceed high-conf background scale variance
```

If Gate-F0 fails, EACF training is blocked and no seed42/43/44 training is
allowed.
