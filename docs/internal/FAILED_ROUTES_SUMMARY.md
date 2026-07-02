# Failed Routes Summary

The following routes are stopped for AAAI method decisions and must not be resumed as main methods:

| Route | Stop Reason |
|---|---|
| SPS / TC-v2 / region / peak | Gate0 candidate diagnostics failed; selected/OHEM overlap was high and candidates were sparse. |
| ERD / ERD-v3 | Trainable reliability and suppression heads did not pass Full/HC-Val gates. |
| TCD / TCE | Teacher/student soft-label information was insufficient. |
| PFR | Full Gate failed and head audit showed evidence-branch pollution. |
| APF | Gate-A candidate audit failed due to flat-background candidates. |
| AEC / hard-clutter component mining | Gate-ECA found too few detached far-FP components in train split. |

Current allowed route:

```text
EACF-MSHNet only, under Gate-F0/F1/F2 ordering.
```

Current prohibitions:

```text
Do not run seed43/44.
Do not tune beta/lambda/topk/threshold for failed routes.
Do not enter HC-Test, blind, or external evaluation before the relevant gate passes.
```
