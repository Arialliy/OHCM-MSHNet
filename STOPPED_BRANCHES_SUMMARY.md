# Stopped Diagnostic Branches

Current final decision:

```text
READ_ONLY_FAILURE_ARCHIVE_STATE.
No active AAAI main-method branch remains.
TCE-4, TCSR, TWA, late-snapshot, and post-hoc checkpoint-selection final routes are stopped.
```

Stopped branches:

| Branch | Status |
|---|---|
| PFRMSHNet | Stopped after Full gate failure. |
| ERDMSHNet | Stopped after HC-Val / Full reliability failure. |
| ERDMSHNetV3 | Stopped after Full gate failure. |
| CDVMSHNet | Stopped after Gate-B flat-artifact failure. |
| ECDVMSHNet | Stopped after Gate-B flat-artifact failure. |
| MSCVMSHNet | Stopped after Gate-B candidate / target-top20 failure. |
| BCVMSHNet | Stopped after Gate-D2: residual/shape suppressibility insufficient. |
| OHCMMSHNetFull | Stopped after full/prototype branch failure. |

Training guard:

```bash
python train.py --model_names BCVMSHNet ...
```

is blocked by default. Use `--allow_stopped_branch` only for diagnostic reproduction.

## TWA-4 without BN recalibration

Status: stopped at Gate-TWA-E.

Reason:

- TWA-4 passed OHEM comparison and TCE-retention checks,
  but failed the best-single-late mechanism check.
- Best single late checkpoint `ep250` dominated TWA-4 on seed42 HC-Val.

Decision:

- Do not run seed43/44 for TWA-4.
- Do not run HC-Test, blind, or external for TWA-4.
- Do not tune BN recalibration.
- Do not search new TWA checkpoint combinations.
- `ep250` may be evaluated only through the separate `LateSnapshot-ep250` Gate-LS-A protocol.

## LateSnapshot-ep250 stopped at Gate-LS-A

Decision: `STOP_LATE_SNAPSHOT_EP250_AT_GATE_A`.

Reason:
- ep250 is very strong on seed42 HC-Val.
- However, ep250 fails seed42 Full non-regression vs OHEM-400.

Metrics:
- Full delta mIoU: -0.003254
- Full delta Precision: -0.000187
- Full delta FA ppm: -0.114901
- Full delta Pd: +0.000000
- HC-Val delta mIoU vs OHEM: +0.105858
- HC-Val delta FA ppm vs OHEM: -175.476

Interpretation:
- ep250 is retained as a stopped hard-clutter diagnostic control.
- ep250 must not be advanced to seed43/44, HC-Test, blind, or external.
- The next allowed audit is Gate-TWA-E2-FSC, which compares TWA-4 with Full-safe single-late controls only.

## Post-hoc checkpoint / seed selection stopped

Decision: STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD.

Reason:
- Gate-TWA-E2-FSC selected LateSnapshot-ep300, but its HC-Val mIoU advantage over TWA-4 is only +0.000204.
- The selected candidate is not a new architecture, loss, or inference mechanism.
- Choosing a seed / epoch / checkpoint after validation is a selection procedure, not a robust method.

Retained as diagnostics:
- ep250: hard-clutter specialist but Full unsafe.
- ep300: Full-safe late snapshot diagnostic.
- TWA-4: trajectory-averaging diagnostic.
- TCE-4: trajectory-consensus oracle.

Next structural route:
- TCSR-MSHNet, starting with Gate-TCSR-A bank audit.

## TCSR-v1 stopped at bank audit

Decision: STOP_TCSR_AT_BANK_AUDIT.

Reason:
- Gate-TCSR-A found only 1 / 697 train images with sparse negative reliability pixels.
- Total negative pixels = 130, below the fixed minimum requirement of 500.
- Target leakage and protect overlap were zero, so this is not a safety failure or code-alignment failure.
- The train-only sparse hard-clutter negative signal required by TCSR-v1 is absent under the frozen definition.

Forbidden:
- Do not run TCSR activation sanity.
- Do not add TCSR loss/net/train changes.
- Do not tune bank thresholds or lambdas to rescue the route.
- Do not run seed43/44, HC-Test, blind, or external for TCSR.

Next route:
- Freeze TCE-4-OHEM trajectory-consensus inference as the final AAAI candidate and aggregate already validated internal evidence.

## TCE-4 stopped at F3 external Pd regression

Decision: STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION.

Gate:
- Gate-TCE-F3-blind-external-once.

Once-lock:
- `gate_tce_f3_once_lock.json` status: STOPPED_BY_F3_PD_REGRESSION.

Fail summary:
- `gate_tce_f3_fail_summary.json` decision: F3_FAIL_NO_REDESIGN.

Status:
- Gate-TCE-F0 freeze consistency: PASS.
- Gate-TCE-F1 internal evidence aggregation: PASS.
- Gate-TCE-F2 threshold/component report: PASS.
- Gate-TCE-F3 blind/external once: FAIL / stopped.

Frozen method:
- Candidate: TCE-4-OHEM.
- Baseline: MSHNetOHEM-400.
- Checkpoints: 250, 300, 350, 400.
- Threshold: fixed 0.5.
- Runtime: 4x checkpoint forwards.

F3 stop evidence:
- `external_nuaa_sirst`: min delta Pd = -0.018348624.
- `external_irstd_1k`: min delta Pd = -0.013468013.

Not completed:
- `external_sirst3` was stopped due to manifest integrity failure:
  - `test_SIRST3.txt` entries: 1079.
  - missing masks: 365.
  - missing images: 1.

Interpretation:
- TCE-4 remains useful as internal trajectory-consensus diagnostic evidence.
- It is not a final AAAI main method under the frozen F3 rule because external Pd regresses.
- F3 is a frozen-method once gate, so no rescue is allowed.

Forbidden:
- Do not rerun blind/external after seeing F3 results.
- Do not change threshold, seed, checkpoint, split, model, loss, train code, or dataset code to rescue TCE-4.
- Do not replace SIRST3 with a labeled subset as a rescue path.
- Do not restart TCSR, TWA, or post-hoc checkpoint selection as final routes.
- Do not use SIRST3 mask imputation.
- Do not add a new verifier / suppression head.
- Do not run TCSR Stage 2.

Allowed:
- final stop-state checker.
- archive / failure-analysis report.
- repository freeze.
