# ERD-MSHNet / SPS-OHEM MSHNet

Research code for infrared small target detection (IRSTD) experiments built on
BasicIRSTD and MSHNet. The repository contains the training, evaluation, export,
and analysis pipeline for MSHNetOHEM, the stopped SPS-OHEM reranking route, and
the current ERD-MSHNet architecture design route.

## Current Official Status

Strong anchor: MSHNetOHEM.

Final development decision:

- STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION.
- STOP_TCSR_AT_BANK_AUDIT.
- STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD.
- STOP_TWA_NO_BN_AT_GATE_E.
- STOP_LATE_SNAPSHOT_EP250_AT_GATE_A.

No active AAAI main-method branch remains.

Latest final gate:

- Gate-TCE-F3-blind-external-once was started after F0/F1/F2 passed.
- Preflight passed and once-lock was created.
- F3 stopped because completed external splits showed Pd regression:
  - `external_nuaa_sirst`: min_delta_Pd = -0.018348624
  - `external_irstd_1k`: min_delta_Pd = -0.013468013
- `external_sirst3` was not completed due to manifest integrity failure:
  - total entries = 1079
  - missing masks = 365
  - missing images = 1

Decision:

- F3_FAIL_NO_REDESIGN.
- Do not generate an F3 final report.
- Do not rerun blind/external.
- Do not modify external split definitions.
- Do not use SIRST3 mask imputation or labeled-subset rescue.

Allowed next work:

- final stop-state consistency checker
- stopped-branch archive
- failure-analysis table
- repository freeze

Stopped / diagnostic branches:

- TWA with BN recalibration: stopped.
- TWA-4 no-BN: diagnostic only; not promoted as final single-forward method.
- LateSnapshot-ep250: stopped at Gate-LS-A because Full split is unsafe.
- LateSnapshot-ep300: diagnostic only; not promoted because its HC-Val advantage over TWA-4 is a numerical tie / post-hoc checkpoint effect.
- TCSR-v1: stopped at Gate-TCSR-A bank audit because the train-only sparse hard-clutter negative bank is too sparse.
- Post-hoc seed / checkpoint / epoch selection: stopped as AAAI main method.
- TCE-4-OHEM: stopped at F3 external Pd regression; retained only as internal diagnostic evidence.

TCE-4 frozen method:

- TCE-4-OHEM trajectory-consensus inference.
- Base model: MSHNetOHEM.
- Checkpoints: ep250 / ep300 / ep350 / ep400.
- Inference: 4 forwards, existing official TCE aggregation, fixed threshold 0.5.
- Training: no new training.
- Seeds for final paired reporting: 42 / 43 / 44.

Internal evidence:

- Gate-TCE-F0 freeze consistency: PASS.
- Gate-TCE-F1 internal evidence aggregation: PASS.
- Gate-TCE-F2 threshold/component report: PASS.

F3 status:

- F3 once-lock was opened and is now closed as `STOPPED_BY_F3_PD_REGRESSION`.
- Stop summary: `docs/internal/tce_final/gate_tce_f3_fail_summary.json`.
- Final stop-state summary: `docs/internal/final_stop_state_summary.json`.

Forbidden:

- new training
- new evaluation
- rerunning blind/external after seeing results
- external split redefinition
- SIRST3 mask imputation
- SIRST3 labeled subset replacement
- seed search
- checkpoint search
- threshold search
- BN recalibration tuning
- TCSR training
- new model / loss / verifier / suppression structure
- changes to loss.py / net.py / train.py / dataset.py / model/

## Archived Research Status

Historical decision after BCV Gate-D2:

```text
Stop single-frame false-alarm suppression branches.
Do not train verifier / candidate-mining / residual-shape BCV variants.
Do not run HC-Val, seed43/44, HC-Test, blind, or external for those stopped branches.
This route was later stopped; no active AAAI main-method branch remains.
```

BCV status:

- BCV-A beta=0 parity: PASS.
- BCV-B background residual separation: PASS.
- BCV-C residual magnitude OHEM-FP audit: FAIL for safe component suppressibility.
- BCV-D residual shape audit: FAIL for safe component suppressibility.
- BCV-D2 mass-weighted residual/shape audit: FAIL for FP pixel/confidence mass suppressibility.

The previous SPS/ERD/TCD/PFR/APF/AEC routes are stopped for AAAI design decisions.
EACF-v1 and SACF-v1 are also stopped because they collapsed to the MSHNetOHEM identity path.

Reason:
- SPS-style candidate reranking failed Gate0 diagnostics.
- ERD/PFR-style trainable heads polluted the MSHNetOHEM evidence branch.
- APF/AEC hard-clutter mining is not supported by enough detached far-FP components in the train split.
- EACF/SACF preserved MSHNetOHEM but did not activate useful prediction changes.

The previously considered architecture direction was:

```text
CGA-MSHNet:
Component-Geometry Aligned MSHNet
```

CGA kept the MSHNet final mask inference path and added component-center, scale,
boundary, and local-peak auxiliary supervision. It is archived as design/audit
context only; no training or new architecture development is active.

## EACF-MSHNet Status

EACF-v1 is stopped after Gate-F3.

Result:
- Full: base and final are identical, eta = 0.0.
- HC-Val: base and final are identical, eta = 0.0.

Decision:
- EACF-v1 collapsed to the MSHNetOHEM identity path.
- Do not run seed43/44.
- Do not tune eta/lambda/hidden/lr/epoch.
- Do not use HC-Test, blind, or external evaluation.
- Next allowed work: SACF-MSHNet activation sanity and structure redesign.

## SACF-MSHNet Status

SACF-v1 is stopped after Gate-S2a.

Result:
- 1 epoch activation sanity failed.
- mean_abs_final_minus_base_prob = 1.020973338e-06.
- Required threshold was > 1e-4.
- fail_reasons = ["final_equals_base_identity_collapse"].

Decision:
- SACF-v1 collapsed to the MSHNetOHEM identity path.
- Do not run SACF 80 epoch.
- Do not tune SACF gate/lambda/lr.
- Do not use HC-Test, blind, or external evaluation.
- Next allowed work: CGA-MSHNet component target audit and activation sanity.

PFR-MSHNet seed42 Full Gate failed and is stopped.

Checkpoint:

```text
/home/ly/AAAI/OHCM-MSHNet/results/official/PFRMSHNet/seed42/NUDT-SIRST/PFRMSHNet_400.pth.tar
```

Full split result:

| Metric | PFR | Required / OHEM reference | Status |
|---|---:|---:|---|
| mIoU | 0.780196 | >= 0.833393 | FAIL |
| Pd | 0.986243 | >= 0.979894 | PASS |
| Precision | 0.865911 | >= 0.906277 | FAIL |
| FA ppm | 89.416 | <= 63.449 | FAIL |
| FP components | 114 | OHEM = 47 | FAIL |

Failure audit:

```text
total_target_lost_count = 2
total_boundary_excess_delta = 652
failure_mode = structural_suppression_or_calibration_regression
```

Decision:

```text
PFR-MSHNet is stopped.
Do not run seed43/44.
Do not tune beta/lambda/topk/threshold.
Do not use HC-Test, blind, or external evaluation for PFR decisions.
PFR is retained only for failure analysis.
```

PFR head decomposition audit:

| Metric | Evidence-only | Final |
|---|---:|---:|
| mIoU | 0.778985 | 0.780196 |
| Pd | 0.985185 | 0.986243 |
| Precision | 0.872860 | 0.865911 |
| FA ppm | 83.280 | 89.416 |
| FP components | 111 | 114 |

Residual audit:

```text
residual_new_fp_components = 234
residual_removed_fp_components = 0
residual_boundary_excess_pixels = 158
delta_positive_far_bg_ratio = 0.576573
```

Decision from head audit:

```text
PFR training polluted the MSHNetOHEM evidence branch.
All PFR-style end-to-end trainable head routes are stopped.
Do not add PFR-v2 / PFR-v3 trainable correction heads.
Any future method must preserve the MSHNetOHEM inference anchor.
Next allowed work is APF-OHEM candidate audit only; APF training is blocked
until the candidate audit passes.
```

## APF-OHEM Status

APF-OHEM is currently stopped at Gate-A candidate audit.

Anchor maps:

- num_images = 697
- num_written = 697
- gate_pass = true

Candidate audit:

- gate_pass = false
- candidate_to_budget_ratio_mean = 1.0
- flat_bg_ratio_mean = 0.9993765
- target_leakage_pixels_total = 0
- ohem_fp_component_coverage_mean = 0.5

Decision:

- Do not run APF seed42.
- Do not tune APF candidate parameters or loss lambdas for training.
- Do not run seed43/44, HC-Test, blind, or external.
- APF is retained only for candidate-mechanism audit.
- Next allowed work is OHEM error-component audit.

## OHEM Error-Component Audit Status

OHEM error-component mining is stopped after Gate-ECA.

Build result:

- num_images = 697
- num_written = 697
- component_count_total = 1013
- target_hit_components = 978
- target_leakage_pixels_total = 0

Gate-ECA audit:

- gate_pass = false
- total_detached_far_fp_components = 28
- images_with_detached_far_fp_ratio = 0.004304
- nonflat_detached_far_fp_ratio = 1.0
- target_like_area_detached_far_fp_ratio = 0.464286
- mean_detached_far_fp_peak_prob = 0.893753
- train_candidate_to_budget_ratio_mean = 0.000454
- flat_component_ratio = 0.0

Decision:

- Stop APF-OHEM / APF-v2 / AEC-OHEM / component-mining OHEM routes.
- Do not run seed42, seed43/44, HC-Test, blind, or external for these routes.
- Do not tune APF thresholds or loss lambdas to force a pass.
- Keep MSHNetOHEM as the strong baseline anchor and retain diagnostics only.

SPS HOLD: SPS-OHEM reranking is currently stopped after Gate0 diagnostics:

- pixel-level target-contrast Gate0 failed;
- `region_component` Gate0 failed;
- `peak_region` Gate0 failed.

Offline TCE/OHEM reliability pseudo labels are also stopped because negative
reliability labels are too sparse.

The current allowed next step is APF-OHEM candidate audit only. APF-OHEM must
keep the MSHNetOHEM inference graph unchanged and must first prove that its
train-only far-background candidates are target-safe. Do not run APF-OHEM
training until the candidate audit and `check_apf_ready.py` both pass.

## Current ERD Gate Status

ERD-MSHNet v2 online gate has completed seed42 training.

Full split passed:

- Delta mIoU +0.0498
- Delta Pd +0.0011
- Delta Precision +0.0316
- Delta FA -20.93 ppm
- Delta FP components -10

HC-Val failed:

- Delta mIoU -0.0360
- Delta Pd +0.0000
- Delta Precision +0.0015
- Delta FA -35.60 ppm
- Delta FP components -1

Decision:

- ERD-v2 is stopped.
- Do not run seed43/44 for ERD-v2.
- Do not use HC-Test, blind, or external evaluation for ERD-v2 design decisions.
- Next allowed work is ERD-v2 failure audit and ERD-v3 target-preserving clutter suppression design.

## Current Gate Status: ERDMSHNetV3 stopped

ERDMSHNetV3 seed42 was trained to epoch 400 but failed the Full split gate.

Checkpoint:
`results/official/ERDMSHNetV3/seed42/NUDT-SIRST/ERDMSHNetV3_400.pth.tar`

Full metrics:

- mIoU: 0.808049
- Pd: 0.984127
- Precision: 0.889289
- FA: 72.778 ppm
- FP components: 38

Compared with the MSHNetOHEM strong baseline, ERDMSHNetV3 degrades mIoU,
Precision, and FA on the Full split. Therefore ERD-v2/v3 suppression-style
reliability-gating is stopped.

Do not run seed43/seed44, HC-Test, blind, or external evaluation for
ERDMSHNetV3.

The next allowed work is:

1. ERD-v3 failure-mode audit;
2. TCE dense soft-label audit;
3. a new TCE-guided residual calibration design that keeps MSHNetOHEM as the
   protected evidence anchor.

## Current TCD Gate Status

TCD-MSHNet dense TCE distillation is currently stopped.

Generated train-only TCE soft labels:

- path: `docs/internal/tce_soft_labels/seed42_train`
- count: `697 / 697`

Gate-T2 failed:

- `teacher_student_absdiff_mean = 1.637e-05`
- gate requirement: `> 0.001`

Decision:

- Do not run TCD training.
- Do not run seed42 / seed43 / seed44 TCD.
- Do not use HC-Test, blind, or external sets.
- Next allowed step: Gate-T2R teacher-information root-cause audit.

## Project Scope

This codebase is centered on three questions:

1. How strong is MSHNet under the BasicIRSTD training and evaluation protocol?
2. Can hard-negative mining reduce false alarms in hard-clutter IRSTD scenes?
3. Can self-perturbation stability improve OHEM-style mining while preserving
   single-model, single-forward inference?

Main variants kept in the official path include:

- `MSHNet`: MSHNet adapted to single-channel IRSTD input.
- `MSHNetFocal`: MSHNet with focal-loss variant.
- `MSHNetOHEM`: MSHNet with online hard-example mining.
- `MSHNetTopKNeg`: MSHNet with top-k negative mining.
- `MSHNetSPSOHEM`: SPS/OHEM variant using perturbation stability for
  fixed-budget hard-negative reranking.

OHCM/prototype, PCAR, TSR, and old step-gate exploration artifacts are retained
only under `legacy/`, `tools/legacy/`, and `docs/internal/`.

## Repository Layout

```text
.
|-- configs/                         # Experiment configs and frozen settings
|-- model/                           # MSHNet, baselines, and retained legacy models
|-- tools/official/                  # Official OHEM/SPS/eval entry points
|-- tools/legacy/                    # Historical PCAR/OHCM/TSR/step diagnostics
|-- docs/internal/                   # Internal design notes and gate records
|-- legacy/                          # Old launch scripts and legacy configs
|-- utils/                           # Metrics and visualization helpers
|-- dataset.py                       # Dataset loaders
|-- train.py                         # Main training entry point
|-- test.py                          # Legacy test entry point
|-- evaluate.py                      # Legacy mask-result evaluator
|-- net.py                           # Model registry and loss dispatch
|-- loss.py                          # SLS, OHEM, SPS-OHEM, and retained legacy losses
|-- probability.py                   # Unified foreground-probability conversion
|-- requirements.txt                 # Core Python dependencies
```

Large runtime artifacts are intentionally not tracked by Git:

```text
datasets/
results/
checkpoints/
repro_smoke/
*.pth
*.pt
*.ckpt
```

## Data

Place datasets under `./datasets/`. The local workspace uses this layout:

```text
datasets/
|-- NUAA-SIRST/
|   |-- images/
|   |-- masks/
|   `-- img_idx/
|-- NUDT-SIRST/
|   |-- images/
|   |-- masks/
|   `-- img_idx/
|-- IRSTD-1K/
|   |-- images/
|   |-- masks/
|   `-- img_idx/
`-- SIRST3/
    |-- images/
    |-- masks/
    |-- masks_centroid/
    |-- masks_coarse/
    `-- img_idx/
```

The expected split files live in each dataset's `img_idx/` directory, following
the BasicIRSTD naming convention, for example:

```text
train_NUDT-SIRST.txt
test_NUDT-SIRST.txt
```

Dataset images and masks are not included in this repository. Download them from
their official project pages or dataset providers, then place them in the layout
above.

## Environment

The code is PyTorch-based. The project has been run in a Docker/GPU environment
with Python 3, PyTorch, NumPy, PIL/Pillow, tqdm, scikit-image, and matplotlib.

Install the core Python packages in your own environment:

```bash
pip install -r requirements.txt
```

For ISNet/DCNv2 baselines only, compile DCNv2:

```bash
cd model/ISNet/DCNv2
bash make.sh
```

MSHNet, OHEM, and SPS-OHEM do not require compiling DCNv2.

## Current Gate Status

SPS-OHEM training is currently on hold. The updated Gate0 roadmap records that
the current pixel-level SPS candidate rule failed mechanism diagnostics:

```text
target top20 > 0.15
selected/OHEM overlap = 1.0
```

Until a new Gate0 census passes, do not start SPS seed42/43/44 training or use
HC-Test, blind, or external evaluation for SPS decisions. The only allowed SPS
work in this state is Gate0 diagnostic logging and candidate-rule census under
`tools/official/sps_perturbation_census.py`.

## Training

Basic MSHNet training:

```bash
python train.py \
  --model_names MSHNet \
  --dataset_names NUDT-SIRST \
  --dataset_dir ./datasets \
  --batchSize 4 \
  --patchSize 256 \
  --nEpochs 400 \
  --optimizer_name Adagrad \
  --learning_rate 0.05 \
  --mshnet_warm_epoch 5 \
  --mshnet_in_channels 1 \
  --save ./log
```

MSHNet + OHEM:

```bash
python train.py \
  --model_names MSHNetOHEM \
  --dataset_names NUDT-SIRST \
  --dataset_dir ./datasets \
  --batchSize 4 \
  --patchSize 256 \
  --nEpochs 400 \
  --optimizer_name Adagrad \
  --learning_rate 0.05 \
  --lambda_variant 0.2 \
  --ohem_ratio 0.01 \
  --mshnet_warm_epoch 5 \
  --mshnet_in_channels 1 \
  --save ./log
```

SPS-OHEM reranking remains in the codebase for diagnostics, but training runs
are blocked until Gate0 passes with a candidate rule that is target-safe and not
OHEM-overlapping.

Legacy OHCM/prototype launch material is kept under `legacy/` and
`docs/internal/`; it is not part of the clean SPS restart path.

## Export And Evaluation

Export probability maps, logits, masks, and visualizations:

```bash
python tools/official/export_step0_predictions.py \
  --dataset_dir ./datasets \
  --dataset_name NUDT-SIRST \
  --model_name MSHNetOHEM \
  --checkpoint ./log/NUDT-SIRST/MSHNetOHEM_400.pth.tar \
  --output_dir ./results/exports/NUDT-SIRST/MSHNetOHEM_seed42 \
  --threshold 0.5
```

By default, export runs a direct/export parity gate:

```text
tools/official/check_direct_export_parity.py
```

This gate checks that direct checkpoint inference and exported probability maps
agree on probability values, masks, mIoU, Pd, and FA. Evaluation refuses to run
unless the parity gate exists and passes, unless explicitly skipped.

In this repository, `Pd` denotes target-level detection probability computed by
connected-component matching. Legacy `BinaryMetricsGPU` pixel recall is reported
as `PixelRecall`, not `Pd`.

Evaluate exported probabilities:

```bash
python tools/official/evaluate_prediction_exports.py \
  --dataset_dir ./datasets \
  --dataset_name NUDT-SIRST \
  --exports_dir ./results/exports/NUDT-SIRST/MSHNetOHEM_seed42 \
  --output_dir ./results/eval/NUDT-SIRST/MSHNetOHEM_seed42 \
  --threshold 0.5
```

## Result Summary

The full `results/` directory is not included in Git because it contains
checkpoints, dense probability maps, visualizations, and per-image exports. The
tables below copy the key scalar results from local result artifacts so that the
repository still documents the experimental state.

### Dataset Baseline

MSHNet baseline, fixed threshold `0.5`.

Source:

```text
results/step0_baseline/20260611_155232/step0_mean_std.csv
```

| Dataset | Seeds | mIoU | Pd | FA ppm | Precision | F1 | FP comp. |
|---|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | 1 | 0.588606 | 0.858586 | 45.738 | 0.805504 | 0.741035 | 36.0 |
| NUAA-SIRST | 1 | 0.681373 | 0.935780 | 70.992 | 0.863919 | 0.810496 | 7.0 |
| NUDT-SIRST | 3 | 0.806636 +/- 0.036913 | 0.976720 +/- 0.005892 | 70.909 +/- 10.202 | 0.891218 +/- 0.017109 | 0.892659 +/- 0.022816 | 56.33 +/- 12.90 |

### Legacy OHCM-Light Baseline

The frozen OHCM-light checkpoint is retained as an internal baseline. Its legacy
configuration now lives at `legacy/configs/ohcm/OHCM-light.yaml`.

| Split | mIoU | Pd | FA ppm | Precision | F1 | FP comp. |
|---|---:|---:|---:|---:|---:|---:|
| NUDT-SIRST Full | 0.794235 | 0.967196 | 81.166 | 0.877472 | 0.885319 | 42.0 |
| NUDT-SIRST HC-Set | 0.608656 | 0.823529 | 211.828 | 0.740944 | 0.756726 | 12.0 |

### Current SPS-OHEM Dev Candidate

Current best SPS-OHEM development candidate:

```text
sps_start_epoch = 0
learning_rate = 0.001
sps_lambda = 0.15
epoch = 40
sps_candidate_topk_ratio = 0.02
sps_target_safe = true
```

Source:

```text
results/sps_ohem/20260626_sps_topk02_tsafe_start0_lr001_e40_3seed_dev/
```

Fixed-threshold `0.5` gate against paired `MSHNetOHEM`:

| Seed | Split | OHEM mIoU | SPS mIoU | OHEM Pd | SPS Pd | OHEM FA ppm | SPS FA ppm | Delta mIoU | Delta FA ppm | Gate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | Full | 0.834393 | 0.840023 | 0.979894 | 0.982011 | 61.449 | 57.496 | +0.005631 | -3.953 | PASS |
| 42 | HC-Val | 0.604790 | 0.638655 | 0.833333 | 0.833333 | 386.556 | 322.978 | +0.033865 | -63.578 | PASS |
| 43 | Full | 0.818573 | 0.825547 | 0.976720 | 0.976720 | 70.917 | 63.011 | +0.006974 | -7.905 | PASS |
| 43 | HC-Val | 0.487805 | 0.611336 | 0.833333 | 0.833333 | 676.473 | 368.754 | +0.123531 | -307.719 | PASS |
| 44 | Full | 0.842908 | 0.851795 | 0.976720 | 0.977778 | 54.555 | 52.188 | +0.008887 | -2.367 | PASS |
| 44 | HC-Val | 0.649237 | 0.691781 | 0.833333 | 0.833333 | 279.744 | 226.339 | +0.042543 | -53.406 | PASS |

Mean deltas from the same run:

| Split | Delta mIoU | Delta Pd | Delta targets | Delta FA ppm | Delta Precision |
|---|---:|---:|---:|---:|---:|
| Full | +0.007164 | +0.001058 | +1.000 | -4.742 | +0.006682 |
| HC-Val | +0.066647 | +0.000000 | 0.000 | -141.568 | +0.075850 |

FP census at threshold `0.5`:

| Split | Result |
|---|---|
| HC-Val far-FP components | decreased for 3/3 seeds: -4, -3, -4 |
| HC-Val far-FP pixels | decreased for 3/3 seeds: -23, -121, -25 |
| Full far-FP components | decreased for 2/3 seeds: -10, -2, +1 |

Current status:

```text
Decision: HOLD_SPS_E40_CANDIDATE
Fixed 0.5 gate: PASS for 3/3 Full and 3/3 HC-Val.
AAAI-ready: not yet.
Reason: threshold-matched evidence is partial; ablations and final blind/external
evaluation are still missing. HC-Test remains sealed.
```

### Target-Margin SPS Screen

Target-margin SPS improved seed42 metrics but did not beat the mechanism
controls on HC-Val, so it is not promoted.

Source:

```text
results/sps_ohem/20260627_target_margin_seed42_combined_summary.csv
```

| Run | Split | mIoU | Pd | FA ppm | Precision | Delta mIoU | Delta FA ppm | Delta Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TM-A015-q85 | Full | 0.839574 | 0.980952 | 57.519 | 0.911795 | +0.005181 | -3.930 | +0.005517 |
| TM-A015-q85 | HC-Val | 0.637895 | 0.833333 | 320.435 | 0.706294 | +0.033104 | -66.121 | +0.040360 |

Mechanism comparison on seed42 HC-Val:

| Method | mIoU | FA ppm | Precision |
|---|---:|---:|---:|
| two-view OHEM | 0.639241 | 317.891 | 0.707944 |
| SPS no-far-mask | 0.640592 | 315.348 | 0.709602 |
| best target-margin SPS | 0.637895 | 320.435 | 0.706294 |

Current status:

```text
Decision: HOLD_TARGET_MARGIN_SPS
AAAI submit: false.
Reason: target-margin fixes candidate distinctness and seed42 performance is
valid, but it does not beat two-view OHEM or no-far-mask on seed42 HC-Val.
```

## Important Implementation Notes

Foreground probability is centralized in `probability.py`:

- `B x 1 x H x W` logits use `sigmoid`.
- `B x 2 x H x W` logits use `softmax(..., dim=1)[:, 1:2]`.

MSHNet warm-up behavior is handled in `net.py`:

- training uses `warm_flag = epoch > mshnet_warm_epoch`;
- testing and export force `warm_flag=True` to use the final fused head.

SPS-OHEM reranking keeps the OHEM negative budget fixed:

- OHEM selects `floor(num_background_pixels * ohem_ratio)` negatives;
- SPS changes the ranking score, not the number of selected negatives;
- rerank mode replaces the OHEM variant loss with the SPS-ranked OHEM loss.
- strict fallback is enabled by default, so empty SPS candidate pools fall back
  to a metric-ranked shortlist rather than treating all background pixels as
  SPS candidates;
- `--sps_candidate_min_metric` can require top-ratio candidates to have a
  positive candidate score; use it only after confirming the setting is not
  fallback-dominated on training-crop census;
- `--sps_no_two_view_base` disables two-view averaging in the base SLS/OHEM
  loss. SPS rerank still uses the perturbed view to compute instability for
  negative ranking, while positive pixels use the weak-view loss.

`train.py` saves checkpoints during training but does not run the legacy test
set evaluation unless `--eval_during_train` is explicitly passed. Formal Pd,
FA, threshold curves, and component FP census should come from the
`tools/official/` evaluation scripts.

`tools/official/evaluate_prediction_exports.py` writes both aggregate metrics and
component-aware FP analysis:

- `fp_components.csv`;
- `fp_census_at_threshold` in `summary_metrics.json`;
- boundary excess, detached near-FP, far-FP, and matched target components.

## Experiment Notes

Project notes and audit records are kept in:

```text
docs/internal/ohcm/
docs/internal/sps/
docs/internal/gates/
```

These files document the current experimental status, known risks, and the
AAAI-oriented validation gates. They are notes for reproducibility and internal
review, not final paper claims.

## Citation

If you use this workspace, cite the original datasets and the original MSHNet /
BasicIRSTD sources as appropriate. Add the final project citation here after the
paper metadata is fixed.
