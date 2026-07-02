# CGA-MSHNet Design

CGA-MSHNet is the active AAAI redesign after SPS, APF/AEC, ERD/PFR/TCD,
EACF, and SACF were stopped by gate audits.

## Scope

CGA keeps the original MSHNet final fused mask as the inference decision path.
It does not add a post-hoc residual, reliability, clutter, suppression, or
calibration head.

## Added Training Signals

- Component center heatmap supervision from GT connected components.
- Component scale-bin supervision on target pixels.
- Core and boundary auxiliary supervision.
- Local background peak suppression as a dense false-alarm proxy.
- Easy-region anchoring against the OHEM evidence path.

## Gate Order

1. Gate-C0: failed routes are documented and guarded.
2. Gate-C1: component geometry target audit passes before training.
3. Gate-C2: compile and CGA semantic tests pass.
4. Gate-C3: seed42 1-epoch `decoder_aux` activation sanity passes.
5. Gate-C4: seed42 80-epoch Full and HC-Val sanity is allowed only after C3.

If any gate fails, stop at that gate and do not run later gates.
