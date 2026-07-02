#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Block SACF training unless activation sanity passed.")
    parser.add_argument("--activation_summary", required=True)
    args = parser.parse_args()

    errors = []
    path = Path(args.activation_summary)
    if not path.exists():
        errors.append("missing_activation_summary")
        summary = {}
    else:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("gate_pass", False):
            errors.append("activation_audit_failed")
        if float(summary.get("mean_abs_final_minus_base_prob", 0.0)) <= 1e-4:
            errors.append("final_equals_base_identity_collapse")
        if float(summary.get("fusion_gate_mean", 0.0)) <= 1e-3:
            errors.append("fusion_gate_not_active")
        if float(summary.get("fusion_delta_abs_mean", 0.0)) <= 1e-5:
            errors.append("fusion_delta_not_active")
        if float(summary.get("changed_pixel_ratio_at_0p5", 0.0)) <= 0.0:
            errors.append("no_threshold_changed_pixels")
        if not summary.get("checkpoint_has_fusion_keys", False):
            errors.append("missing_fusion_keys")
        if not summary.get("optimizer_has_fusion_params", False):
            errors.append("fusion_params_not_in_optimizer")

    payload = {
        "sacf_ready": len(errors) == 0,
        "errors": errors,
        "activation_summary": str(path),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["sacf_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
