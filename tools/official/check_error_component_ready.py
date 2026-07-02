#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Block component-mining routes unless Gate-ECA passed.")
    parser.add_argument("--audit_summary", "--summary", dest="audit_summary", required=True)
    parser.add_argument("--min_components", type=int, default=50)
    parser.add_argument("--min_nonflat_ratio", type=float, default=0.30)
    parser.add_argument("--max_flat_ratio", type=float, default=0.50)
    args = parser.parse_args()

    summary_path = Path(args.audit_summary)
    errors = []
    if not summary_path.exists():
        errors.append(f"missing_summary:{summary_path}")
        summary = {}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if not summary.get("gate_pass", False):
        errors.append("error_component_audit_gate_failed")
    if int(summary.get("total_detached_far_fp_components", 0)) < args.min_components:
        errors.append("total_detached_far_fp_components_too_low")
    if float(summary.get("nonflat_detached_far_fp_ratio", 0.0)) < args.min_nonflat_ratio:
        errors.append("nonflat_detached_far_fp_ratio_too_low")
    if float(summary.get("flat_bg_ratio_mean", 1.0)) > args.max_flat_ratio:
        errors.append("flat_bg_ratio_mean_too_high")
    if int(summary.get("target_leakage_components", 1)) > 0:
        errors.append("target_leakage_components_nonzero")

    payload = {"error_component_ready": len(errors) == 0, "errors": errors}
    print(json.dumps(payload, indent=2))
    return 0 if payload["error_component_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
