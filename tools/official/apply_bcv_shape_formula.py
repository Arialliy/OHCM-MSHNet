#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic BCV shape_formula config after Gate-D pass.")
    parser.add_argument("--gate_d_summary", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--beta", type=float, required=True, choices=[0.02, 0.05, 0.10])
    parser.add_argument("--shape_temp", type=float, default=0.2)
    parser.add_argument("--allow_failed_gate", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.gate_d_summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("gate_pass", False)) and not args.allow_failed_gate:
        print("[BLOCKED] Gate-D did not pass; deterministic shape calibration is not allowed.")
        raise SystemExit(2)

    theta = float(summary.get("target_protection_shape_threshold", 0.0))
    config = {
        "model_name": "BCVMSHNet",
        "bcv_validity_mode": "shape_formula",
        "bcv_shape_theta": theta,
        "bcv_shape_temp": float(args.shape_temp),
        "bcv_beta_max": float(args.beta),
        "bcv_eval_beta": float(args.beta),
        "gate_d_summary": str(summary_path.resolve()),
    }
    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
