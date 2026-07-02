#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CGA gate readiness.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["activation", "full"], default="activation")
    parser.add_argument("--component_audit", required=True)
    parser.add_argument("--activation_audit", default="")
    args = parser.parse_args()

    errors = []
    component_path = Path(args.component_audit)
    try:
        component = load_json(component_path)
    except Exception as exc:
        errors.append(f"component_audit_missing_or_unreadable: {exc}")
        component = {}

    if component and not component.get("gate_pass", False):
        errors.append("component_target_audit_failed")
    if component and component.get("dataset") != args.dataset_name:
        errors.append("component_audit_dataset_mismatch")

    activation = {}
    if args.stage == "full":
        if not args.activation_audit:
            errors.append("activation_audit_required_for_full_stage")
        else:
            activation_path = Path(args.activation_audit)
            try:
                activation = load_json(activation_path)
            except Exception as exc:
                errors.append(f"activation_audit_missing_or_unreadable: {exc}")
            if activation and not activation.get("gate_pass", False):
                errors.append("activation_audit_failed")
            if activation and activation.get("dataset") != args.dataset_name:
                errors.append("activation_audit_dataset_mismatch")

    payload = {
        "gate_pass": len(errors) == 0,
        "stage": args.stage,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "component_audit": str(component_path),
        "activation_audit": args.activation_audit,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
