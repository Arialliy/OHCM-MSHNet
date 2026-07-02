#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Block EACF training unless Gate-F0 scale audit passed.")
    parser.add_argument("--scale_consensus_summary", required=True)
    parser.add_argument("--ohem_checkpoint", required=True)
    args = parser.parse_args()

    errors = []
    summary_path = Path(args.scale_consensus_summary)
    checkpoint_path = Path(args.ohem_checkpoint)
    if not summary_path.exists():
        errors.append("missing_scale_consensus_audit")
        summary = {}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("gate_pass", False):
            errors.append("scale_consensus_audit_failed")
    if not checkpoint_path.exists():
        errors.append("missing_ohem_checkpoint")

    payload = {
        "eacf_ready": len(errors) == 0,
        "errors": errors,
        "scale_consensus_summary": str(summary_path),
        "ohem_checkpoint": str(checkpoint_path),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["eacf_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
