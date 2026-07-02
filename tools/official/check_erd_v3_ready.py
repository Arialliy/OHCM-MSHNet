#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Check ERD-v3 TP-CS readiness before seed training.")
    parser.add_argument("--project_dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--candidate_gate_json",
        default="docs/internal/erd_v3_candidate_audit_train/gate_pass.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    gate_path = Path(args.candidate_gate_json)
    if not gate_path.is_absolute():
        gate_path = project_dir / gate_path

    required = [
        project_dir / "model/ERD_MSHNet.py",
        project_dir / "docs/internal/ERD_V2_SEED42_STOP.md",
        project_dir / "docs/internal/ERD_V3_TPCS_DESIGN.md",
        gate_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing ERD-v3 readiness files:\n" + "\n".join(missing))

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate_pass", False):
        raise SystemExit("ERD-v3 candidate audit did not pass: %s" % gate)
    if gate.get("split") != "train":
        raise SystemExit("ERD-v3 candidate audit must be train split: %s" % gate)

    print("ERD-v3 ready: PASS")


if __name__ == "__main__":
    main()
