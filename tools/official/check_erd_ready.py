#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Check ERD-MSHNet v2 readiness before seed training.")
    parser.add_argument("--audit_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    required_files = [
        "model/ERD_MSHNet.py",
        "tests/test_erd_mshnet_semantics.py",
        "tools/official/audit_online_gate_candidates.py",
    ]
    for path in required_files:
        if not os.path.exists(path):
            raise SystemExit("Missing required file: %s" % path)

    with open(args.audit_json, "r", encoding="utf-8") as f:
        audit = json.load(f)
    if not audit.get("gate_pass", False):
        raise SystemExit("Gate audit is not passed: %s" % audit)
    if audit.get("source_split") != "train":
        raise SystemExit("Gate audit must be train split only: %s" % audit.get("source_split"))

    print("ERD_READY: PASS")


if __name__ == "__main__":
    main()
