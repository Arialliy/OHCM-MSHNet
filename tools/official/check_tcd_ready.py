#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Block TCD training unless Gate-T2R passed.")
    parser.add_argument("--audit_summary", required=True)
    args = parser.parse_args()

    path = Path(args.audit_summary)
    if not path.exists():
        raise FileNotFoundError(path)

    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("gate_pass", False):
        print("TCD_NOT_READY")
        print(
            json.dumps(
                {
                    "gate_pass": summary.get("gate_pass"),
                    "fail_reasons": summary.get("fail_reasons", []),
                    "topk_far_absdiff_mean": summary.get("topk_far_absdiff_mean"),
                    "teacher_lower_on_student_high_far_rate": summary.get(
                        "teacher_lower_on_student_high_far_rate"
                    ),
                },
                indent=2,
            )
        )
        sys.exit(2)

    print("TCD_READY")
    sys.exit(0)


if __name__ == "__main__":
    main()
