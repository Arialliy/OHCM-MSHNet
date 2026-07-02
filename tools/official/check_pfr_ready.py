#!/usr/bin/env python3
"""Guard for PFR-MSHNet training after seed42 Full Gate failure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def write_json(path: str | None, summary: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def validate_candidate_audit(path: Path) -> tuple[bool, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    passed = bool(summary.get("gate_pass", False))
    payload = {
        "gate_pass": summary.get("gate_pass"),
        "fail_reasons": summary.get("fail_reasons", []),
        "candidate_empty_image_ratio": summary.get("candidate_empty_image_ratio"),
        "target_leakage_pixels": summary.get("target_leakage_pixels"),
    }
    return passed, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Block PFR training after seed42 Full Gate failure.")
    parser.add_argument("--audit_summary", default=None)
    parser.add_argument("--allow_failed_pfr", action="store_true")
    parser.add_argument("--stop_doc", default="docs/internal/PFR_SEED42_FULL_STOP.md")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    stop_doc = Path(args.stop_doc)
    blocked = stop_doc.exists() and not args.allow_failed_pfr
    summary = {
        "pfr_ready": not blocked,
        "blocked": blocked,
        "reason": "PFR seed42 Full Gate failed" if blocked else "explicit override or no stop doc",
        "stop_doc": str(stop_doc),
        "audit_summary": args.audit_summary,
    }

    if blocked:
        write_json(args.json_out, summary)
        print("[BLOCKED] PFR-MSHNet is stopped after seed42 Full Gate failure.")
        print(f"[INFO] stop_doc={stop_doc}")
        print("[INFO] Use --allow_failed_pfr only for failure-analysis reruns.")
        return 2

    if args.audit_summary:
        audit_passed, audit_payload = validate_candidate_audit(Path(args.audit_summary))
        summary["candidate_audit"] = audit_payload
        if not audit_passed:
            summary["pfr_ready"] = False
            summary["blocked"] = True
            summary["reason"] = "PFR candidate audit did not pass"
            write_json(args.json_out, summary)
            print("PFR_NOT_READY")
            print(json.dumps(audit_payload, indent=2))
            return 2

    write_json(args.json_out, summary)
    print("[PASS] PFR ready guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
