#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="ECDV Gate-B decoy bank summary gate.")
    parser.add_argument("--bank_dir", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--min_decoys_per_image", type=float, default=1.0)
    parser.add_argument("--min_evidence_success_ratio", type=float, default=0.50)
    parser.add_argument("--min_mean_prob_gain", type=float, default=0.20)
    parser.add_argument("--min_area_in_target_range_ratio", type=float, default=0.80)
    parser.add_argument("--max_flat_artifact_ratio", type=float, default=0.30)
    args = parser.parse_args()

    summary_path = Path(args.summary) if args.summary else Path(args.bank_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "target_dilate_overlap_pixels": int(summary.get("target_dilate_overlap_pixels", -1)) == 0,
        "decoys_per_image_mean": float(summary.get("decoys_per_image_mean", 0.0)) >= args.min_decoys_per_image,
        "evidence_response_success_ratio": float(summary.get("evidence_response_success_ratio", 0.0)) >= args.min_evidence_success_ratio,
        "mean_prob_gain": float(summary.get("mean_prob_gain", 0.0)) >= args.min_mean_prob_gain,
        "area_in_target_range_ratio": float(summary.get("area_in_target_range_ratio", 0.0)) >= args.min_area_in_target_range_ratio,
        "flat_artifact_ratio": float(summary.get("flat_artifact_ratio", 1.0)) <= args.max_flat_artifact_ratio,
        "preview_audit_pass": bool(summary.get("preview_audit_pass", False)) is True,
    }
    summary["checks"] = checks
    summary["gate_pass"] = all(checks.values())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
