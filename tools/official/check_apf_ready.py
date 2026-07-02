#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Block APF-OHEM unless candidate audit passed.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--anchor_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--min_candidate_to_budget_ratio", type=float, default=1.5)
    parser.add_argument("--max_flat_bg_ratio", type=float, default=0.35)
    parser.add_argument("--min_ohem_fp_component_coverage", type=float, default=0.40)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    anchor_dir = Path(args.anchor_dir)
    candidate_dir = Path(args.candidate_dir)
    checkpoint = Path(args.checkpoint)
    errors = []

    if not summary_path.exists():
        errors.append(f"missing_summary:{summary_path}")
        summary = {}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if not summary.get("gate_pass", False):
        errors.append("candidate_audit_gate_failed")
    if int(summary.get("target_leakage_pixels_total", -1)) != 0:
        errors.append("target_leakage_pixels_total_nonzero")
    if float(summary.get("candidate_to_budget_ratio_mean", 0.0)) <= args.min_candidate_to_budget_ratio:
        errors.append("candidate_to_budget_ratio_mean_too_low")
    if float(summary.get("flat_bg_ratio_mean", 1.0)) >= args.max_flat_bg_ratio:
        errors.append("flat_bg_ratio_mean_too_high")
    if float(summary.get("ohem_fp_component_coverage_mean", 0.0)) < args.min_ohem_fp_component_coverage:
        errors.append("ohem_fp_component_coverage_mean_too_low")

    if not anchor_dir.exists():
        errors.append(f"missing_anchor_dir:{anchor_dir}")
    if not candidate_dir.exists():
        errors.append(f"missing_candidate_dir:{candidate_dir}")
    if not checkpoint.exists():
        errors.append(f"missing_checkpoint:{checkpoint}")

    num_images = int(summary.get("num_images", 0) or 0)
    if anchor_dir.exists():
        anchor_npz = [p for p in anchor_dir.glob("*.npz")]
        if num_images and len(anchor_npz) != num_images:
            errors.append("anchor_count_mismatch")
        anchor_summary_path = anchor_dir / "summary.json"
        if not anchor_summary_path.exists():
            errors.append("missing_anchor_summary")
        elif checkpoint.exists():
            anchor_summary = json.loads(anchor_summary_path.read_text(encoding="utf-8"))
            expected_hash = sha256_file(checkpoint)
            if anchor_summary.get("checkpoint_sha256") != expected_hash:
                errors.append("checkpoint_hash_mismatch")
    if candidate_dir.exists():
        candidate_npz = [p for p in candidate_dir.glob("*.npz")]
        if num_images and len(candidate_npz) != num_images:
            errors.append("candidate_count_mismatch")

    payload = {
        "apf_ready": len(errors) == 0,
        "errors": errors,
        "summary": str(summary_path),
        "anchor_dir": str(anchor_dir),
        "candidate_dir": str(candidate_dir),
        "checkpoint": str(checkpoint),
        "thresholds": {
            "min_candidate_to_budget_ratio": args.min_candidate_to_budget_ratio,
            "max_flat_bg_ratio": args.max_flat_bg_ratio,
            "min_ohem_fp_component_coverage": args.min_ohem_fp_component_coverage,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["apf_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
