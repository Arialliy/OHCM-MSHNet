#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def evaluate(
    summary: Dict[str, Any],
    *,
    min_images_with_neg: int = 50,
    min_neg_pixels_total: int = 500,
    max_target_leakage_pixels: int = 0,
    max_neg_protect_overlap_pixels: int = 0,
    expected_num_images: int | None = None,
) -> Dict[str, Any]:
    items = summary.get("items", [])
    if expected_num_images is None:
        expected_num_images = int(summary.get("train_images", len(items)))

    num_images = int(summary.get("num_images", len(items)))
    checks = {
        "train_only": bool(summary.get("train_only", False)) and str(summary.get("split", "")).lower() == "train",
        "num_images_matches_train": num_images == expected_num_images,
        "num_images_matches_items": num_images == len(items),
        "enough_images_with_neg": int(summary.get("num_images_with_neg", 0)) >= min_images_with_neg,
        "enough_neg_pixels": int(summary.get("neg_pixels_total", 0)) >= min_neg_pixels_total,
        "has_protect_pixels": int(summary.get("protect_pixels_total", 0)) > 0,
        "no_target_leakage": int(summary.get("target_leakage_pixels_total", 0)) <= max_target_leakage_pixels,
        "no_neg_protect_overlap": int(summary.get("neg_protect_overlap_pixels_total", 0)) <= max_neg_protect_overlap_pixels,
    }
    fail_reasons: List[str] = [name for name, ok in checks.items() if not ok]
    gate_pass = len(fail_reasons) == 0
    return {
        "gate": "Gate-TCSR-A",
        "gate_name": "train-only sparse bank audit",
        "gate_pass": gate_pass,
        "decision": "PASS_TCSR_BANK_AUDIT" if gate_pass else "STOP_TCSR_AT_BANK_AUDIT",
        "next_allowed_gate": "Gate-TCSR-B-activation-sanity" if gate_pass else None,
        "checks": checks,
        "fail_reasons": fail_reasons,
        "thresholds": {
            "min_images_with_neg": min_images_with_neg,
            "min_neg_pixels_total": min_neg_pixels_total,
            "max_target_leakage_pixels": max_target_leakage_pixels,
            "max_neg_protect_overlap_pixels": max_neg_protect_overlap_pixels,
            "expected_num_images": expected_num_images,
        },
        "summary": {
            "num_images": num_images,
            "train_images": int(summary.get("train_images", expected_num_images)),
            "item_count": len(items),
            "num_images_with_neg": int(summary.get("num_images_with_neg", 0)),
            "neg_pixels_total": int(summary.get("neg_pixels_total", 0)),
            "protect_pixels_total": int(summary.get("protect_pixels_total", 0)),
            "target_leakage_pixels_total": int(summary.get("target_leakage_pixels_total", 0)),
            "neg_protect_overlap_pixels_total": int(summary.get("neg_protect_overlap_pixels_total", 0)),
        },
        "forbidden_next_actions": [
            "seed43",
            "seed44",
            "HC-Test",
            "blind",
            "external",
            "threshold_search",
            "BN_recalibration_tuning",
            "seed_search",
            "checkpoint_selection_as_final_method",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Gate-TCSR-A sparse bank audit.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min_images_with_neg", type=int, default=50)
    parser.add_argument("--min_neg_pixels_total", type=int, default=500)
    parser.add_argument("--max_target_leakage_pixels", type=int, default=0)
    parser.add_argument("--max_neg_protect_overlap_pixels", type=int, default=0)
    parser.add_argument("--expected_num_images", type=int, default=None)
    args = parser.parse_args()

    summary = load_json(args.summary)
    result = evaluate(
        summary,
        min_images_with_neg=args.min_images_with_neg,
        min_neg_pixels_total=args.min_neg_pixels_total,
        max_target_leakage_pixels=args.max_target_leakage_pixels,
        max_neg_protect_overlap_pixels=args.max_neg_protect_overlap_pixels,
        expected_num_images=args.expected_num_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["gate_pass"]:
        raise SystemExit(result["decision"])


if __name__ == "__main__":
    main()
