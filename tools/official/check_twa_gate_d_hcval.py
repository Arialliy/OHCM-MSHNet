#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(summary: dict, key: str) -> float:
    if key in summary:
        return float(summary[key])
    for section in ("metrics_at_threshold", "metrics"):
        metrics = summary.get(section)
        if isinstance(metrics, dict) and key in metrics:
            return float(metrics[key])
    raise KeyError(f"Metric {key} not found in summary keys: {sorted(summary.keys())}")


def evaluate_gate(
    ohem_summary: dict,
    twa_summary: dict,
    min_delta_miou: float,
    min_fa_reduction: float,
    min_delta_precision: float,
    min_delta_pd: float,
) -> dict:
    ohem_miou = get_metric(ohem_summary, "mIoU")
    twa_miou = get_metric(twa_summary, "mIoU")
    ohem_fa = get_metric(ohem_summary, "FA_ppm")
    twa_fa = get_metric(twa_summary, "FA_ppm")
    ohem_precision = get_metric(ohem_summary, "Precision")
    twa_precision = get_metric(twa_summary, "Precision")
    ohem_pd = get_metric(ohem_summary, "Pd")
    twa_pd = get_metric(twa_summary, "Pd")

    delta_miou = twa_miou - ohem_miou
    delta_fa = twa_fa - ohem_fa
    delta_precision = twa_precision - ohem_precision
    delta_pd = twa_pd - ohem_pd

    checks = {
        "delta_mIoU": delta_miou >= min_delta_miou,
        "delta_FA_ppm": delta_fa <= -min_fa_reduction,
        "delta_Precision": delta_precision >= min_delta_precision,
        "delta_Pd": delta_pd >= min_delta_pd,
    }
    gate_pass = all(checks.values())

    return {
        "gate": "Gate-TWA-D",
        "method": "TWA without BN",
        "split": "HC-Val",
        "seed": 42,
        "threshold": get_metric(twa_summary, "threshold"),
        "criteria": {
            "min_delta_mIoU": min_delta_miou,
            "min_fa_reduction_ppm": min_fa_reduction,
            "min_delta_Precision": min_delta_precision,
            "min_delta_Pd": min_delta_pd,
        },
        "ohem": {
            "mIoU": ohem_miou,
            "FA_ppm": ohem_fa,
            "Precision": ohem_precision,
            "Pd": ohem_pd,
        },
        "twa": {
            "mIoU": twa_miou,
            "FA_ppm": twa_fa,
            "Precision": twa_precision,
            "Pd": twa_pd,
        },
        "delta": {
            "mIoU": delta_miou,
            "FA_ppm": delta_fa,
            "Precision": delta_precision,
            "Pd": delta_pd,
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "next_allowed_gate": "Gate-TWA-E" if gate_pass else "STOP_TWA_4_NO_BN",
        "forbidden_if_fail": [
            "seed43",
            "seed44",
            "HC-Test",
            "blind",
            "external",
            "new TWA tuning",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Gate-TWA-D HC-Val criteria.")
    parser.add_argument("--ohem_summary", required=True)
    parser.add_argument("--twa_summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_fa_reduction", type=float, default=10.0)
    parser.add_argument("--min_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_delta_pd", type=float, default=0.0)
    args = parser.parse_args()

    result = evaluate_gate(
        load_summary(args.ohem_summary),
        load_summary(args.twa_summary),
        min_delta_miou=args.min_delta_miou,
        min_fa_reduction=args.min_fa_reduction,
        min_delta_precision=args.min_delta_precision,
        min_delta_pd=args.min_delta_pd,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if not result["gate_pass"]:
        raise SystemExit("Gate-TWA-D failed. Stop TWA without BN.")


if __name__ == "__main__":
    main()
