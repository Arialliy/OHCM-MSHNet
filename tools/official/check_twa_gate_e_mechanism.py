#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.twa_gate_utils import (  # noqa: E402
    DeltaRecord,
    MetricRecord,
    delta_metrics,
    load_json,
    load_metrics,
    load_named_metrics,
    pass_hcval_improvement,
    pass_nonregression,
    safe_positive_ratio,
    write_json,
)


FORBIDDEN_IF_FAIL = [
    "seed43",
    "seed44",
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new model training",
    "new verifier",
    "new suppression head",
]

FORBIDDEN_BEFORE_LATER_GATES = [
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new model training",
    "new verifier",
    "new suppression head",
]


def _metric_dict(record: MetricRecord) -> dict[str, float]:
    return record.to_dict()


def _delta_dict(delta: DeltaRecord) -> dict[str, float]:
    return delta.to_dict()


def _best_single(records: dict[str, MetricRecord]) -> tuple[str, MetricRecord]:
    if not records:
        raise ValueError("Gate-E requires at least one --single_late NAME=summary.json entry.")
    return max(records.items(), key=lambda item: (item[1].mIoU, -item[1].FA_ppm, item[1].Precision, item[1].Pd))


def _check_gate_d(gate_d_summary: str | None) -> tuple[bool, dict[str, Any]]:
    if gate_d_summary is None:
        return False, {"reason": "missing --gate_d_summary"}

    summary = load_json(gate_d_summary)
    gate_pass = bool(summary.get("gate_pass", False))
    next_allowed_gate = summary.get("next_allowed_gate")
    candidate = summary.get("candidate") or summary.get("current_candidate") or summary.get("method")

    ok = gate_pass and next_allowed_gate == "Gate-TWA-E"
    return ok, {
        "path": gate_d_summary,
        "gate_pass": gate_pass,
        "next_allowed_gate": next_allowed_gate,
        "candidate": candidate,
    }


def _retention_report(
    *,
    ohem_hcval: MetricRecord,
    twa4_hcval: MetricRecord,
    tce4_hcval: MetricRecord,
    min_tce_retention: float,
) -> tuple[bool, dict[str, Any]]:
    twa_delta = delta_metrics(twa4_hcval, ohem_hcval)
    tce_delta = delta_metrics(tce4_hcval, ohem_hcval)

    miou_retention = safe_positive_ratio(twa_delta.mIoU, tce_delta.mIoU)
    fa_retention = safe_positive_ratio(-twa_delta.FA_ppm, -tce_delta.FA_ppm)
    precision_retention = safe_positive_ratio(twa_delta.Precision, tce_delta.Precision)

    checks: dict[str, bool | None] = {
        "mIoU": None if miou_retention is None else miou_retention >= min_tce_retention,
        "FA_ppm": None if fa_retention is None else fa_retention >= min_tce_retention,
    }
    active_checks = [value for value in checks.values() if value is not None]
    ok = bool(active_checks) and all(active_checks)

    return ok, {
        "min_tce_retention": min_tce_retention,
        "twa4_delta_vs_ohem_hcval": _delta_dict(twa_delta),
        "tce4_delta_vs_ohem_hcval": _delta_dict(tce_delta),
        "retention_ratio": {
            "mIoU": miou_retention,
            "FA_ppm": fa_retention,
            "Precision": precision_retention,
        },
        "checked_metrics": checks,
        "pass": ok,
    }


def _variant_trend_report(
    *,
    ohem_hcval: MetricRecord,
    variant_hcval: dict[str, MetricRecord],
    trend_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    if "TWA-4" not in variant_hcval:
        return False, {
            "reason": "TWA-4 missing from --twa_variant_hcval entries",
            "variant_names": sorted(variant_hcval),
        }
    if len(variant_hcval) < 3:
        return False, {
            "reason": "Gate-E requires at least TWA-2/TWA-3/TWA-4 HC-Val summaries",
            "variant_names": sorted(variant_hcval),
        }

    records: dict[str, Any] = {}
    positive_count = 0
    for name, metrics in sorted(variant_hcval.items()):
        delta = delta_metrics(metrics, ohem_hcval)
        positive = (
            delta.mIoU >= -trend_tolerance
            and delta.FA_ppm <= trend_tolerance
            and delta.Precision >= -trend_tolerance
            and delta.Pd >= -trend_tolerance
        )
        if positive:
            positive_count += 1
        records[name] = {
            "metrics": _metric_dict(metrics),
            "delta_vs_ohem_hcval": _delta_dict(delta),
            "nonnegative_direction": positive,
        }

    min_variant_miou = min(metrics.mIoU for metrics in variant_hcval.values())
    twa4_not_worst = variant_hcval["TWA-4"].mIoU >= min_variant_miou - trend_tolerance
    enough_positive = positive_count >= 2
    ok = enough_positive and twa4_not_worst

    return ok, {
        "variant_names": sorted(variant_hcval),
        "records": records,
        "positive_count": positive_count,
        "required_positive_count": 2,
        "twa4_not_worst_by_mIoU": twa4_not_worst,
        "pass": ok,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate-TWA-E mechanism checker for TWA without BN on seed42 only.")
    parser.add_argument("--gate_d_summary", required=True)
    parser.add_argument("--ohem_full", required=True)
    parser.add_argument("--ohem_hcval", required=True)
    parser.add_argument("--twa4_full", required=True)
    parser.add_argument("--twa4_hcval", required=True)
    parser.add_argument("--tce4_hcval", required=True)
    parser.add_argument("--single_late", action="append", default=[])
    parser.add_argument("--twa_variant_hcval", action="append", default=[])
    parser.add_argument("--twa_variant_full", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_hcval_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_hcval_fa_reduction", type=float, default=10.0)
    parser.add_argument("--min_tce_retention", type=float, default=0.30)
    parser.add_argument("--trend_tolerance", type=float, default=1e-12)
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gate_d_ok, gate_d_report = _check_gate_d(args.gate_d_summary)

    ohem_full = load_metrics(args.ohem_full)
    ohem_hcval = load_metrics(args.ohem_hcval)
    twa4_full = load_metrics(args.twa4_full)
    twa4_hcval = load_metrics(args.twa4_hcval)
    tce4_hcval = load_metrics(args.tce4_hcval)

    single_late_records = load_named_metrics(args.single_late)
    best_single_name, best_single_metrics = _best_single(single_late_records)

    variant_hcval = load_named_metrics(args.twa_variant_hcval)
    variant_hcval.setdefault("TWA-4", twa4_hcval)
    variant_full = load_named_metrics(args.twa_variant_full)
    variant_full.setdefault("TWA-4", twa4_full)

    twa4_full_delta = delta_metrics(twa4_full, ohem_full)
    twa4_hcval_delta = delta_metrics(twa4_hcval, ohem_hcval)
    twa4_vs_best_single_delta = delta_metrics(twa4_hcval, best_single_metrics)

    full_guard_pass = pass_nonregression(twa4_full_delta)
    hcval_guard_pass = pass_hcval_improvement(
        twa4_hcval_delta,
        min_delta_miou=args.min_hcval_delta_miou,
        min_fa_reduction=args.min_hcval_fa_reduction,
    )
    best_single_guard_pass = pass_nonregression(twa4_vs_best_single_delta)
    retention_pass, retention = _retention_report(
        ohem_hcval=ohem_hcval,
        twa4_hcval=twa4_hcval,
        tce4_hcval=tce4_hcval,
        min_tce_retention=args.min_tce_retention,
    )
    trend_pass, trend = _variant_trend_report(
        ohem_hcval=ohem_hcval,
        variant_hcval=variant_hcval,
        trend_tolerance=args.trend_tolerance,
    )

    conditions = {
        "gate_d_passed_and_allows_gate_e": gate_d_ok,
        "twa4_full_nonregression_vs_ohem": full_guard_pass,
        "twa4_hcval_improvement_vs_ohem": hcval_guard_pass,
        "twa4_not_worse_than_best_single_late_hcval": best_single_guard_pass,
        "twa4_retains_tce_hard_split_gain": retention_pass,
        "twa2_twa3_twa4_trend_reasonable": trend_pass,
    }
    gate_pass = all(conditions.values())

    return {
        "gate": "Gate-TWA-E",
        "method": "TWA without BN recalibration",
        "seed": 42,
        "split_scope": ["Full", "HC-Val"],
        "threshold": 0.5,
        "gate_pass": gate_pass,
        "decision": "PROCEED_TWA_NO_BN_TO_THREE_SEED_GATE" if gate_pass else "STOP_TWA_NO_BN_AT_GATE_E",
        "next_allowed_gate": (
            "Gate-TWA-F-seed43-seed44-paired-Full-HCVal" if gate_pass else "STOP_TWA_NO_BN_AT_GATE_E"
        ),
        "conditions": conditions,
        "gate_d_report": gate_d_report,
        "ohem": {"Full": _metric_dict(ohem_full), "HC-Val": _metric_dict(ohem_hcval)},
        "twa4_without_bn": {
            "Full": _metric_dict(twa4_full),
            "HC-Val": _metric_dict(twa4_hcval),
            "delta_full_vs_ohem": _delta_dict(twa4_full_delta),
            "delta_hcval_vs_ohem": _delta_dict(twa4_hcval_delta),
        },
        "best_single_late_checkpoint": {
            "name": best_single_name,
            "metrics": _metric_dict(best_single_metrics),
            "twa4_delta_vs_best_single_hcval": _delta_dict(twa4_vs_best_single_delta),
            "all_single_late_hcval": {
                name: _metric_dict(metrics) for name, metrics in sorted(single_late_records.items())
            },
        },
        "tce4_hcval": _metric_dict(tce4_hcval),
        "tce_retention": retention,
        "twa_variant_trend_hcval": trend,
        "twa_variant_full_report": {
            name: {"metrics": _metric_dict(metrics), "delta_vs_ohem_full": _delta_dict(delta_metrics(metrics, ohem_full))}
            for name, metrics in sorted(variant_full.items())
        },
        "forbidden_if_fail": FORBIDDEN_IF_FAIL,
        "forbidden_before_later_gates": FORBIDDEN_BEFORE_LATER_GATES,
        "notes": [
            "Gate-E is a mechanism/compression gate, not a new training gate.",
            "Do not run seed43/44 unless this gate passes.",
            "Do not run HC-Test, blind, or external at Gate-E.",
            "BN recalibration remains stopped and is not part of the active candidate.",
        ],
    }


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    result = evaluate(args)
    write_json(args.output, result)
    if not result["gate_pass"]:
        raise SystemExit("Gate-TWA-E failed. Stop TWA without BN before seed43/44.")


if __name__ == "__main__":
    main()
