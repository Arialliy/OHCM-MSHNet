#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.twa_gate_utils import (  # noqa: E402
    DeltaRecord,
    MetricRecord,
    delta_metrics,
    load_json,
    load_metrics,
    metrics_from_summary,
    pass_hcval_improvement,
    pass_nonregression,
    write_json,
)


BEST_SINGLE_CONDITION = "twa4_not_worse_than_best_single_late_hcval"
STOP_TWA_GATE_E = "STOP_TWA_NO_BN_AT_GATE_E"

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
    "new checkpoint combination search",
    "threshold search",
]

FORBIDDEN_BEFORE_LATER_GATES = [
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new verifier",
    "new suppression head",
    "new checkpoint combination search",
    "threshold search",
]


def _metric_dict(record: MetricRecord) -> dict[str, float]:
    return record.to_dict()


def _delta_dict(delta: DeltaRecord) -> dict[str, float]:
    return delta.to_dict()


def _failed_conditions(conditions: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key, value in conditions.items() if value is not True)


def _gate_e_failure_report(gate_e_summary: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    conditions = gate_e_summary.get("conditions")
    if not isinstance(conditions, Mapping):
        return False, {
            "reason": "missing_or_invalid_conditions",
            "conditions_type": type(conditions).__name__,
        }

    failed = _failed_conditions(conditions)
    gate_pass = gate_e_summary.get("gate_pass") is True
    decision = gate_e_summary.get("decision")
    next_allowed_gate = gate_e_summary.get("next_allowed_gate")

    stopped_at_gate_e = (
        gate_pass is False
        and (decision == STOP_TWA_GATE_E or next_allowed_gate == STOP_TWA_GATE_E)
    )
    only_best_single_failed = failed == [BEST_SINGLE_CONDITION]

    ok = stopped_at_gate_e and only_best_single_failed
    return ok, {
        "gate_pass": gate_pass,
        "decision": decision,
        "next_allowed_gate": next_allowed_gate,
        "failed_conditions": failed,
        "required_failed_conditions": [BEST_SINGLE_CONDITION],
        "stopped_at_gate_e": stopped_at_gate_e,
        "only_best_single_failed": only_best_single_failed,
        "pass": ok,
    }


def _records_from_gate_e_best_single(gate_e_summary: Mapping[str, Any]) -> dict[str, MetricRecord]:
    best_block = gate_e_summary.get("best_single_late_checkpoint")
    if not isinstance(best_block, Mapping):
        return {}

    records: dict[str, MetricRecord] = {}
    all_single = best_block.get("all_single_late_hcval")
    if isinstance(all_single, Mapping):
        for name, payload in all_single.items():
            if isinstance(payload, Mapping):
                records[str(name)] = metrics_from_summary(payload)

    best_name = best_block.get("name")
    best_metrics = best_block.get("metrics")
    if isinstance(best_name, str) and isinstance(best_metrics, Mapping) and best_name not in records:
        records[best_name] = metrics_from_summary(best_metrics)

    return records


def _metric_priority(item: tuple[str, MetricRecord]) -> tuple[float, float, float, float]:
    _, metrics = item
    return (metrics.mIoU, -metrics.FA_ppm, metrics.Precision, metrics.Pd)


def _best_single_report(
    *,
    gate_e_summary: Mapping[str, Any],
    expected_snapshot_name: str,
    min_unique_miou_margin: float,
) -> tuple[bool, dict[str, Any]]:
    best_block = gate_e_summary.get("best_single_late_checkpoint")
    if not isinstance(best_block, Mapping):
        return False, {"reason": "missing_best_single_late_checkpoint_block"}

    reported_name = best_block.get("name")
    records = _records_from_gate_e_best_single(gate_e_summary)
    if not records:
        return False, {
            "reason": "missing_all_single_late_hcval_records",
            "reported_name": reported_name,
        }

    ranked = sorted(records.items(), key=_metric_priority, reverse=True)
    computed_best_name, computed_best_metrics = ranked[0]
    second_name = ranked[1][0] if len(ranked) > 1 else None
    second_metrics = ranked[1][1] if len(ranked) > 1 else None
    unique_miou_margin = None
    unique_pass = False
    if second_metrics is not None:
        unique_miou_margin = computed_best_metrics.mIoU - second_metrics.mIoU
        unique_pass = unique_miou_margin >= min_unique_miou_margin

    name_matches_report = reported_name == computed_best_name
    name_is_expected = reported_name == expected_snapshot_name

    ok = name_matches_report and name_is_expected and unique_pass
    return ok, {
        "reported_best_single": reported_name,
        "computed_best_single": computed_best_name,
        "expected_snapshot_name": expected_snapshot_name,
        "name_matches_report": name_matches_report,
        "name_is_expected": name_is_expected,
        "second_best_single": second_name,
        "unique_miou_margin": unique_miou_margin,
        "min_unique_miou_margin": min_unique_miou_margin,
        "unique_pass": unique_pass,
        "ranked_single_late_hcval": [
            {"name": name, "metrics": _metric_dict(metrics)}
            for name, metrics in ranked
        ],
        "pass": ok,
    }


def _comparison_pass_hcval(
    delta: DeltaRecord,
    *,
    min_delta_miou: float,
    min_fa_reduction: float,
    min_delta_precision: float = 0.0,
    min_delta_pd: float = 0.0,
) -> bool:
    return pass_hcval_improvement(
        delta,
        min_delta_miou=min_delta_miou,
        min_fa_reduction=min_fa_reduction,
        min_delta_precision=min_delta_precision,
        min_delta_pd=min_delta_pd,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate-LS-A checker: promote the Gate-E best single late checkpoint "
            "to a frozen late-snapshot candidate only if seed42 Full and HC-Val guards pass."
        )
    )
    parser.add_argument("--gate_e_summary", required=True)
    parser.add_argument("--ohem_full", required=True)
    parser.add_argument("--ohem_hcval", required=True)
    parser.add_argument("--snapshot_full", required=True)
    parser.add_argument("--snapshot_hcval", required=True)
    parser.add_argument("--twa4_hcval", required=True)
    parser.add_argument("--snapshot_name", default="ep250")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_unique_miou_margin", type=float, default=0.005)
    parser.add_argument("--min_hcval_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_hcval_fa_reduction", type=float, default=10.0)
    parser.add_argument("--min_snapshot_vs_twa4_miou", type=float, default=0.005)
    parser.add_argument("--min_snapshot_vs_twa4_fa_reduction", type=float, default=10.0)
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gate_e_summary = load_json(args.gate_e_summary)

    gate_e_ok, gate_e_report = _gate_e_failure_report(gate_e_summary)
    best_single_ok, best_single_report = _best_single_report(
        gate_e_summary=gate_e_summary,
        expected_snapshot_name=args.snapshot_name,
        min_unique_miou_margin=args.min_unique_miou_margin,
    )

    ohem_full = load_metrics(args.ohem_full)
    ohem_hcval = load_metrics(args.ohem_hcval)
    snapshot_full = load_metrics(args.snapshot_full)
    snapshot_hcval = load_metrics(args.snapshot_hcval)
    twa4_hcval = load_metrics(args.twa4_hcval)

    snapshot_full_delta_vs_ohem = delta_metrics(snapshot_full, ohem_full)
    snapshot_hcval_delta_vs_ohem = delta_metrics(snapshot_hcval, ohem_hcval)
    snapshot_hcval_delta_vs_twa4 = delta_metrics(snapshot_hcval, twa4_hcval)

    full_nonregression_pass = pass_nonregression(snapshot_full_delta_vs_ohem)
    hcval_vs_ohem_pass = _comparison_pass_hcval(
        snapshot_hcval_delta_vs_ohem,
        min_delta_miou=args.min_hcval_delta_miou,
        min_fa_reduction=args.min_hcval_fa_reduction,
    )
    hcval_vs_twa4_pass = _comparison_pass_hcval(
        snapshot_hcval_delta_vs_twa4,
        min_delta_miou=args.min_snapshot_vs_twa4_miou,
        min_fa_reduction=args.min_snapshot_vs_twa4_fa_reduction,
    )

    conditions = {
        "gate_e_failed_only_because_best_single_won": gate_e_ok,
        "best_single_is_frozen_snapshot": best_single_ok,
        "snapshot_full_nonregression_vs_ohem": full_nonregression_pass,
        "snapshot_hcval_improvement_vs_ohem": hcval_vs_ohem_pass,
        "snapshot_hcval_improvement_vs_twa4": hcval_vs_twa4_pass,
    }
    gate_pass = all(conditions.values())

    return {
        "gate": "Gate-LS-A",
        "method": "LateSnapshot-ep250",
        "origin": "Gate-TWA-E best single late checkpoint audit",
        "seed": 42,
        "threshold": 0.5,
        "gate_pass": gate_pass,
        "decision": (
            "PROCEED_LATE_SNAPSHOT_EP250_TO_GATE_LS_B_SEED43_44"
            if gate_pass
            else "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"
        ),
        "next_allowed_gate": (
            "Gate-LS-B-seed43-seed44-paired-Full-HCVal"
            if gate_pass
            else "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"
        ),
        "conditions": conditions,
        "gate_e_report": gate_e_report,
        "best_single_report": best_single_report,
        "ohem": {
            "Full": _metric_dict(ohem_full),
            "HC-Val": _metric_dict(ohem_hcval),
        },
        "snapshot": {
            "name": args.snapshot_name,
            "Full": _metric_dict(snapshot_full),
            "HC-Val": _metric_dict(snapshot_hcval),
            "delta_full_vs_ohem": _delta_dict(snapshot_full_delta_vs_ohem),
            "delta_hcval_vs_ohem": _delta_dict(snapshot_hcval_delta_vs_ohem),
            "delta_hcval_vs_twa4": _delta_dict(snapshot_hcval_delta_vs_twa4),
        },
        "twa4_without_bn": {
            "HC-Val": _metric_dict(twa4_hcval),
        },
        "gate_criteria": {
            "min_unique_miou_margin": args.min_unique_miou_margin,
            "full_nonregression": {
                "min_delta_mIoU": 0.0,
                "max_delta_FA_ppm": 0.0,
                "min_delta_Precision": 0.0,
                "min_delta_Pd": 0.0,
            },
            "hcval_vs_ohem": {
                "min_delta_mIoU": args.min_hcval_delta_miou,
                "min_fa_reduction_ppm": args.min_hcval_fa_reduction,
                "min_delta_Precision": 0.0,
                "min_delta_Pd": 0.0,
            },
            "hcval_vs_twa4": {
                "min_delta_mIoU": args.min_snapshot_vs_twa4_miou,
                "min_fa_reduction_ppm": args.min_snapshot_vs_twa4_fa_reduction,
                "min_delta_Precision": 0.0,
                "min_delta_Pd": 0.0,
            },
        },
        "forbidden_if_fail": FORBIDDEN_IF_FAIL,
        "forbidden_before_later_gates": FORBIDDEN_BEFORE_LATER_GATES,
        "notes": [
            "Do not rewrite Gate-E as PASS; Gate-E remains failed for TWA-4 weight averaging.",
            "This gate only validates the ep250 snapshot exposed by Gate-E; it does not authorize checkpoint search.",
            "If this gate passes, seed43/44 are allowed only for the frozen ep250-vs-ep400 paired protocol.",
            "HC-Test, blind, and external remain forbidden before later gates.",
        ],
    }


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    result = evaluate(args)
    write_json(args.output, result)
    if not result["gate_pass"]:
        raise SystemExit("Gate-LS-A failed. Stop LateSnapshot-ep250 before seed43/44.")


if __name__ == "__main__":
    main()
