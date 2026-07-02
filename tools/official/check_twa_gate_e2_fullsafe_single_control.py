#!/usr/bin/env python3
"""
Gate-TWA-E2-FSC: Full-Safe Single-Late Control.

This checker does not train models and does not tune thresholds. It only
arbitrates among pre-registered late checkpoints after ep250 has been shown to
be Full-unsafe.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


METRIC_ALIASES = {
    "mIoU": ("mIoU", "miou", "mean_iou", "mean_IoU"),
    "Precision": ("Precision", "precision", "prec"),
    "Pd": ("Pd", "pd", "PD", "recall"),
    "FA_ppm": ("FA_ppm", "fa_ppm", "FAppm", "FA", "fa"),
}

NESTED_METRIC_KEYS = (
    "metrics",
    "metrics_at_threshold",
    "official_metrics",
    "summary_metrics",
    "aggregate",
    "overall",
)

ALLOWED_EPOCHS = {250, 300, 350, 400}
CANDIDATE_EPOCHS = {250, 300, 350}
BASELINE_EPOCH = 400


@dataclass(frozen=True)
class Metrics:
    mIoU: float
    Precision: float
    Pd: float
    FA_ppm: float


@dataclass(frozen=True)
class DeltaMetrics:
    mIoU: float
    Precision: float
    Pd: float
    FA_ppm: float


@dataclass(frozen=True)
class SnapshotRecord:
    epoch: int
    full: Metrics
    hcval: Metrics
    full_delta_vs_ohem: DeltaMetrics
    hcval_delta_vs_ohem: DeltaMetrics
    full_safe: bool
    hcval_positive: bool
    eligible_single: bool
    ineligible_reasons: Tuple[str, ...]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _lookup_nested(summary: Dict[str, Any], key: str) -> Optional[Any]:
    if key in summary:
        return summary[key]
    for nested_key in NESTED_METRIC_KEYS:
        metrics = summary.get(nested_key)
        if isinstance(metrics, dict) and key in metrics:
            return metrics[key]
    return None


def get_metric(summary: Dict[str, Any], canonical_key: str) -> float:
    aliases = METRIC_ALIASES[canonical_key]
    for key in aliases:
        value = _lookup_nested(summary, key)
        if value is not None:
            return float(value)
    available = sorted(list(summary.keys()))
    for nested_key in NESTED_METRIC_KEYS:
        nested = summary.get(nested_key)
        if isinstance(nested, dict):
            available.extend(f"{nested_key}.{k}" for k in nested.keys())
    raise KeyError(f"Metric {canonical_key} not found. Available keys: {available}")


def metrics_from_summary(summary: Dict[str, Any]) -> Metrics:
    return Metrics(
        mIoU=get_metric(summary, "mIoU"),
        Precision=get_metric(summary, "Precision"),
        Pd=get_metric(summary, "Pd"),
        FA_ppm=get_metric(summary, "FA_ppm"),
    )


def delta(candidate: Metrics, baseline: Metrics) -> DeltaMetrics:
    return DeltaMetrics(
        mIoU=candidate.mIoU - baseline.mIoU,
        Precision=candidate.Precision - baseline.Precision,
        Pd=candidate.Pd - baseline.Pd,
        FA_ppm=candidate.FA_ppm - baseline.FA_ppm,
    )


def check_full_safe(
    d: DeltaMetrics,
    *,
    min_delta_miou: float,
    min_delta_precision: float,
    min_delta_pd: float,
    max_delta_fa_ppm: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if d.mIoU < min_delta_miou:
        reasons.append("full_miou_regression")
    if d.Precision < min_delta_precision:
        reasons.append("full_precision_regression")
    if d.Pd < min_delta_pd:
        reasons.append("full_pd_regression")
    if d.FA_ppm > max_delta_fa_ppm:
        reasons.append("full_fa_regression")
    return len(reasons) == 0, reasons


def check_hcval_positive(
    d: DeltaMetrics,
    *,
    min_delta_miou: float,
    min_fa_reduction_ppm: float,
    min_delta_precision: float,
    min_delta_pd: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if d.mIoU < min_delta_miou:
        reasons.append("hcval_miou_not_enough")
    if d.FA_ppm > -min_fa_reduction_ppm:
        reasons.append("hcval_fa_reduction_not_enough")
    if d.Precision < min_delta_precision:
        reasons.append("hcval_precision_regression")
    if d.Pd < min_delta_pd:
        reasons.append("hcval_pd_regression")
    return len(reasons) == 0, reasons


def build_snapshot_record(
    *,
    epoch: int,
    full_summary: Dict[str, Any],
    hcval_summary: Dict[str, Any],
    ohem_full: Metrics,
    ohem_hcval: Metrics,
    min_full_delta_miou: float,
    min_full_delta_precision: float,
    min_full_delta_pd: float,
    max_full_delta_fa_ppm: float,
    min_hcval_delta_miou: float,
    min_hcval_fa_reduction_ppm: float,
    min_hcval_delta_precision: float,
    min_hcval_delta_pd: float,
) -> SnapshotRecord:
    if epoch not in ALLOWED_EPOCHS:
        raise ValueError(f"Epoch {epoch} is not allowed. Allowed: {sorted(ALLOWED_EPOCHS)}")

    full = metrics_from_summary(full_summary)
    hcval = metrics_from_summary(hcval_summary)
    full_delta = delta(full, ohem_full)
    hcval_delta = delta(hcval, ohem_hcval)

    full_safe, full_reasons = check_full_safe(
        full_delta,
        min_delta_miou=min_full_delta_miou,
        min_delta_precision=min_full_delta_precision,
        min_delta_pd=min_full_delta_pd,
        max_delta_fa_ppm=max_full_delta_fa_ppm,
    )
    hcval_positive, hc_reasons = check_hcval_positive(
        hcval_delta,
        min_delta_miou=min_hcval_delta_miou,
        min_fa_reduction_ppm=min_hcval_fa_reduction_ppm,
        min_delta_precision=min_hcval_delta_precision,
        min_delta_pd=min_hcval_delta_pd,
    )

    reasons: List[str] = []
    if epoch == BASELINE_EPOCH:
        reasons.append("baseline_epoch_400_not_candidate")
    if epoch not in CANDIDATE_EPOCHS:
        reasons.append("not_candidate_epoch")
    reasons.extend(full_reasons)
    reasons.extend(hc_reasons)

    eligible = epoch in CANDIDATE_EPOCHS and full_safe and hcval_positive
    return SnapshotRecord(
        epoch=epoch,
        full=full,
        hcval=hcval,
        full_delta_vs_ohem=full_delta,
        hcval_delta_vs_ohem=hcval_delta,
        full_safe=full_safe,
        hcval_positive=hcval_positive,
        eligible_single=eligible,
        ineligible_reasons=tuple(reasons),
    )


def parse_snapshot_arg(raw: str) -> Tuple[int, Path, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--snapshot must have format EPOCH:FULL_SUMMARY:HCVAL_SUMMARY")
    return int(parts[0]), Path(parts[1]), Path(parts[2])


def choose_best_eligible(records: Sequence[SnapshotRecord]) -> Optional[SnapshotRecord]:
    eligible = [record for record in records if record.eligible_single]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            record.hcval_delta_vs_ohem.mIoU,
            -record.hcval_delta_vs_ohem.FA_ppm,
            record.hcval_delta_vs_ohem.Precision,
            record.hcval_delta_vs_ohem.Pd,
        ),
    )


def validate_context(gate_e_summary: Dict[str, Any], ep250_gate_a_summary: Dict[str, Any]) -> Dict[str, Any]:
    gate_e_pass = bool(gate_e_summary.get("gate_pass", False))
    ep250_gate_a_pass = bool(ep250_gate_a_summary.get("gate_pass", False))
    context = {
        "gate_e_pass": gate_e_pass,
        "gate_e_decision": gate_e_summary.get("decision") or gate_e_summary.get("status"),
        "ep250_gate_a_pass": ep250_gate_a_pass,
        "ep250_gate_a_decision": ep250_gate_a_summary.get("decision") or ep250_gate_a_summary.get("status"),
    }
    context["context_valid"] = (not gate_e_pass) and (not ep250_gate_a_pass)
    return context


def decide(
    *,
    twa_full: Metrics,
    twa_hcval: Metrics,
    ohem_full: Metrics,
    ohem_hcval: Metrics,
    best_eligible: Optional[SnapshotRecord],
    min_full_delta_miou: float,
    min_full_delta_precision: float,
    min_full_delta_pd: float,
    max_full_delta_fa_ppm: float,
    min_hcval_delta_miou: float,
    min_hcval_fa_reduction_ppm: float,
    min_hcval_delta_precision: float,
    min_hcval_delta_pd: float,
    twa_vs_single_eps: float,
    min_hc_miou_switch_margin: float,
    disable_posthoc_single_promotion: bool,
) -> Dict[str, Any]:
    twa_full_delta = delta(twa_full, ohem_full)
    twa_hc_delta = delta(twa_hcval, ohem_hcval)
    twa_full_safe, twa_full_reasons = check_full_safe(
        twa_full_delta,
        min_delta_miou=min_full_delta_miou,
        min_delta_precision=min_full_delta_precision,
        min_delta_pd=min_full_delta_pd,
        max_delta_fa_ppm=max_full_delta_fa_ppm,
    )
    twa_hc_positive, twa_hc_reasons = check_hcval_positive(
        twa_hc_delta,
        min_delta_miou=min_hcval_delta_miou,
        min_fa_reduction_ppm=min_hcval_fa_reduction_ppm,
        min_delta_precision=min_hcval_delta_precision,
        min_delta_pd=min_hcval_delta_pd,
    )
    twa_eligible = twa_full_safe and twa_hc_positive

    result: Dict[str, Any] = {
        "twa4": {
            "full": asdict(twa_full),
            "hcval": asdict(twa_hcval),
            "full_delta_vs_ohem": asdict(twa_full_delta),
            "hcval_delta_vs_ohem": asdict(twa_hc_delta),
            "full_safe": twa_full_safe,
            "hcval_positive": twa_hc_positive,
            "eligible": twa_eligible,
            "ineligible_reasons": twa_full_reasons + twa_hc_reasons,
        }
    }

    if not twa_eligible and best_eligible is None:
        result.update(
            {
                "gate_pass": False,
                "decision": "STOP_ALL_SINGLE_FORWARD_TRAJECTORY_COMPRESSION",
                "selected_candidate": None,
                "next_allowed_gate": None,
                "reason": "Neither TWA-4 nor any pre-registered single-late checkpoint is Full-safe and HC-Val positive.",
            }
        )
        return result

    if best_eligible is None:
        result.update(
            {
                "gate_pass": True,
                "decision": "REOPEN_TWA4_TO_GATE_F_SEED43_44",
                "selected_candidate": "TWA-4-noBN",
                "next_allowed_gate": "Gate-TWA-F-seed43-44-Full-HCVal",
                "selection_status": "TWA4_RETAINED_NO_ELIGIBLE_SINGLE",
                "promotion_allowed": True,
                "reason": "No Full-safe HC-positive single-late checkpoint exists; ep250 is HC-strong but Full-unsafe.",
            }
        )
        return result

    best_hc = best_eligible.hcval_delta_vs_ohem.mIoU
    twa_hc = twa_hc_delta.mIoU
    best_advantage_over_twa4 = best_hc - twa_hc
    twa_not_worse_than_best_fullsafe_single = twa_eligible and (twa_hc + twa_vs_single_eps >= best_hc)
    result["best_eligible_single"] = {
        "epoch": best_eligible.epoch,
        "hcval_delta_miou": best_eligible.hcval_delta_vs_ohem.mIoU,
        "hcval_delta_fa_ppm": best_eligible.hcval_delta_vs_ohem.FA_ppm,
        "full_delta_miou": best_eligible.full_delta_vs_ohem.mIoU,
    }
    result["best_eligible_by_hcval"] = f"LateSnapshot-ep{best_eligible.epoch}"
    result["twa4_hcval_delta_miou"] = twa_hc
    result["best_hcval_delta_miou"] = best_hc
    result["best_advantage_over_twa4"] = best_advantage_over_twa4
    result["min_hc_miou_switch_margin"] = min_hc_miou_switch_margin
    result["twa4_not_worse_than_best_fullsafe_single_hcval"] = twa_not_worse_than_best_fullsafe_single

    if twa_not_worse_than_best_fullsafe_single:
        result.update(
            {
                "gate_pass": True,
                "decision": "REOPEN_TWA4_TO_GATE_F_SEED43_44",
                "selected_candidate": "TWA-4-noBN",
                "next_allowed_gate": "Gate-TWA-F-seed43-44-Full-HCVal",
                "selection_status": "TWA4_NOT_WORSE_THAN_FULLSAFE_SINGLE",
                "promotion_allowed": True,
                "reason": "TWA-4 is not worse than the best Full-safe single-late control on HC-Val.",
            }
        )
    elif best_advantage_over_twa4 < min_hc_miou_switch_margin:
        result.update(
            {
                "gate_pass": True,
                "decision": "STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD",
                "selected_candidate": None,
                "next_allowed_gate": "STOP_POSTHOC_CHECKPOINT_SELECTION",
                "selection_status": "TIE_OR_NUMERICAL_NOISE_NO_SWITCH",
                "promotion_allowed": False,
                "reason": "Best Full-safe single-late checkpoint advantage over TWA-4 is below the practical HC-Val mIoU switch margin.",
            }
        )
    elif disable_posthoc_single_promotion:
        result.update(
            {
                "gate_pass": True,
                "decision": "STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD",
                "selected_candidate": f"LateSnapshot-ep{best_eligible.epoch}",
                "next_allowed_gate": "STOP_POSTHOC_CHECKPOINT_SELECTION",
                "selection_status": "DIAGNOSTIC_SINGLE_SNAPSHOT_WINNER_NOT_PROMOTED",
                "promotion_allowed": False,
                "reason": "A Full-safe single-late checkpoint wins, but post-hoc single snapshot promotion is disabled.",
            }
        )
    else:
        result.update(
            {
                "gate_pass": True,
                "decision": f"PROCEED_FULLSAFE_SINGLE_EP{best_eligible.epoch}_TO_GATE_LS_B_SEED43_44",
                "selected_candidate": f"LateSnapshot-ep{best_eligible.epoch}",
                "next_allowed_gate": f"Gate-LS-B-ep{best_eligible.epoch}-seed43-44-Full-HCVal",
                "selection_status": "SELECTED_WITH_PRACTICAL_MARGIN",
                "promotion_allowed": True,
                "reason": "A Full-safe single-late checkpoint dominates TWA-4 on HC-Val.",
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate_e_summary", required=True, type=Path)
    parser.add_argument("--ep250_gate_a_summary", required=True, type=Path)
    parser.add_argument("--ohem_full", required=True, type=Path)
    parser.add_argument("--ohem_hcval", required=True, type=Path)
    parser.add_argument("--twa_full", required=True, type=Path)
    parser.add_argument("--twa_hcval", required=True, type=Path)
    parser.add_argument("--snapshot", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min_full_delta_miou", type=float, default=0.0)
    parser.add_argument("--min_full_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_full_delta_pd", type=float, default=0.0)
    parser.add_argument("--max_full_delta_fa_ppm", type=float, default=0.0)
    parser.add_argument("--min_hcval_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_hcval_fa_reduction_ppm", type=float, default=10.0)
    parser.add_argument("--min_hcval_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_hcval_delta_pd", type=float, default=0.0)
    parser.add_argument("--twa_vs_single_eps", type=float, default=1e-12)
    parser.add_argument("--min_hc_miou_switch_margin", type=float, default=0.005)
    parser.add_argument("--disable_posthoc_single_promotion", action="store_true")
    args = parser.parse_args()

    gate_e_summary = load_json(args.gate_e_summary)
    ep250_gate_a_summary = load_json(args.ep250_gate_a_summary)
    context = validate_context(gate_e_summary, ep250_gate_a_summary)
    ohem_full = metrics_from_summary(load_json(args.ohem_full))
    ohem_hcval = metrics_from_summary(load_json(args.ohem_hcval))
    twa_full = metrics_from_summary(load_json(args.twa_full))
    twa_hcval = metrics_from_summary(load_json(args.twa_hcval))

    records: List[SnapshotRecord] = []
    seen_epochs = set()
    for raw in args.snapshot:
        epoch, full_path, hcval_path = parse_snapshot_arg(raw)
        if epoch in seen_epochs:
            raise ValueError(f"Duplicate snapshot epoch: {epoch}")
        seen_epochs.add(epoch)
        records.append(
            build_snapshot_record(
                epoch=epoch,
                full_summary=load_json(full_path),
                hcval_summary=load_json(hcval_path),
                ohem_full=ohem_full,
                ohem_hcval=ohem_hcval,
                min_full_delta_miou=args.min_full_delta_miou,
                min_full_delta_precision=args.min_full_delta_precision,
                min_full_delta_pd=args.min_full_delta_pd,
                max_full_delta_fa_ppm=args.max_full_delta_fa_ppm,
                min_hcval_delta_miou=args.min_hcval_delta_miou,
                min_hcval_fa_reduction_ppm=args.min_hcval_fa_reduction_ppm,
                min_hcval_delta_precision=args.min_hcval_delta_precision,
                min_hcval_delta_pd=args.min_hcval_delta_pd,
            )
        )

    missing = CANDIDATE_EPOCHS.difference(seen_epochs)
    if missing:
        raise ValueError(f"Missing candidate epochs {sorted(missing)}. Provide all of 250,300,350.")

    best_eligible = choose_best_eligible(records)
    decision = decide(
        twa_full=twa_full,
        twa_hcval=twa_hcval,
        ohem_full=ohem_full,
        ohem_hcval=ohem_hcval,
        best_eligible=best_eligible,
        min_full_delta_miou=args.min_full_delta_miou,
        min_full_delta_precision=args.min_full_delta_precision,
        min_full_delta_pd=args.min_full_delta_pd,
        max_full_delta_fa_ppm=args.max_full_delta_fa_ppm,
        min_hcval_delta_miou=args.min_hcval_delta_miou,
        min_hcval_fa_reduction_ppm=args.min_hcval_fa_reduction_ppm,
        min_hcval_delta_precision=args.min_hcval_delta_precision,
        min_hcval_delta_pd=args.min_hcval_delta_pd,
        twa_vs_single_eps=args.twa_vs_single_eps,
        min_hc_miou_switch_margin=args.min_hc_miou_switch_margin,
        disable_posthoc_single_promotion=args.disable_posthoc_single_promotion,
    )

    output = {
        "gate": "Gate-TWA-E2-FSC",
        "gate_name": "Full-Safe Single-Late Control",
        "seed": 42,
        "split_scope": ["Full", "HC-Val"],
        "threshold": 0.5,
        "context": context,
        "allowed_epochs": sorted(ALLOWED_EPOCHS),
        "candidate_epochs": sorted(CANDIDATE_EPOCHS),
        "baseline_epoch": BASELINE_EPOCH,
        "snapshot_records": [asdict(record) for record in records],
        "best_eligible_single_epoch": best_eligible.epoch if best_eligible else None,
        **decision,
        "forbidden_next_actions": [
            "seed43_or_seed44_before_this_gate_passes",
            "HC-Test",
            "blind",
            "external",
            "threshold_search",
            "BN_tuning",
            "new_training",
            "new_model_structure",
            "new_loss",
            "new_verifier_or_suppression_head",
            "new_checkpoint_epochs_outside_250_300_350_400",
            "new_TWA_combinations",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not output["gate_pass"]:
        raise SystemExit(output["decision"])


if __name__ == "__main__":
    main()
