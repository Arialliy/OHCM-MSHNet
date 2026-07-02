#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ["mIoU", "Precision", "Pd", "FA_ppm"]
METRIC_ALIASES = {
    "mIoU": ("mIoU", "miou", "mean_iou", "mean_IoU"),
    "Precision": ("Precision", "precision", "Prec", "prec"),
    "Pd": ("Pd", "pd", "PD", "Recall", "recall"),
    "FA_ppm": ("FA_ppm", "fa_ppm", "FAppm", "FAppm", "FA", "fa"),
    "FP_components": ("FP_components", "fp_components", "FP", "fp"),
}
NESTED_METRIC_KEYS = (
    "metrics_at_threshold",
    "metrics",
    "official_metrics",
    "summary_metrics",
    "aggregate",
    "overall",
)


def resolve_existing_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
        text = str(path)
        if text.startswith("/home/ly/AAAI/"):
            candidates.append(Path("/home/AAAI") / text[len("/home/ly/AAAI/"):])
        if text.startswith("/home/AAAI/"):
            candidates.append(Path("/home/ly/AAAI") / text[len("/home/AAAI/"):])
    else:
        candidates.extend([PROJECT_ROOT / path, Path.cwd() / path, path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def load_json(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_existing_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {resolved}")
    return data


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def lookup_nested(summary: Dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    for nested_key in NESTED_METRIC_KEYS:
        nested = summary.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def get_metric(summary: Dict[str, Any], key: str) -> float:
    for alias in METRIC_ALIASES[key]:
        value = lookup_nested(summary, alias)
        if value is not None:
            return float(value)
    raise KeyError(f"Metric {key} not found. Available top-level keys: {sorted(summary.keys())}")


def delta_row(split: str, seed: int, ohem_path: str, tce4_path: str) -> Dict[str, Any]:
    ohem_resolved = resolve_existing_path(ohem_path)
    tce4_resolved = resolve_existing_path(tce4_path)
    ohem = load_json(ohem_resolved)
    tce4 = load_json(tce4_resolved)
    row: Dict[str, Any] = {
        "split": split,
        "seed": seed,
        "ohem_path": str(ohem_resolved),
        "tce4_path": str(tce4_resolved),
        "ohem": {},
        "tce4": {},
        "delta": {},
    }
    for metric in METRICS:
        ohem_value = get_metric(ohem, metric)
        tce4_value = get_metric(tce4, metric)
        row["ohem"][metric] = ohem_value
        row["tce4"][metric] = tce4_value
        row["delta"][metric] = tce4_value - ohem_value

    try:
        ohem_value = get_metric(ohem, "FP_components")
        tce4_value = get_metric(tce4, "FP_components")
        row["ohem"]["FP_components"] = ohem_value
        row["tce4"]["FP_components"] = tce4_value
        row["delta"]["FP_components"] = tce4_value - ohem_value
        row["fp_components_available"] = True
    except Exception:
        row["fp_components_available"] = False
    return row


def summarize_metric(rows: Iterable[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    values = [row["delta"][metric] for row in rows]
    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def summarize_split(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    delta_metrics = {metric: summarize_metric(rows, metric) for metric in METRICS}
    out: Dict[str, Any] = {
        "num_seeds": len(rows),
        "mean_delta_mIoU": delta_metrics["mIoU"]["mean"],
        "mean_delta_Precision": delta_metrics["Precision"]["mean"],
        "mean_delta_Pd": delta_metrics["Pd"]["mean"],
        "mean_delta_FA_ppm": delta_metrics["FA_ppm"]["mean"],
        "min_delta_mIoU": delta_metrics["mIoU"]["min"],
        "min_delta_Precision": delta_metrics["Precision"]["min"],
        "min_delta_Pd": delta_metrics["Pd"]["min"],
        "max_delta_FA_ppm": delta_metrics["FA_ppm"]["max"],
        "delta_metrics": delta_metrics,
        "per_seed": rows,
    }

    strong = (
        out["mean_delta_mIoU"] >= args.strong_min_mean_delta_miou
        and out["mean_delta_FA_ppm"] <= -args.strong_min_mean_fa_reduction
        and out["mean_delta_Precision"] >= args.min_mean_delta_precision
        and out["min_delta_Pd"] >= args.min_delta_pd
        and out["min_delta_mIoU"] >= args.min_seed_delta_miou
        and out["max_delta_FA_ppm"] <= args.max_seed_fa_increase
    )
    mixed = (
        out["mean_delta_mIoU"] >= args.min_mean_delta_miou
        and out["mean_delta_FA_ppm"] <= args.max_mean_delta_fa
        and out["min_delta_Pd"] >= args.min_delta_pd
    )

    if strong:
        out["split_verdict"] = "F3_PASS_STRONG"
        out["split_pass"] = True
    elif mixed:
        out["split_verdict"] = "F3_PASS_MIXED_REPORTABLE"
        out["split_pass"] = True
    else:
        out["split_verdict"] = "F3_FAIL_NO_REDESIGN"
        out["split_pass"] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-TCE-F3 blind/external once report checker.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--once_lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strong_min_mean_delta_miou", type=float, default=0.001)
    parser.add_argument("--strong_min_mean_fa_reduction", type=float, default=5.0)
    parser.add_argument("--min_mean_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_delta_pd", type=float, default=0.0)
    parser.add_argument("--min_seed_delta_miou", type=float, default=-0.005)
    parser.add_argument("--max_seed_fa_increase", type=float, default=10.0)
    parser.add_argument("--min_mean_delta_miou", type=float, default=0.0)
    parser.add_argument("--max_mean_delta_fa", type=float, default=0.0)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    lock_path = Path(args.once_lock)
    if not lock_path.exists():
        raise SystemExit(f"F3 once lock missing: {lock_path}")
    lock = load_json(lock_path)
    if lock.get("status") == "COMPLETED":
        raise SystemExit("F3 once lock already completed. Do not recompute final report.")

    splits = manifest["splits"]
    seeds = [int(seed) for seed in manifest.get("seeds", [42, 43, 44])]
    all_split_summaries: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []

    for split in splits:
        rows: List[Dict[str, Any]] = []
        for seed in seeds:
            pair = manifest["summary_paths"][split][str(seed)]
            ohem_path = resolve_existing_path(pair["ohem"])
            tce4_path = resolve_existing_path(pair["tce4"])
            if not ohem_path.exists():
                raise SystemExit(f"Missing OHEM summary: split={split}, seed={seed}, path={ohem_path}")
            if not tce4_path.exists():
                raise SystemExit(f"Missing TCE4 summary: split={split}, seed={seed}, path={tce4_path}")
            row = delta_row(split, seed, str(ohem_path), str(tce4_path))
            rows.append(row)
            all_rows.append(row)
        all_split_summaries[split] = summarize_split(rows, args)

    all_pass = all(item["split_pass"] for item in all_split_summaries.values())
    all_strong = all(item["split_verdict"] == "F3_PASS_STRONG" for item in all_split_summaries.values())
    if all_strong:
        verdict = "F3_PASS_STRONG"
        gate_pass = True
        next_action = "WRITE_AAAI_MAIN_RESULTS_WITH_4X_COST"
    elif all_pass:
        verdict = "F3_PASS_MIXED_REPORTABLE"
        gate_pass = True
        next_action = "WRITE_AAAI_RESULTS_CONSERVATIVELY_WITH_4X_COST"
    else:
        verdict = "F3_FAIL_NO_REDESIGN"
        gate_pass = False
        next_action = "STOP_TCE4_AS_FINAL_AAAI_MAIN_METHOD"

    global_delta_metrics = {metric: summarize_metric(all_rows, metric) for metric in METRICS}
    report = {
        "gate": "Gate-TCE-F3-blind-external-once",
        "candidate": manifest.get("candidate"),
        "baseline": manifest.get("baseline"),
        "threshold": manifest.get("threshold"),
        "seeds": seeds,
        "tce_epochs": manifest.get("tce_epochs"),
        "splits": splits,
        "split_datasets": manifest.get("split_datasets", {}),
        "gate_pass": gate_pass,
        "verdict": verdict,
        "split_summaries": all_split_summaries,
        "global_delta_metrics": global_delta_metrics,
        "next_action": next_action,
        "forbidden_after_f3": [
            "rerun_blind_external",
            "threshold_search",
            "seed_search",
            "checkpoint_search",
            "new_training",
            "method_change_after_external",
        ],
    }
    save_json(report, args.output)

    lock["status"] = "COMPLETED"
    lock["completed_utc"] = datetime.now(timezone.utc).isoformat()
    lock["final_report"] = str(args.output)
    lock["verdict"] = verdict
    save_json(lock, lock_path)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not gate_pass:
        raise SystemExit("Gate-TCE-F3 failed. Stop; do not redesign or rerun after external results.")


if __name__ == "__main__":
    main()
