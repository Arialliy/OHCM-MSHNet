#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple


METRICS = ["mIoU", "Precision", "Pd", "FA_ppm"]
METRIC_ALIASES = {
    "mIoU": ("mIoU", "miou", "mean_iou", "mean_IoU"),
    "Precision": ("Precision", "precision", "prec"),
    "Pd": ("Pd", "pd", "PD", "recall"),
    "FA_ppm": ("FA_ppm", "fa_ppm", "FAppm", "FA", "fa"),
}
NESTED_METRIC_KEYS = (
    "metrics_at_threshold",
    "metrics",
    "official_metrics",
    "summary_metrics",
    "aggregate",
    "overall",
)
DEFAULT_EXPECTED_SEEDS = [42, 43, 44]
DEFAULT_EXPECTED_SPLITS = ["full", "hcval", "hctest"]


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_existing_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    text = str(path)
    candidates = []
    if text.startswith("/home/ly/AAAI/"):
        candidates.append(Path("/home/AAAI") / text[len("/home/ly/AAAI/"):])
    if text.startswith("/home/AAAI/"):
        candidates.append(Path("/home/ly/AAAI") / text[len("/home/AAAI/"):])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


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


def summarize(values: List[float]) -> Dict[str, Any]:
    return {
        "mean": mean(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def parse_int_list(items: Iterable[Any]) -> List[int]:
    return [int(item) for item in items]


def parse_str_list(items: Iterable[Any]) -> List[str]:
    return [str(item).lower() for item in items]


def expected_pairs(manifest: Dict[str, Any]) -> List[Tuple[int, str]]:
    seeds = parse_int_list(manifest.get("expected_seeds", DEFAULT_EXPECTED_SEEDS))
    splits = parse_str_list(manifest.get("expected_splits", DEFAULT_EXPECTED_SPLITS))
    return [(seed, split) for seed in seeds for split in splits]


def pair_key(item: Dict[str, Any]) -> Tuple[int, str]:
    return int(item["seed"]), str(item["split"]).lower()


def row_passes(split: str, delta: Dict[str, float], min_hc_miou_delta: float, min_hc_fa_reduction: float) -> bool:
    if split == "full":
        return (
            delta["mIoU"] >= 0.0
            and delta["Precision"] >= 0.0
            and delta["Pd"] >= 0.0
            and delta["FA_ppm"] <= 0.0
        )
    return (
        delta["mIoU"] >= min_hc_miou_delta
        and delta["Precision"] >= 0.0
        and delta["Pd"] >= 0.0
        and delta["FA_ppm"] <= -min_hc_fa_reduction
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen TCE-4 internal evidence.")
    parser.add_argument("--manifest", required=True, help="JSON manifest listing paired OHEM/TCE summaries.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_hc_miou_delta", type=float, default=0.005)
    parser.add_argument("--min_hc_fa_reduction", type=float, default=10.0)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    pairs = manifest.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("manifest['pairs'] must be a list")

    expected = expected_pairs(manifest)
    present = {pair_key(item) for item in pairs}
    missing_manifest_pairs = [
        {"seed": seed, "split": split}
        for seed, split in expected
        if (seed, split) not in present
    ]

    rows: List[Dict[str, Any]] = []
    missing_summary_files: List[Dict[str, Any]] = []
    for item in pairs:
        seed, split = pair_key(item)
        ohem_path = resolve_existing_path(item["ohem_summary"])
        tce_path = resolve_existing_path(item["tce_summary"])
        missing = []
        if not ohem_path.exists():
            missing.append("ohem_summary")
        if not tce_path.exists():
            missing.append("tce_summary")
        if missing:
            missing_summary_files.append(
                {
                    "seed": seed,
                    "split": split,
                    "missing": missing,
                    "ohem_summary": str(ohem_path),
                    "tce_summary": str(tce_path),
                }
            )
            continue

        ohem = load_json(ohem_path)
        tce = load_json(tce_path)
        row: Dict[str, Any] = {
            "seed": seed,
            "split": split,
            "ohem_summary": str(ohem_path),
            "tce_summary": str(tce_path),
            "delta": {},
        }
        for metric in METRICS:
            row[f"ohem_{metric}"] = get_metric(ohem, metric)
            row[f"tce_{metric}"] = get_metric(tce, metric)
            row["delta"][metric] = row[f"tce_{metric}"] - row[f"ohem_{metric}"]
        row["pass"] = row_passes(split, row["delta"], args.min_hc_miou_delta, args.min_hc_fa_reduction)
        rows.append(row)

    by_split: Dict[str, Any] = {}
    for split in sorted({split for _, split in expected}):
        split_rows = [row for row in rows if row["split"] == split]
        delta_summary = {metric: summarize([row["delta"][metric] for row in split_rows]) for metric in METRICS}
        by_split[split] = {
            "num_seeds": len(split_rows),
            "expected_num_seeds": len({seed for seed, exp_split in expected if exp_split == split}),
            "num_pass": int(sum(1 for row in split_rows if row["pass"])),
            "all_pass": bool(split_rows) and all(row["pass"] for row in split_rows),
            "delta_summary": delta_summary,
            "rows": split_rows,
        }

    manifest_complete = not missing_manifest_pairs and not missing_summary_files
    gate_pass = manifest_complete and all(item["all_pass"] for item in by_split.values())
    result = {
        "gate": "Gate-TCE-F1-internal-evidence-aggregation",
        "method": manifest.get("method", "TCE-4-OHEM"),
        "checkpoints": manifest.get("checkpoints", [250, 300, 350, 400]),
        "threshold": manifest.get("threshold", 0.5),
        "aggregation": manifest.get("aggregation", "existing_official_tce_aggregation"),
        "gate_pass": gate_pass,
        "manifest_complete": manifest_complete,
        "missing_manifest_pairs": missing_manifest_pairs,
        "missing_summary_files": missing_summary_files,
        "thresholds": {
            "min_hc_miou_delta": args.min_hc_miou_delta,
            "min_hc_fa_reduction": args.min_hc_fa_reduction,
        },
        "by_split": by_split,
        "rows": rows,
        "decision": "PROCEED_TO_TCE_THRESHOLD_COMPONENT_REPORT" if gate_pass else "TCE_INTERNAL_EVIDENCE_PARTIAL_OR_FAIL",
        "next_allowed_gate": "Gate-TCE-F2-threshold-component-report" if gate_pass else "STOP_OR_REPORT_LIMITATION",
        "forbidden_next_actions": [
            "seed_search",
            "checkpoint_search",
            "threshold_search",
            "BN_recalibration_tuning",
            "TCSR_training",
            "new_loss",
            "new_model_structure",
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not gate_pass:
        raise SystemExit("Gate-TCE-F1 failed or partial. Do not select seeds/checkpoints to rescue.")


if __name__ == "__main__":
    main()
