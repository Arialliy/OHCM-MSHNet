#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]
METHOD_ORDER = [
    "MSHNet",
    "MSHNetFocal",
    "MSHNetOHEM",
    "MSHNetTopKNeg",
    "MSHNetSPSOHEM",
    "TCE-OHEM",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_curve(curves: dict, method: str, dataset: str, seed: str, split: str, path: Path) -> None:
    if not path.exists():
        return
    rows = []
    for row in read_csv(path):
        item = {"method": method, "dataset": dataset, "seed": seed, "split": split, "source": str(path)}
        for key, value in row.items():
            item[key] = to_float(value)
        rows.append(item)
    if rows:
        curves[(method, seed, split)] = rows


def collect_curves(run_root: Path, baseline_root: Path, reference_root: Path | None, dataset: str, seeds: list[str]) -> dict:
    curves: dict[tuple[str, str, str], list[dict]] = {}

    for seed in seeds:
        add_curve(curves, "MSHNet", dataset, seed, "full", baseline_root / dataset / f"seed_{seed}" / "exports" / "threshold_curve.csv")
        add_curve(
            curves,
            "MSHNet",
            dataset,
            seed,
            "hcset",
            baseline_root / "step2_eval" / "MSHNet" / dataset / f"seed_{seed}" / "hcset" / "threshold_curve.csv",
        )

    if reference_root is not None:
        for method in ["MSHNetFocal", "MSHNetOHEM", "MSHNetTopKNeg", "MSHNetSPSOHEM", "TCE-OHEM"]:
            add_curve(curves, method, dataset, "42", "full", reference_root / method / dataset / "seed_42" / "eval_full" / "threshold_curve.csv")
            add_curve(curves, method, dataset, "42", "hcset", reference_root / method / dataset / "seed_42" / "eval_hcset" / "threshold_curve.csv")

    for path in sorted(run_root.glob(f"*/{dataset}/seed_*/eval_*/threshold_curve.csv")):
        parts = path.relative_to(run_root).parts
        method = parts[0]
        seed = parts[2].replace("seed_", "")
        split = parts[3].replace("eval_", "")
        add_curve(curves, method, dataset, seed, split, path)

    return curves


def row_at_threshold(curve: list[dict], threshold: float) -> dict | None:
    if not curve:
        return None
    return min(curve, key=lambda row: abs(row.get("threshold", math.nan) - threshold))


def best_match(curve: list[dict], metric: str, target: float) -> dict | None:
    valid = [row for row in curve if not math.isnan(row.get(metric, math.nan))]
    if not valid or math.isnan(target):
        return None
    return min(valid, key=lambda row: (abs(row[metric] - target), row.get("FA_ppm", math.inf)))


def threshold_matched_rows(curves: dict, threshold: float) -> list[dict]:
    out = []
    for (method, seed, split), curve in sorted(curves.items()):
        if method == "MSHNet":
            continue
        target = row_at_threshold(curve, threshold)
        base_curve = curves.get(("MSHNet", seed, split))
        if target is None or not base_curve:
            continue
        for match_metric in ["Pd", "mIoU"]:
            base = best_match(base_curve, match_metric, target.get(match_metric, math.nan))
            if base is None:
                continue
            row = {
                "method": method,
                "dataset": target["dataset"],
                "seed": seed,
                "split": split,
                "match_metric": match_metric,
                "method_threshold": target["threshold"],
                "mshnet_threshold": base["threshold"],
                "matched_abs_error": abs(base.get(match_metric, math.nan) - target.get(match_metric, math.nan)),
            }
            for metric in METRICS:
                row[f"method_{metric}"] = target.get(metric, "")
                row[f"mshnet_{metric}"] = base.get(metric, "")
                row[f"delta_{metric}"] = target.get(metric, math.nan) - base.get(metric, math.nan)
            out.append(row)
    return out


def mean_curve_rows(curves: dict) -> list[dict]:
    grouped = defaultdict(list)
    for (method, seed, split), curve in curves.items():
        for row in curve:
            grouped[(method, split, row["threshold"])].append(row)

    out = []
    for (method, split, threshold), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][1],
            METHOD_ORDER.index(item[0][0]) if item[0][0] in METHOD_ORDER else 99,
            item[0][2],
        ),
    ):
        item = {"method": method, "split": split, "threshold": threshold, "n": len(rows)}
        for metric in METRICS:
            vals = [row.get(metric, math.nan) for row in rows]
            vals = [value for value in vals if not math.isnan(value)]
            if vals:
                item[metric] = sum(vals) / len(vals)
        out.append(item)
    return out


def plot_curves(mean_rows: list[dict], output_dir: Path) -> None:
    by_split_method = defaultdict(list)
    for row in mean_rows:
        by_split_method[(row["split"], row["method"])].append(row)

    for split in sorted({row["split"] for row in mean_rows}):
        methods = [method for method in METHOD_ORDER if (split, method) in by_split_method]
        if not methods:
            continue

        plt.figure(figsize=(7.0, 4.8))
        for method in methods:
            rows = sorted(by_split_method[(split, method)], key=lambda row: row["threshold"])
            x = [row.get("FA_ppm", math.nan) for row in rows]
            y = [row.get("Pd", math.nan) for row in rows]
            plt.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=method)
        plt.xlabel("FA (ppm)")
        plt.ylabel("Pd")
        plt.title(f"Pd-FA Curve ({split})")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{split}_pd_fa_curve.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7.0, 4.8))
        for method in methods:
            rows = sorted(by_split_method[(split, method)], key=lambda row: row["threshold"])
            x = [row.get("Recall", math.nan) for row in rows]
            y = [row.get("Precision", math.nan) for row in rows]
            plt.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=method)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({split})")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{split}_pr_curve.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold-matched and curve analysis for AAAI P0 runs.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--baseline_root", default="/home/AAAI/OHCM-MSHNet/results/step0_baseline/20260611_155232")
    parser.add_argument("--reference_root", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "threshold_matched"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]

    reference_root = Path(args.reference_root) if args.reference_root else None
    curves = collect_curves(run_root, Path(args.baseline_root), reference_root, args.dataset, seeds)
    matched = threshold_matched_rows(curves, args.threshold)
    matched_fields = [
        "method",
        "dataset",
        "seed",
        "split",
        "match_metric",
        "method_threshold",
        "mshnet_threshold",
        "matched_abs_error",
    ]
    for metric in METRICS:
        matched_fields.extend([f"method_{metric}", f"mshnet_{metric}", f"delta_{metric}"])
    write_csv(output_dir / "threshold_matched_vs_mshnet.csv", matched, matched_fields)

    mean_rows = mean_curve_rows(curves)
    mean_fields = ["method", "split", "threshold", "n", *METRICS]
    write_csv(output_dir / "threshold_curve_mean.csv", mean_rows, mean_fields)
    plot_curves(mean_rows, output_dir)

    print(f"Wrote {output_dir / 'threshold_matched_vs_mshnet.csv'}")
    print(f"Wrote {output_dir / 'threshold_curve_mean.csv'}")
    print(f"Wrote curve plots under {output_dir}")


if __name__ == "__main__":
    main()
