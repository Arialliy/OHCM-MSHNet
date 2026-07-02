#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]
METHOD_ORDER = [
    "MSHNet",
    "MSHNetFocal",
    "MSHNetOHEM",
    "MSHNetTopKNeg",
    "TSR-OHEM-R1",
    "TSR-OHEM-R2",
    "TSR-OHEM",
    "OHCM-light",
    "OHCM-late-inhibition",
    "OHCM-FrozenCalib",
]


def read_summary(path: Path, method: str, split: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics_at_threshold"]
    row = {
        "method": method,
        "dataset": data.get("dataset", ""),
        "seed": str(data.get("seed", "")),
        "split": split,
        "num_images": data.get("num_images", ""),
        "summary": str(path),
    }
    row.update({metric: metrics.get(metric, "") for metric in METRICS})
    return row


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_if_exists(rows: list[dict], path: Path, method: str, split: str) -> None:
    if path.exists():
        rows.append(read_summary(path, method, split))


def collect_rows(run_root: Path, baseline_root: Path, step3_root: Path, dataset: str, seeds: list[str]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        add_if_exists(rows, baseline_root / dataset / f"seed_{seed}" / "exports" / "summary_metrics.json", "MSHNet", "full")
        add_if_exists(
            rows,
            baseline_root / "step2_eval" / "MSHNet" / dataset / f"seed_{seed}" / "hcset" / "summary_metrics.json",
            "MSHNet",
            "hcset",
        )

    for method in ["MSHNetFocal", "MSHNetOHEM", "MSHNetTopKNeg", "OHCM-light"]:
        add_if_exists(rows, step3_root / method / dataset / "seed_42" / "eval_full" / "summary_metrics.json", method, "full")
        add_if_exists(rows, step3_root / method / dataset / "seed_42" / "eval_hcset" / "summary_metrics.json", method, "hcset")

    for path in sorted(run_root.glob(f"*{('/' + dataset + '/seed_*/eval_full/summary_metrics.json')}")):
        method = path.relative_to(run_root).parts[0]
        rows.append(read_summary(path, method, "full"))
    for path in sorted(run_root.glob(f"*{('/' + dataset + '/seed_*/eval_hcset/summary_metrics.json')}")):
        method = path.relative_to(run_root).parts[0]
        rows.append(read_summary(path, method, "hcset"))
    for path in sorted(run_root.glob(f"*{('/' + dataset + '/seed_*/eval_hcval/summary_metrics.json')}")):
        method = path.relative_to(run_root).parts[0]
        rows.append(read_summary(path, method, "hcval"))
    for path in sorted(run_root.glob(f"*{('/' + dataset + '/seed_*/eval_hctest/summary_metrics.json')}")):
        method = path.relative_to(run_root).parts[0]
        rows.append(read_summary(path, method, "hctest"))

    rows.sort(key=lambda r: (r["split"], r["seed"], METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99))
    return rows


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def delta_rows(rows: list[dict]) -> list[dict]:
    by_key = {(row["seed"], row["split"], row["method"]): row for row in rows}
    out = []
    for row in rows:
        if row["method"] == "MSHNet":
            continue
        base = by_key.get((row["seed"], row["split"], "MSHNet"))
        if not base:
            continue
        delta = {
            "method": row["method"],
            "dataset": row["dataset"],
            "seed": row["seed"],
            "split": row["split"],
            "delta_mIoU": to_float(row["mIoU"]) - to_float(base["mIoU"]),
            "delta_Pd": to_float(row["Pd"]) - to_float(base["Pd"]),
            "delta_FA_ppm": to_float(row["FA_ppm"]) - to_float(base["FA_ppm"]),
            "delta_FA_percent": "",
            "delta_Precision": to_float(row["Precision"]) - to_float(base["Precision"]),
            "delta_F1": to_float(row["F1"]) - to_float(base["F1"]),
            "delta_FP_components": to_float(row["FP_components"]) - to_float(base["FP_components"]),
        }
        base_fa = to_float(base["FA_ppm"])
        if base_fa != 0 and not math.isnan(base_fa):
            delta["delta_FA_percent"] = (to_float(row["FA_ppm"]) - base_fa) / base_fa * 100.0
        out.append(delta)
    return out


def mean_std_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dataset"], row["split"])].append(row)
    out = []
    for (method, dataset, split), items in sorted(grouped.items()):
        for metric in METRICS:
            vals = [to_float(row.get(metric, "")) for row in items]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
            out.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "metric": metric,
                    "mean": mean,
                    "std": math.sqrt(var),
                    "n": len(vals),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AAAI P0 paired-seed results.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--baseline_root", default="/home/AAAI/OHCM-MSHNet/results/step0_baseline/20260611_155232")
    parser.add_argument("--step3_root", default="/home/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]
    rows = collect_rows(run_root, Path(args.baseline_root), Path(args.step3_root), args.dataset, seeds)

    fields = ["method", "dataset", "seed", "split", "num_images", *METRICS, "summary"]
    write_csv(run_root / "aaai_p0_paired_table.csv", rows, fields)

    deltas = delta_rows(rows)
    write_csv(
        run_root / "aaai_p0_paired_delta_vs_mshnet.csv",
        deltas,
        [
            "method",
            "dataset",
            "seed",
            "split",
            "delta_mIoU",
            "delta_Pd",
            "delta_FA_ppm",
            "delta_FA_percent",
            "delta_Precision",
            "delta_F1",
            "delta_FP_components",
        ],
    )

    stats = mean_std_rows(rows)
    write_csv(run_root / "aaai_p0_paired_mean_std.csv", stats, ["method", "dataset", "split", "metric", "mean", "std", "n"])

    print(f"Wrote {run_root / 'aaai_p0_paired_table.csv'}")
    print(f"Wrote {run_root / 'aaai_p0_paired_delta_vs_mshnet.csv'}")
    print(f"Wrote {run_root / 'aaai_p0_paired_mean_std.csv'}")


if __name__ == "__main__":
    main()
