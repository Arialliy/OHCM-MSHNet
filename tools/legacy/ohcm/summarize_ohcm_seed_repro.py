#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]


def read_summary(path: Path, method: str, split: str, seed: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics_at_threshold"]
    row = {
        "method": method,
        "seed": seed,
        "split": split,
        "dataset": data.get("dataset", ""),
        "num_images": data.get("num_images", ""),
        "summary": str(path),
    }
    row.update({metric: metrics.get(metric, "") for metric in METRICS})
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(rows: list[dict], split: str) -> list[dict]:
    out = []
    split_rows = [row for row in rows if row["split"] == split]
    for metric in METRICS:
        values = [float(row[metric]) for row in split_rows if row.get(metric, "") != ""]
        if not values:
            continue
        out.append(
            {
                "split": split,
                "metric": metric,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OHCM three-seed reproducibility.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows = []
    for seed in seeds:
        base = run_root / "OHCM" / args.dataset / f"seed_{seed}"
        full = base / "eval_full" / "summary_metrics.json"
        hc = base / "eval_hcset" / "summary_metrics.json"
        if full.exists():
            rows.append(read_summary(full, "OHCM", "full", seed))
        if hc.exists():
            rows.append(read_summary(hc, "OHCM", "hcset", seed))

    fields = ["method", "seed", "split", "dataset", "num_images", *METRICS, "summary"]
    write_csv(run_root / "ohcm_three_seed_table.csv", rows, fields)

    stat_rows = mean_std(rows, "full") + mean_std(rows, "hcset")
    write_csv(run_root / "ohcm_three_seed_mean_std.csv", stat_rows, ["split", "metric", "mean", "std", "n"])

    stat = {(row["split"], row["metric"]): row for row in stat_rows}
    hc_miou = stat.get(("hcset", "mIoU"), {}).get("mean")
    hc_fa = stat.get(("hcset", "FA_ppm"), {}).get("mean")
    hc_precision = stat.get(("hcset", "Precision"), {}).get("mean")
    complete = len([row for row in rows if row["split"] == "hcset"]) == len(seeds)
    pass_gate = (
        complete
        and hc_miou is not None
        and hc_fa is not None
        and hc_precision is not None
        and float(hc_miou) >= 0.59
        and float(hc_fa) <= 240.0
        and float(hc_precision) >= 0.72
    )
    decision = "PASS_OHCM_THREE_SEED" if pass_gate else ("INCOMPLETE" if not complete else "HOLD_OHCM_STABILITY")

    def fmt(value):
        return "NA" if value is None else f"{float(value):.4f}"

    lines = [
        "# OHCM Three-Seed Reproducibility",
        "",
        f"Decision: {decision}",
        "",
        "| Method | Seed | Full mIoU | Full FA ppm | Full Precision | HC-mIoU | HC-FA ppm | HC-Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["split"]] = row
    for seed in seeds:
        full = by_seed.get(seed, {}).get("full", {})
        hc = by_seed.get(seed, {}).get("hcset", {})
        lines.append(
            f"| OHCM | {seed} | {fmt(full.get('mIoU'))} | {fmt(full.get('FA_ppm'))} | {fmt(full.get('Precision'))} | "
            f"{fmt(hc.get('mIoU'))} | {fmt(hc.get('FA_ppm'))} | {fmt(hc.get('Precision'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- HC-mIoU mean >= 0.59: observed {fmt(hc_miou)}",
            f"- HC-FA mean <= 240 ppm: observed {fmt(hc_fa)}",
            f"- HC-Precision mean >= 0.72: observed {fmt(hc_precision)}",
            "",
            "If this gate passes, proceed to Step5 with OHCM as the final method. If it fails, tune OHCM only and keep prototype stopped.",
            "",
        ]
    )
    (run_root / "OHCM_THREE_SEED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {run_root / 'ohcm_three_seed_table.csv'}")
    print(f"Wrote {run_root / 'ohcm_three_seed_mean_std.csv'}")
    print(f"Wrote {run_root / 'OHCM_THREE_SEED_REPORT.md'}")


if __name__ == "__main__":
    main()
