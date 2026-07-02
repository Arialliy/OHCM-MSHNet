#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]


def read_summary(path: Path, seed: int, epoch: int, split: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics_at_threshold"]
    row = {
        "method": data.get("method", "OHCM"),
        "seed": seed,
        "epoch": epoch,
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


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def best_hc_row(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            as_float(row, "mIoU"),
            -as_float(row, "FA_ppm"),
            as_float(row, "Precision"),
        ),
        reverse=True,
    )[0]


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


def fmt(value) -> str:
    return "NA" if value is None or value == "" else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OHCM checkpoint sweep for early-stop diagnosis.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", default="50,100,150,200,250,300,350,400")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    epochs = [int(item) for item in args.epochs.split(",") if item.strip()]
    rows = []
    for seed in seeds:
        for epoch in epochs:
            base = run_root / "OHCM" / args.dataset / f"seed_{seed}" / f"epoch_{epoch}"
            full = base / "eval_full" / "summary_metrics.json"
            hc = base / "eval_hcset" / "summary_metrics.json"
            if full.exists():
                rows.append(read_summary(full, seed, epoch, "full"))
            if hc.exists():
                rows.append(read_summary(hc, seed, epoch, "hcset"))

    fields = ["method", "seed", "epoch", "split", "dataset", "num_images", *METRICS, "summary"]
    write_csv(run_root / "ohcm_checkpoint_sweep_table.csv", rows, fields)

    best_rows = []
    for seed in seeds:
        hc_rows = [row for row in rows if row["seed"] == seed and row["split"] == "hcset"]
        if not hc_rows:
            continue
        best_hc = best_hc_row(hc_rows)
        best_rows.append(best_hc)
        full_match = [
            row
            for row in rows
            if row["seed"] == seed and row["epoch"] == best_hc["epoch"] and row["split"] == "full"
        ]
        if full_match:
            best_rows.append(full_match[0])

    write_csv(run_root / "ohcm_checkpoint_sweep_best_by_hc.csv", best_rows, fields)
    stat_rows = mean_std(best_rows, "full") + mean_std(best_rows, "hcset")
    write_csv(run_root / "ohcm_checkpoint_sweep_best_mean_std.csv", stat_rows, ["split", "metric", "mean", "std", "n"])

    stat = {(row["split"], row["metric"]): row for row in stat_rows}
    hc_miou = stat.get(("hcset", "mIoU"), {}).get("mean")
    hc_fa = stat.get(("hcset", "FA_ppm"), {}).get("mean")
    hc_precision = stat.get(("hcset", "Precision"), {}).get("mean")
    complete = len([row for row in best_rows if row["split"] == "hcset"]) == len(seeds)
    pass_gate = (
        complete
        and hc_miou is not None
        and hc_fa is not None
        and hc_precision is not None
        and float(hc_miou) >= 0.59
        and float(hc_fa) <= 240.0
        and float(hc_precision) >= 0.72
    )
    decision = "CHECKPOINT_RULE_CAN_RESCUE" if pass_gate else ("INCOMPLETE" if not complete else "CHECKPOINT_RULE_NOT_ENOUGH")

    by_seed = {}
    for row in best_rows:
        by_seed.setdefault(row["seed"], {})[row["split"]] = row
    lines = [
        "# OHCM Checkpoint Sweep",
        "",
        f"Decision: {decision}",
        "",
        "This is a stability diagnostic for OHCM-light/OHCM only. It does not re-enable OHCM-full or prototype.",
        "",
        "| Seed | Selected Epoch | Full mIoU | Full FA ppm | Full Precision | HC-mIoU | HC-FA ppm | HC-Precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        full = by_seed.get(seed, {}).get("full", {})
        hc = by_seed.get(seed, {}).get("hcset", {})
        epoch = hc.get("epoch", "NA")
        lines.append(
            f"| {seed} | {epoch} | {fmt(full.get('mIoU'))} | {fmt(full.get('FA_ppm'))} | "
            f"{fmt(full.get('Precision'))} | {fmt(hc.get('mIoU'))} | {fmt(hc.get('FA_ppm'))} | "
            f"{fmt(hc.get('Precision'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate on Best-HC Checkpoints",
            "",
            f"- HC-mIoU mean >= 0.59: observed {fmt(hc_miou)}",
            f"- HC-FA mean <= 240 ppm: observed {fmt(hc_fa)}",
            f"- HC-Precision mean >= 0.72: observed {fmt(hc_precision)}",
            "",
        ]
    )
    (run_root / "OHCM_CHECKPOINT_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {run_root / 'ohcm_checkpoint_sweep_table.csv'}")
    print(f"Wrote {run_root / 'ohcm_checkpoint_sweep_best_by_hc.csv'}")
    print(f"Wrote {run_root / 'ohcm_checkpoint_sweep_best_mean_std.csv'}")
    print(f"Wrote {run_root / 'OHCM_CHECKPOINT_SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()
