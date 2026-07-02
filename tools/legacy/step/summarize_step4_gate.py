#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]


def read_summary(path: Path, method: str, split: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics_at_threshold"]
    row = {
        "method": method,
        "dataset": data.get("dataset", ""),
        "seed": data.get("seed", ""),
        "split": split,
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


def add_baseline_rows(rows: list[dict], baseline_root: Path, dataset: str, seed: str) -> None:
    full = baseline_root / dataset / f"seed_{seed}" / "exports" / "summary_metrics.json"
    hc = baseline_root / "step2_eval" / "MSHNet" / dataset / f"seed_{seed}" / "hcset" / "summary_metrics.json"
    if full.exists():
        rows.append(read_summary(full, "MSHNet", "full"))
    if hc.exists():
        rows.append(read_summary(hc, "MSHNet", "hcset"))


def add_light_rows(rows: list[dict], step3_root: Path, dataset: str, seed: str) -> None:
    base = step3_root / "OHCM-light" / dataset / f"seed_{seed}"
    full = base / "eval_full" / "summary_metrics.json"
    hc = base / "eval_hcset" / "summary_metrics.json"
    if full.exists():
        rows.append(read_summary(full, "OHCM-light", "full"))
    if hc.exists():
        rows.append(read_summary(hc, "OHCM-light", "hcset"))


def add_fp_row(fp_rows: list[dict], path: Path, method: str) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    fp_rows.append(
        {
            "method": method,
            "dataset": data.get("dataset", ""),
            "seed": data.get("seed", ""),
            "false_positive_components": data.get("false_positive_components", ""),
            "target_like_hard_clutter_components": data.get("target_like_hard_clutter_components", ""),
            "hard_clutter_fraction_of_fp": data.get("hard_clutter_fraction_of_fp", ""),
            "summary": str(path),
        }
    )


def row_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(row["method"]), str(row["split"])): row for row in rows}


def as_float(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_report(path: Path, rows: list[dict]) -> None:
    idx = row_index(rows)
    light_hc = idx.get(("OHCM-light", "hcset"))
    full_hc = idx.get(("OHCM-full", "hcset"))
    light_full = idx.get(("OHCM-light", "full"))
    full_full = idx.get(("OHCM-full", "full"))

    light_hc_fa = as_float(light_hc, "FA_ppm")
    full_hc_fa = as_float(full_hc, "FA_ppm")
    light_hc_miou = as_float(light_hc, "mIoU")
    full_hc_miou = as_float(full_hc, "mIoU")
    light_hc_precision = as_float(light_hc, "Precision")
    full_hc_precision = as_float(full_hc, "Precision")
    light_full_miou = as_float(light_full, "mIoU")
    full_full_miou = as_float(full_full, "mIoU")
    light_full_pd = as_float(light_full, "Pd")
    full_full_pd = as_float(full_full, "Pd")

    hc_improved = False
    hc_reasons = []
    if full_hc_fa is not None and light_hc_fa is not None and full_hc_fa < light_hc_fa:
        hc_improved = True
        hc_reasons.append(f"HC-FA decreased {light_hc_fa:.2f} -> {full_hc_fa:.2f} ppm")
    if full_hc_miou is not None and light_hc_miou is not None and full_hc_miou > light_hc_miou:
        hc_improved = True
        hc_reasons.append(f"HC-mIoU increased {light_hc_miou:.4f} -> {full_hc_miou:.4f}")
    if full_hc_precision is not None and light_hc_precision is not None and full_hc_precision > light_hc_precision:
        hc_improved = True
        hc_reasons.append(f"HC-Precision increased {light_hc_precision:.4f} -> {full_hc_precision:.4f}")

    full_not_broken = True
    stability_reasons = []
    if full_full_miou is not None and light_full_miou is not None:
        drop = light_full_miou - full_full_miou
        stability_reasons.append(f"full mIoU {light_full_miou:.4f} -> {full_full_miou:.4f}")
        if drop > 0.02:
            full_not_broken = False
    if full_full_pd is not None and light_full_pd is not None:
        drop = light_full_pd - full_full_pd
        stability_reasons.append(f"full Pd {light_full_pd:.4f} -> {full_full_pd:.4f}")
        if drop > 0.03:
            full_not_broken = False

    decision = "PASS" if hc_improved and full_not_broken else "HOLD"
    recommendation = (
        "Keep prototype for the next Step4 checks."
        if decision == "PASS"
        else "Do not advance prototype to Step5 yet; inspect mining/prototype settings first."
    )

    lines = [
        "# Step4 Gate Report",
        "",
        f"Decision: {decision}",
        "",
        "## OHCM-light vs OHCM-full",
        "",
        f"- HC-mIoU: {fmt(light_hc_miou)} -> {fmt(full_hc_miou)}",
        f"- HC-FA ppm: {fmt(light_hc_fa)} -> {fmt(full_hc_fa)}",
        f"- HC-Precision: {fmt(light_hc_precision)} -> {fmt(full_hc_precision)}",
        f"- Full mIoU: {fmt(light_full_miou)} -> {fmt(full_full_miou)}",
        f"- Full Pd: {fmt(light_full_pd)} -> {fmt(full_full_pd)}",
        "",
        "## Gate Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in (hc_reasons or ["No HC improvement over OHCM-light was detected."]))
    lines.extend(f"- {reason}" for reason in (stability_reasons or ["Full-test stability could not be checked."]))
    lines.extend(["", f"Recommendation: {recommendation}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Step4 OHCM-full prototype gate.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--baseline_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step0_baseline/20260611_155232")
    parser.add_argument("--step3_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate")
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    baseline_root = Path(args.baseline_root)
    step3_root = Path(args.step3_root)
    rows: list[dict] = []

    add_baseline_rows(rows, baseline_root, args.dataset, args.seed)
    add_light_rows(rows, step3_root, args.dataset, args.seed)
    for method_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        method = method_dir.name
        for path in sorted(method_dir.glob("*/seed_*/eval_full/summary_metrics.json")):
            rows.append(read_summary(path, method, "full"))
        for path in sorted(method_dir.glob("*/seed_*/eval_hcset/summary_metrics.json")):
            rows.append(read_summary(path, method, "hcset"))

    fields = ["method", "dataset", "seed", "split", "num_images", *METRICS, "summary"]
    write_csv(run_root / "step4_gate_table.csv", rows, fields)

    fp_rows: list[dict] = []
    add_fp_row(fp_rows, baseline_root / args.dataset / f"seed_{args.seed}" / "step1" / "step1_summary.json", "MSHNet")
    add_fp_row(fp_rows, step3_root / "OHCM-light" / args.dataset / f"seed_{args.seed}" / "fp_analysis" / "step1_summary.json", "OHCM-light")
    for path in sorted(run_root.glob("*/*/seed_*/fp_analysis/step1_summary.json")):
        add_fp_row(fp_rows, path, path.relative_to(run_root).parts[0])
    write_csv(
        run_root / "step4_fp_table.csv",
        fp_rows,
        ["method", "dataset", "seed", "false_positive_components", "target_like_hard_clutter_components", "hard_clutter_fraction_of_fp", "summary"],
    )
    write_report(run_root / "STEP4_GATE_REPORT.md", rows)

    print(f"Wrote {run_root / 'step4_gate_table.csv'}")
    print(f"Wrote {run_root / 'step4_fp_table.csv'}")
    print(f"Wrote {run_root / 'STEP4_GATE_REPORT.md'}")


if __name__ == "__main__":
    main()
