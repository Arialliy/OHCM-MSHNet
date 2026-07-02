#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics_at_threshold"]


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize F0-train parity result.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--light_hc", default="/home/ly/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/OHCM-light/NUDT-SIRST/seed_42/eval_hcset/summary_metrics.json")
    parser.add_argument("--f0_root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    light_hc = read_metrics(Path(args.light_hc))
    f0_hc = read_metrics(Path(args.f0_root) / "eval_hcset" / "summary_metrics.json")
    f0_full = read_metrics(Path(args.f0_root) / "eval_full" / "summary_metrics.json")

    rows = []
    for method, split, metrics in [
        ("OHCM-light", "hcset", light_hc),
        ("F0-train", "hcset", f0_hc),
        ("F0-train", "full", f0_full),
    ]:
        row = {"method": method, "split": split}
        row.update(metrics)
        rows.append(row)

    fields = ["method", "split", "threshold", "mIoU", "nIoU", "Pd", "FA", "FA_ppm", "Precision", "Recall", "F1", "FP_components"]
    with (run_root / "f0_train_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    pass_gate = (
        f0_hc["mIoU"] >= 0.603
        and f0_hc["FA_ppm"] <= 225.0
        and f0_hc["Precision"] >= 0.730
    )
    decision = "PASS_F0_TRAIN" if pass_gate else "HOLD_FULL_BRANCH"
    lines = [
        "# F0-Train Parity Report",
        "",
        f"Decision: {decision}",
        "",
        "| Method | HC-mIoU | HC-FA ppm | HC-Precision | HC-Pd |",
        "|---|---:|---:|---:|---:|",
        f"| OHCM-light | {fmt(light_hc['mIoU'])} | {fmt(light_hc['FA_ppm'])} | {fmt(light_hc['Precision'])} | {fmt(light_hc['Pd'])} |",
        f"| F0-train | {fmt(f0_hc['mIoU'])} | {fmt(f0_hc['FA_ppm'])} | {fmt(f0_hc['Precision'])} | {fmt(f0_hc['Pd'])} |",
        "",
        "Gate: HC-mIoU >= 0.603, HC-FA <= 225 ppm, HC-Precision >= 0.730.",
        "",
    ]
    if pass_gate:
        lines.append("F0-train reproduces OHCM-light closely enough for the next train-parity stage.")
    else:
        lines.append("F0-train does not reproduce OHCM-light; keep prototype suspended and do not enter Step5 with OHCM-full.")
    (run_root / "F0_TRAIN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {run_root / 'f0_train_table.csv'}")
    print(f"Wrote {run_root / 'F0_TRAIN_REPORT.md'}")


if __name__ == "__main__":
    main()
