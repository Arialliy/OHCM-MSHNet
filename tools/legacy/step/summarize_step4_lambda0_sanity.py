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


def by_method_split(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["method"], row["split"]): row for row in rows}


def f(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OHCM-full lambda_proto=0 sanity check.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--step3_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate")
    parser.add_argument("--proto_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step4_ohcm_full_proto/20260614_step4_proto_3ca")
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    step3_root = Path(args.step3_root)
    proto_root = Path(args.proto_root)
    dataset = args.dataset
    seed_dir = f"seed_{args.seed}"

    rows = []
    sources = [
        ("OHCM-light", step3_root / "OHCM-light" / dataset / seed_dir),
        ("OHCM-full, lambda_proto=0", run_root / "OHCM-full-lambda0" / dataset / seed_dir),
        ("OHCM-full, lambda_proto>0", proto_root / "OHCM-full" / dataset / seed_dir),
    ]
    for method, base in sources:
        full = base / "eval_full" / "summary_metrics.json"
        hc = base / "eval_hcset" / "summary_metrics.json"
        if full.exists():
            rows.append(read_summary(full, method, "full"))
        if hc.exists():
            rows.append(read_summary(hc, method, "hcset"))

    fields = ["method", "dataset", "seed", "split", "num_images", *METRICS, "summary"]
    write_csv(run_root / "lambda0_sanity_table.csv", rows, fields)

    idx = by_method_split(rows)
    light = idx.get(("OHCM-light", "hcset"))
    lam0 = idx.get(("OHCM-full, lambda_proto=0", "hcset"))
    proto = idx.get(("OHCM-full, lambda_proto>0", "hcset"))

    light_miou, lam0_miou = f(light, "mIoU"), f(lam0, "mIoU")
    light_fa, lam0_fa = f(light, "FA_ppm"), f(lam0, "FA_ppm")
    light_prec, lam0_prec = f(light, "Precision"), f(lam0, "Precision")

    if lam0 is None:
        decision = "INCOMPLETE"
        interpretation = "lambda_proto=0 results are missing."
    else:
        miou_close = light_miou is not None and lam0_miou is not None and (light_miou - lam0_miou) <= 0.02
        fa_close = light_fa is not None and lam0_fa is not None and lam0_fa <= light_fa * 1.15
        prec_close = light_prec is not None and lam0_prec is not None and (light_prec - lam0_prec) <= 0.03
        if miou_close and fa_close and prec_close:
            decision = "PROTO_LOSS_SUSPECT"
            interpretation = "lambda_proto=0 is close to OHCM-light; the degradation mainly points to prototype mining/loss."
        else:
            decision = "FULL_STRUCTURE_SUSPECT"
            interpretation = "lambda_proto=0 is still clearly worse than OHCM-light; inspect full-branch structure/config before prototype tuning."

    lines = [
        "# Step4 Lambda0 Sanity Check",
        "",
        f"Decision: {decision}",
        "",
        "| Method | HC-mIoU | HC-FA ppm | HC-Precision |",
        "|---|---:|---:|---:|",
    ]
    for method in ["OHCM-light", "OHCM-full, lambda_proto=0", "OHCM-full, lambda_proto>0"]:
        row = idx.get((method, "hcset"))
        lines.append(f"| {method} | {fmt(f(row, 'mIoU'))} | {fmt(f(row, 'FA_ppm'))} | {fmt(f(row, 'Precision'))} |")
    lines.extend(["", interpretation, ""])
    (run_root / "LAMBDA0_SANITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {run_root / 'lambda0_sanity_table.csv'}")
    print(f"Wrote {run_root / 'LAMBDA0_SANITY_REPORT.md'}")
    if proto is None:
        print("Warning: lambda_proto>0 comparison row is missing.")


if __name__ == "__main__":
    main()
