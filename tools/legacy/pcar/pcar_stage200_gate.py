#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("R3-A", "R3-B", "PCAR", "PCAR-low")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict | None, key: str) -> float | None:
    if not summary:
        return None
    value = summary.get("metrics_at_threshold", {}).get(key)
    return float(value) if value is not None else None


def read_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
        return rows


def matched_pd_fa(curve_path: Path, target_pd: float) -> float | None:
    rows = read_curve(curve_path)
    feasible = [row for row in rows if row.get("Pd", -1.0) >= target_pd]
    if not feasible:
        return None
    return min(row["FA_ppm"] for row in feasible)


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def pass_fail(value: bool | None) -> str:
    if value is None:
        return "PENDING"
    return "PASS" if value else "FAIL"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["method"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate PCAR-OHEM stage-200 runs against the OHEM baseline.")
    parser.add_argument("--run_root", default="results/pcar_ohem/20260623_pcar_stage200")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--baseline_full", default="results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/NUDT-SIRST/seed_42/eval_full/summary_metrics.json")
    parser.add_argument("--baseline_hcval", default="results/aaai_p0_paired/20260617_aaai_p0_paired/MSHNetOHEM/NUDT-SIRST/seed_42/eval_hcval/summary_metrics.json")
    parser.add_argument("--baseline_dev_hc", default="results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/NUDT-SIRST/seed_42/eval_hcset/summary_metrics.json")
    parser.add_argument("--bank_summary", default="results/pcar_ohem/20260623_pcar/bank/persistent_clutter_bank_summary.json")
    parser.add_argument("--output_dir", default="results/pcar_ohem/20260623_pcar_stage200/gate")
    args = parser.parse_args()

    baseline_full = read_json(Path(args.baseline_full))
    baseline_hcval = read_json(Path(args.baseline_hcval))
    baseline_dev_hc = read_json(Path(args.baseline_dev_hc))
    bank_summary = read_json(Path(args.bank_summary)) or {}

    full_miou_min = metric(baseline_full, "mIoU") - 0.003
    full_pd_min = metric(baseline_full, "Pd") - 0.005
    hcval_fa_max = metric(baseline_hcval, "FA_ppm")
    hcval_precision_min = metric(baseline_hcval, "Precision")
    hcval_pd_target = metric(baseline_hcval, "Pd")
    bank_activation = float(bank_summary.get("candidate_activation_ratio", bank_summary.get("active_image_ratio", 0.0)))
    bank_leakage = float(bank_summary.get("gt_leakage", bank_summary.get("gt_leak_ratio", 1.0)))

    run_root = Path(args.run_root)
    rows = []
    for method in args.methods:
        base = run_root / method / args.dataset_name / f"seed_{args.seed}"
        full = read_json(base / "eval_full" / "summary_metrics.json")
        hcval = read_json(base / "eval_hcval" / "summary_metrics.json")
        dev_hc = read_json(base / "eval_dev_hc" / "summary_metrics.json")
        hcval_curve = base / "eval_hcval" / "threshold_curve.csv"

        method_matched_pd_fa = matched_pd_fa(hcval_curve, hcval_pd_target) if hcval_pd_target is not None else None
        gate_full_miou = None if full is None else metric(full, "mIoU") >= full_miou_min
        gate_full_pd = None if full is None else metric(full, "Pd") >= full_pd_min
        gate_hcval_fa = None if hcval is None else metric(hcval, "FA_ppm") <= hcval_fa_max
        gate_hcval_precision = None if hcval is None else metric(hcval, "Precision") >= hcval_precision_min
        gate_matched_pd_fa = None if method_matched_pd_fa is None else method_matched_pd_fa <= hcval_fa_max
        persistent_method = method in {"R3-B", "PCAR", "PCAR-low"}
        gate_bank_activation = (bank_activation >= 0.20) if persistent_method else None
        gate_bank_leakage = (bank_leakage == 0.0) if persistent_method else None
        gates = [
            gate_full_miou,
            gate_full_pd,
            gate_hcval_fa,
            gate_hcval_precision,
            gate_matched_pd_fa,
            gate_bank_activation,
            gate_bank_leakage,
        ]
        completed_gates = [item for item in gates if item is not None]
        overall = None if not completed_gates or any(item is None for item in gates[:5]) else all(completed_gates)

        rows.append({
            "method": method,
            "status": "DONE" if full and hcval and dev_hc else "PENDING",
            "full_mIoU": fmt(metric(full, "mIoU")),
            "full_Pd": fmt(metric(full, "Pd")),
            "hcval_FA_ppm": fmt(metric(hcval, "FA_ppm")),
            "hcval_Precision": fmt(metric(hcval, "Precision")),
            "hcval_matched_Pd_FA_ppm": fmt(method_matched_pd_fa),
            "dev_hc_FA_ppm": fmt(metric(dev_hc, "FA_ppm")),
            "dev_hc_Pd": fmt(metric(dev_hc, "Pd")),
            "bank_activation": fmt(bank_activation),
            "bank_gt_leakage": fmt(bank_leakage),
            "gate_full_mIoU": pass_fail(gate_full_miou),
            "gate_full_Pd": pass_fail(gate_full_pd),
            "gate_hcval_FA": pass_fail(gate_hcval_fa),
            "gate_hcval_Precision": pass_fail(gate_hcval_precision),
            "gate_matched_Pd_FA": pass_fail(gate_matched_pd_fa),
            "gate_bank_activation": pass_fail(gate_bank_activation),
            "gate_bank_leakage": pass_fail(gate_bank_leakage),
            "continue_to_400": pass_fail(overall),
        })

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "stage200_gate_table.csv", rows)
    lines = [
        "# PCAR Stage-200 Gate Report",
        "",
        f"- Full mIoU gate: >= {full_miou_min:.6f}",
        f"- Full Pd gate: >= {full_pd_min:.6f}",
        f"- HC-Val FA gate: <= {hcval_fa_max:.6f} ppm",
        f"- HC-Val Precision gate: >= {hcval_precision_min:.6f}",
        f"- HC-Val matched-Pd FA gate: <= {hcval_fa_max:.6f} ppm at Pd >= {hcval_pd_target:.6f}",
        f"- Persistent bank activation: {bank_activation:.6f}",
        f"- Persistent bank GT leakage: {bank_leakage:.6f}",
        "",
        "| Method | Status | Full mIoU | Full Pd | HC-Val FA | HC-Val Precision | Matched-Pd FA | Continue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['status']} | {row['full_mIoU']} | {row['full_Pd']} | "
            f"{row['hcval_FA_ppm']} | {row['hcval_Precision']} | {row['hcval_matched_Pd_FA_ppm']} | "
            f"{row['continue_to_400']} |"
        )
    (output_dir / "STAGE200_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
