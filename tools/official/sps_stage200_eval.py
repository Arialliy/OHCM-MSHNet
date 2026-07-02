#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


RUNS = {
    "SPS": "SPS-rerank-alpha100-gain1005-hybridfb00001",
    "TwoViewOHEM": "TwoViewOHEM-rerank-gain1005-hybridfb00001",
    "ConfidenceOnly": "ConfidenceOnly-rerank-alpha100-gain1005-hybridfb00001",
    "GlobalConsistency": "GlobalConsistency-rerankctrl-gain1005-hybridfb00001",
}


def parse_runs(value: str | None) -> dict[str, str]:
    if not value:
        return dict(RUNS)
    runs = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, run_name = item.split("=", 1)
        else:
            run_name = item
            label = Path(run_name).name
        runs[label.strip()] = run_name.strip()
    if not runs:
        raise ValueError("--runs did not contain any valid run names.")
    return runs


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict | None, key: str) -> float | None:
    if not summary:
        return None
    value = summary.get("metrics_at_threshold", {}).get(key)
    return float(value) if value is not None else None


def fp_metric(summary: dict | None, key: str) -> float | None:
    if not summary:
        return None
    census = summary.get("fp_census_at_threshold", {})
    value = census.get(key)
    return float(value) if value is not None else None


def read_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def matched_pd_fa(curve_path: Path, target_pd: float | None) -> float | None:
    if target_pd is None:
        return None
    feasible = [row for row in read_curve(curve_path) if row.get("Pd", -1.0) >= target_pd]
    if not feasible:
        return None
    return min(row["FA_ppm"] for row in feasible)


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def count_pair(detected: float | None, total: float | None) -> str:
    if detected is None or total is None:
        return "NA"
    return f"{int(round(detected))}/{int(round(total))}"


def pass_fail(value: bool | None) -> str:
    if value is None:
        return "PENDING"
    return "PASS" if value else "FAIL"


def as_float(value: str) -> float | None:
    if value == "NA":
        return None
    return float(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def eval_checkpoint(args, run_label: str, run_name: str, epoch: int, split: str, image_list: str | None) -> Path | None:
    checkpoint = (
        Path(args.run_root)
        / run_name
        / f"seed_{args.seed}"
        / "checkpoints"
        / args.dataset_name
        / f"MSHNetSPSOHEM_{epoch}.pth.tar"
    )
    if not checkpoint.exists():
        return None

    output_dir = Path(args.run_root) / run_name / f"seed_{args.seed}" / f"eval_e{epoch}_{split}"
    summary = output_dir / "summary_metrics.json"
    if summary.exists() and not args.force:
        return summary

    cmd = [
        sys.executable,
        "tools/official/evaluate_checkpoint_direct.py",
        "--dataset_dir",
        args.dataset_dir,
        "--dataset_name",
        args.dataset_name,
        "--model_name",
        "MSHNetSPSOHEM",
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
        "--method",
        run_label,
        "--seed",
        str(args.seed),
        "--threshold",
        str(args.threshold),
        "--thresholds",
        args.thresholds,
    ]
    if image_list:
        cmd.extend(["--image_list", image_list])
    mpl_config = Path(args.run_root) / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(mpl_config))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    subprocess.run(cmd, check=True, env=env)
    return summary


def row_for(
    label: str,
    run_name: str,
    epoch: int,
    full: dict | None,
    hcval: dict | None,
    base_full: dict | None,
    base_hcval: dict | None,
) -> dict:
    base_full_miou = metric(base_full, "mIoU")
    base_full_pd = metric(base_full, "Pd")
    base_full_precision = metric(base_full, "Precision")
    base_full_fa = metric(base_full, "FA_ppm")
    base_hcval_miou = metric(base_hcval, "mIoU")
    base_hcval_pd = metric(base_hcval, "Pd")
    base_hcval_precision = metric(base_hcval, "Precision")
    base_hcval_fa = metric(base_hcval, "FA_ppm")
    base_hcval_far_comp = fp_metric(base_hcval, "far_fp_components")
    base_hcval_far_pix = fp_metric(base_hcval, "far_fp_pixel_mass")

    full_miou = metric(full, "mIoU")
    full_pd = metric(full, "Pd")
    full_detected_targets = metric(full, "detected_targets")
    full_target_components = metric(full, "target_components")
    full_precision = metric(full, "Precision")
    full_fa = metric(full, "FA_ppm")
    full_f1 = metric(full, "F1")
    hcval_miou = metric(hcval, "mIoU")
    hcval_pd = metric(hcval, "Pd")
    hcval_detected_targets = metric(hcval, "detected_targets")
    hcval_target_components = metric(hcval, "target_components")
    hcval_precision = metric(hcval, "Precision")
    hcval_fa = metric(hcval, "FA_ppm")
    hcval_f1 = metric(hcval, "F1")
    hcval_boundary_pix = fp_metric(hcval, "boundary_excess_pixel_mass")
    hcval_detached_near_comp = fp_metric(hcval, "detached_near_fp_components")
    hcval_detached_near_pix = fp_metric(hcval, "detached_near_fp_pixel_mass")
    hcval_far_comp = fp_metric(hcval, "far_fp_components")
    hcval_far_pix = fp_metric(hcval, "far_fp_pixel_mass")
    hcval_unmatched_comp = fp_metric(hcval, "unmatched_fp_components")

    hcval_curve = None
    if hcval:
        hcval_curve = Path(hcval["outputs"]["threshold_curve"])
    hcval_matched_pd_fa = matched_pd_fa(hcval_curve, base_hcval_pd) if hcval_curve else None

    def delta(value: float | None, base: float | None) -> float | None:
        if value is None or base is None:
            return None
        return value - base

    full_gate = None
    if full and base_full:
        full_gate = (
            full_miou >= base_full_miou
            and full_pd >= base_full_pd
            and full_precision >= base_full_precision
            and full_fa <= base_full_fa
        )
    hcval_submit_gate = None
    if hcval and base_hcval:
        hcval_submit_gate = (
            delta(hcval_miou, base_hcval_miou) >= 0.010
            and delta(base_hcval_fa, hcval_fa) >= 20.0
            and delta(hcval_precision, base_hcval_precision) >= 0.012
            and hcval_pd >= base_hcval_pd
        )
    hcval_strong_gate = None
    if hcval and base_hcval:
        hcval_strong_gate = (
            delta(hcval_miou, base_hcval_miou) >= 0.0125
            and hcval_fa <= 360.0
            and delta(hcval_precision, base_hcval_precision) >= 0.015
            and hcval_pd >= base_hcval_pd
        )
    matched_pd_gate = None
    if hcval_matched_pd_fa is not None and base_hcval_fa is not None:
        matched_pd_gate = (base_hcval_fa - hcval_matched_pd_fa) >= 20.0

    return {
        "run_label": label,
        "run_name": run_name,
        "epoch": epoch,
        "status": "DONE" if full and hcval else "PENDING",
        "full_mIoU": fmt(full_miou),
        "full_Pd": fmt(full_pd),
        "full_detected_targets": fmt(full_detected_targets, 0),
        "full_target_components": fmt(full_target_components, 0),
        "full_targets": count_pair(full_detected_targets, full_target_components),
        "full_FA_ppm": fmt(full_fa),
        "full_Precision": fmt(full_precision),
        "full_F1": fmt(full_f1),
        "full_delta_mIoU": fmt(delta(full_miou, base_full_miou)),
        "full_delta_Pd": fmt(delta(full_pd, base_full_pd)),
        "full_delta_FA_ppm": fmt(delta(full_fa, base_full_fa)),
        "full_gate": pass_fail(full_gate),
        "hcval_mIoU": fmt(hcval_miou),
        "hcval_Pd": fmt(hcval_pd),
        "hcval_detected_targets": fmt(hcval_detected_targets, 0),
        "hcval_target_components": fmt(hcval_target_components, 0),
        "hcval_targets": count_pair(hcval_detected_targets, hcval_target_components),
        "hcval_FA_ppm": fmt(hcval_fa),
        "hcval_Precision": fmt(hcval_precision),
        "hcval_F1": fmt(hcval_f1),
        "hcval_delta_mIoU": fmt(delta(hcval_miou, base_hcval_miou)),
        "hcval_delta_Pd": fmt(delta(hcval_pd, base_hcval_pd)),
        "hcval_delta_FA_ppm": fmt(delta(hcval_fa, base_hcval_fa)),
        "hcval_delta_Precision": fmt(delta(hcval_precision, base_hcval_precision)),
        "hcval_boundary_excess_pixels": fmt(hcval_boundary_pix),
        "hcval_detached_near_fp_components": fmt(hcval_detached_near_comp),
        "hcval_detached_near_fp_pixels": fmt(hcval_detached_near_pix),
        "hcval_far_fp_components": fmt(hcval_far_comp),
        "hcval_far_fp_pixels": fmt(hcval_far_pix),
        "hcval_unmatched_fp_components": fmt(hcval_unmatched_comp),
        "hcval_delta_far_fp_components": fmt(delta(hcval_far_comp, base_hcval_far_comp)),
        "hcval_delta_far_fp_pixels": fmt(delta(hcval_far_pix, base_hcval_far_pix)),
        "hcval_matched_Pd_FA_ppm": fmt(hcval_matched_pd_fa),
        "hcval_submit_gate": pass_fail(hcval_submit_gate),
        "hcval_strong_gate": pass_fail(hcval_strong_gate),
        "hcval_matched_Pd_gate": pass_fail(matched_pd_gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and gate SPS Stage-200 checkpoints.")
    parser.add_argument("--run_root", default="results/sps_ohem/20260624_sps_stage200")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", default="50,100,150,200")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--hcval_list", default="results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hcval_NUDT-SIRST.txt")
    parser.add_argument("--baseline_full", default="results/step3_ohcm_light_gate/20260613_step3_gate/MSHNetOHEM/NUDT-SIRST/seed_42/eval_full/summary_metrics.json")
    parser.add_argument("--baseline_hcval", default="results/aaai_p0_paired/20260617_aaai_p0_paired/MSHNetOHEM/NUDT-SIRST/seed_42/eval_hcval/summary_metrics.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--runs", default=None, help="Comma-separated label=run_name entries. Defaults to the original four rerank Stage-200 runs.")
    parser.add_argument("--primary_label", default="SPS")
    parser.add_argument("--control_labels", default="TwoViewOHEM,ConfidenceOnly,GlobalConsistency")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_full = read_json(Path(args.baseline_full))
    base_hcval = read_json(Path(args.baseline_hcval))
    base_hcval_pd = metric(base_hcval, "Pd")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.run_root) / "gate"
    epochs = [int(item) for item in args.epochs.split(",") if item.strip()]
    runs = parse_runs(args.runs)
    control_labels = [item.strip() for item in args.control_labels.split(",") if item.strip()]

    rows = []
    for label, run_name in runs.items():
        for epoch in epochs:
            full_path = eval_checkpoint(args, label, run_name, epoch, "full", None)
            hcval_path = eval_checkpoint(args, label, run_name, epoch, "hcval", args.hcval_list)
            full = read_json(full_path) if full_path else None
            hcval = read_json(hcval_path) if hcval_path else None
            rows.append(row_for(label, run_name, epoch, full, hcval, base_full, base_hcval))

    write_csv(output_dir / "stage200_eval_table.csv", rows)

    lines = [
        "# SPS Stage-200 Gate Report",
        "",
        "HC-Test is intentionally not evaluated in this stage.",
        "",
        f"- Baseline Full mIoU: {fmt(metric(base_full, 'mIoU'))}",
        f"- Baseline Full Pd: {fmt(metric(base_full, 'Pd'))}",
        f"- Baseline Full detected targets: {count_pair(metric(base_full, 'detected_targets'), metric(base_full, 'target_components'))}",
        f"- Baseline Full FA ppm: {fmt(metric(base_full, 'FA_ppm'))}",
        f"- Baseline Full Precision: {fmt(metric(base_full, 'Precision'))}",
        f"- Baseline HC-Val mIoU: {fmt(metric(base_hcval, 'mIoU'))}",
        f"- Baseline HC-Val Pd: {fmt(metric(base_hcval, 'Pd'))}",
        f"- Baseline HC-Val detected targets: {count_pair(metric(base_hcval, 'detected_targets'), metric(base_hcval, 'target_components'))}",
        f"- Baseline HC-Val FA ppm: {fmt(metric(base_hcval, 'FA_ppm'))}",
        f"- Baseline HC-Val Precision: {fmt(metric(base_hcval, 'Precision'))}",
        "",
        "| Run | Epoch | Status | Full mIoU | Full Pd | Full Det/GT | Full FA ppm | Full Gate | HC-Val mIoU | HC-Val Pd | HC-Val Det/GT | HC-Val FA ppm | dFA ppm | dPrecision | Submit | Strong | Pd-matched |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_label']} | {row['epoch']} | {row['status']} | "
            f"{row['full_mIoU']} | {row['full_Pd']} | {row['full_targets']} | "
            f"{row['full_FA_ppm']} | {row['full_gate']} | "
            f"{row['hcval_mIoU']} | {row['hcval_Pd']} | {row['hcval_targets']} | {row['hcval_FA_ppm']} | "
            f"{row['hcval_delta_FA_ppm']} | {row['hcval_delta_Precision']} | "
            f"{row['hcval_submit_gate']} | {row['hcval_strong_gate']} | {row['hcval_matched_Pd_gate']} |"
        )
    lines.extend([
        "",
        "## HC-Val FP Component Census",
        "",
        "| Run | Epoch | Boundary pixels | Detached near-FP comp | Detached near-FP pixels | Far-FP comp | Far-FP pixels | Unmatched FP comp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['run_label']} | {row['epoch']} | {row['hcval_boundary_excess_pixels']} | "
            f"{row['hcval_detached_near_fp_components']} | {row['hcval_detached_near_fp_pixels']} | "
            f"{row['hcval_far_fp_components']} | {row['hcval_far_fp_pixels']} | {row['hcval_unmatched_fp_components']} |"
        )

    control_rows = []
    for epoch in epochs:
        by_label = {row["run_label"]: row for row in rows if row["epoch"] == epoch and row["status"] == "DONE"}
        sps = by_label.get(args.primary_label)
        if not sps:
            continue
        for control in control_labels:
            other = by_label.get(control)
            if not other:
                continue
            sps_miou = as_float(sps["hcval_mIoU"])
            other_miou = as_float(other["hcval_mIoU"])
            sps_fa = as_float(sps["hcval_FA_ppm"])
            other_fa = as_float(other["hcval_FA_ppm"])
            sps_precision = as_float(sps["hcval_Precision"])
            other_precision = as_float(other["hcval_Precision"])
            sps_pd = as_float(sps["hcval_Pd"])
            other_pd = as_float(other["hcval_Pd"])
            sps_matched_fa = as_float(sps["hcval_matched_Pd_FA_ppm"])
            other_matched_fa = as_float(other["hcval_matched_Pd_FA_ppm"])
            sps_far_comp = as_float(sps["hcval_far_fp_components"])
            other_far_comp = as_float(other["hcval_far_fp_components"])
            sps_far_pix = as_float(sps["hcval_far_fp_pixels"])
            other_far_pix = as_float(other["hcval_far_fp_pixels"])
            d_miou = None if sps_miou is None or other_miou is None else sps_miou - other_miou
            d_fa = None if sps_fa is None or other_fa is None else sps_fa - other_fa
            d_precision = None if sps_precision is None or other_precision is None else sps_precision - other_precision
            d_pd = None if sps_pd is None or other_pd is None else sps_pd - other_pd
            d_matched_fa = None if sps_matched_fa is None or other_matched_fa is None else sps_matched_fa - other_matched_fa
            d_far_comp = None if sps_far_comp is None or other_far_comp is None else sps_far_comp - other_far_comp
            d_far_pix = None if sps_far_pix is None or other_far_pix is None else sps_far_pix - other_far_pix
            beats = None
            if None not in (d_miou, d_fa, d_precision, d_matched_fa, sps_pd, base_hcval_pd):
                beats = (
                    d_miou > 0.0
                    and d_fa < 0.0
                    and d_precision > 0.0
                    and d_matched_fa < 0.0
                    and sps_pd + 1e-6 >= base_hcval_pd
                )
            control_rows.append({
                "epoch": epoch,
                "control": control,
                "delta_hcval_mIoU": fmt(d_miou),
                "delta_hcval_FA_ppm": fmt(d_fa),
                "delta_hcval_Precision": fmt(d_precision),
                "delta_hcval_Pd": fmt(d_pd),
                "delta_hcval_matched_Pd_FA_ppm": fmt(d_matched_fa),
                "delta_hcval_far_fp_components": fmt(d_far_comp),
                "delta_hcval_far_fp_pixels": fmt(d_far_pix),
                "sps_beats_control": pass_fail(beats),
            })

    if control_rows:
        write_csv(output_dir / "stage200_sps_vs_controls.csv", control_rows)
        lines.extend([
            "",
            "## SPS vs Controls",
            "",
            "| Epoch | Control | dHC-Val mIoU | dHC-Val FA ppm | dHC-Val Precision | dHC-Val Pd | dMatched-Pd FA ppm | dFar-FP comp | dFar-FP pixels | SPS beats control |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in control_rows:
            lines.append(
                f"| {row['epoch']} | {row['control']} | {row['delta_hcval_mIoU']} | "
                f"{row['delta_hcval_FA_ppm']} | {row['delta_hcval_Precision']} | "
                f"{row['delta_hcval_Pd']} | {row['delta_hcval_matched_Pd_FA_ppm']} | "
                f"{row['delta_hcval_far_fp_components']} | "
                f"{row['delta_hcval_far_fp_pixels']} | {row['sps_beats_control']} |"
            )
    lines.extend([
        "",
        "Stage-200 continuation rule:",
        "",
        "- SPS must preserve Full metrics and beat TwoViewOHEM, ConfidenceOnly, and GlobalConsistency on HC-Val fixed-threshold and baseline-Pd matched FA.",
        "- If ConfidenceOnly or GlobalConsistency matches SPS, the current mechanism claim is not strong enough.",
    ])
    (output_dir / "STAGE200_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
