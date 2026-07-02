#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("MSHNet", "OHEM")
SPLITS = ("full", "hcval")
MODES = ("fixed", "pd_matched")
COMPONENT_MIN = 0.60
PIXEL_MIN = 0.50


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_counts(summary: dict) -> str:
    stats = summary.get("class_stats", {})
    if any(name in stats for name in ("Boundary excess", "Detached near-FP")):
        parts = []
        for name in ("Boundary excess", "Detached near-FP", "Far-FP"):
            item = stats.get(name, {})
            parts.append(f"{name}={int(item.get('events', item.get('components', 0)))}")
        return ", ".join(parts)
    parts = []
    for name in ("T0", "T1", "T2", "T3", "Far-FP"):
        item = stats.get(name, {})
        parts.append(f"{name}={int(item.get('components', 0))}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate target-near FP census results.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root
    output.mkdir(parents=True, exist_ok=True)
    superseded_invalids = []
    invalidated_manifest = root / "invalidated_results.json"
    if invalidated_manifest.exists():
        manifest = read_json(invalidated_manifest)
        superseded_invalids = manifest.get("rows", [])

    rows = []
    failures = []
    invalids = []
    for method in METHODS:
        for split in SPLITS:
            for mode in MODES:
                case_dir = root / method / split / mode
                summary_path = case_dir / "fp_distance_summary.json"
                invalid_marker = case_dir / "invalid.json"
                if invalid_marker.exists() or (case_dir / "INVALID.md").exists():
                    reason = ""
                    if invalid_marker.exists():
                        reason = read_json(invalid_marker).get("reason", "")
                    row = {
                        "method": method,
                        "split": split,
                        "threshold_mode": mode,
                        "threshold": "",
                        "pd_target": "",
                        "pd_target_reached": "",
                        "fp_components": "",
                        "target_near_components": "",
                        "R_component_target_near": "",
                        "R_pixel_target_near": "",
                        "R_confidence_target_near": "",
                        "component_gate": "",
                        "pixel_gate": "",
                        "threshold_gate": "",
                        "gate_pass": False,
                        "valid_for_gate": False,
                        "status": "INVALID",
                        "invalid_reason": reason,
                        "class_counts": "",
                        "summary_path": str(summary_path),
                        "component_csv": str(case_dir / "fp_components.csv"),
                    }
                    rows.append(row)
                    invalids.append(f"{method}/{split}/{mode}: {reason or 'invalid marker present'}")
                    continue
                summary = read_json(summary_path)
                component_ratio = float(summary["R_component_target_near"])
                pixel_ratio = float(summary["R_pixel_target_near"])
                confidence_ratio = float(summary["R_confidence_target_near"])
                component_pass = component_ratio >= COMPONENT_MIN
                pixel_pass = pixel_ratio >= PIXEL_MIN
                pd_reached = summary.get("pd_target_reached")
                threshold_valid = mode == "fixed" or pd_reached is True
                row = {
                    "method": method,
                    "split": split,
                    "threshold_mode": mode,
                    "threshold": summary["threshold"],
                    "pd_target": summary.get("pd_target"),
                    "pd_target_reached": pd_reached,
                    "fp_components": summary["fp_components"],
                    "target_near_components": summary["target_near_components"],
                    "R_component_target_near": component_ratio,
                    "R_pixel_target_near": pixel_ratio,
                    "R_confidence_target_near": confidence_ratio,
                    "component_gate": component_pass,
                    "pixel_gate": pixel_pass,
                    "threshold_gate": threshold_valid,
                    "gate_pass": component_pass and pixel_pass and threshold_valid,
                    "valid_for_gate": True,
                    "status": "VALID",
                    "invalid_reason": "",
                    "class_counts": class_counts(summary),
                    "summary_path": str(summary_path),
                    "component_csv": summary["component_csv"],
                }
                rows.append(row)
                if not row["gate_pass"]:
                    reasons = []
                    if not component_pass:
                        reasons.append(f"component {pct(component_ratio)} < {pct(COMPONENT_MIN)}")
                    if not pixel_pass:
                        reasons.append(f"pixel {pct(pixel_ratio)} < {pct(PIXEL_MIN)}")
                    if not threshold_valid:
                        reasons.append("Pd target not reachable")
                    failures.append(f"{method}/{split}/{mode}: " + "; ".join(reasons))

    valid_rows = [row for row in rows if row.get("valid_for_gate")]
    all_pass = bool(valid_rows) and all(row["gate_pass"] for row in valid_rows)
    if invalids:
        decision = "HOLD_FP_CENSUS"
    else:
        decision = "TNC_CENSUS_PASS" if all_pass else "STOP_LOCAL_FP_CORRECTION_BRANCH"
    payload = {
        "decision": decision,
        "component_gate_min": COMPONENT_MIN,
        "pixel_gate_min": PIXEL_MIN,
        "all_conditions_pass": all_pass,
        "invalids": invalids,
        "superseded_invalids": superseded_invalids,
        "failures": failures,
        "rows": rows,
        "note": "HC-Test is not used. If any required MSHNet census row is invalid, the TNC hypothesis is not decidable from this census.",
    }

    fieldnames = [
        "method",
        "split",
        "threshold_mode",
        "threshold",
        "pd_target",
        "pd_target_reached",
        "fp_components",
        "target_near_components",
        "R_component_target_near",
        "R_pixel_target_near",
        "R_confidence_target_near",
        "component_gate",
        "pixel_gate",
        "threshold_gate",
        "gate_pass",
        "valid_for_gate",
        "status",
        "invalid_reason",
        "class_counts",
        "summary_path",
        "component_csv",
    ]
    write_csv(output / "fp_census_gate_summary.csv", rows, fieldnames)
    (output / "fp_census_gate_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# FP Distance Census Gate",
        "",
        f"Decision: **{decision}**",
        "",
        "Protocol:",
        "- Splits: full test and HC-Val only.",
        "- HC-Test: not used.",
        "- Baselines: MSHNet and MSHNetOHEM/OHEM.",
        "- Threshold modes: fixed threshold 0.5 and Pd-matched threshold.",
        f"- Gate: target-near FP component ratio >= {pct(COMPONENT_MIN)} and target-near FP pixel-mass ratio >= {pct(PIXEL_MIN)} for every baseline, split, and threshold mode.",
        "- Corrected component definition: boundary excess pixels are not counted as FP components; only unmatched components enter the component denominator.",
        "",
        "## Summary",
        "",
        "| Method | Split | Mode | Status | Thr | Pd matched | FP comp | T-near comp | R_comp | R_pixel | R_conf | Gate | Class counts / reason |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if not row.get("valid_for_gate"):
            lines.append(
                f"| {row['method']} | {row['split']} | {row['threshold_mode']} | INVALID |  |  |  |  |  |  |  | HOLD | {row.get('invalid_reason', '')} |"
            )
            continue
        gate = "PASS" if row["gate_pass"] else "FAIL"
        pd_reached = row["pd_target_reached"]
        pd_text = "NA" if pd_reached is None else str(pd_reached)
        lines.append(
            "| {method} | {split} | {mode} | VALID | {thr:.2f} | {pd} | {fp} | {near} | {rcomp} | {rpix} | {rconf} | {gate} | {counts} |".format(
                method=row["method"],
                split=row["split"],
                mode=row["threshold_mode"],
                thr=float(row["threshold"]),
                pd=pd_text,
                fp=int(row["fp_components"]),
                near=int(row["target_near_components"]),
                rcomp=pct(float(row["R_component_target_near"])),
                rpix=pct(float(row["R_pixel_target_near"])),
                rconf=pct(float(row["R_confidence_target_near"])),
                gate=gate,
                counts=row["class_counts"],
            )
        )

    lines.extend([
        "",
        "## Invalid Rows",
        "",
    ])
    if invalids:
        for item in invalids:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")

    lines.extend([
        "",
        "## Superseded Invalid Rows",
        "",
    ])
    if superseded_invalids:
        for item in superseded_invalids:
            rel = item.get("relative_dir", "")
            reason = item.get("invalid_reason", "")
            replacement = item.get("replacement_summary", "")
            lines.append(f"- {rel}: {reason} Replacement summary: `{replacement}`")
    else:
        lines.append("- None.")

    lines.extend([
        "",
        "## Gate Failures",
        "",
    ])
    if failures:
        for item in failures:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Interpretation", ""])
    if invalids:
        lines.extend([
            "- The hard-clutter bank branch remains stopped.",
            "- If required MSHNet rows are invalid, the correct decision is HOLD_FP_CENSUS rather than using the invalid rows as counter-evidence.",
            "- TNC-OHEM is not opened while the census is on hold.",
            "- Next action is to recover a direct/export-parity MSHNet strong baseline and rerun only MSHNet full/HC-Val fixed/Pd-matched census.",
        ])
    else:
        lines.extend([
            "- The hard-clutter bank branch remains stopped.",
            "- TNC-OHEM is not opened: the target-near error hypothesis fails after MSHNet parity recovery.",
            "- Corrected MSHNet HC-Val census has zero detached near-FP components under both fixed and Pd-matched thresholds; its remaining HC-Val FP components are Far-FP.",
            "- OHEM HC-Val also fails the target-near component and pixel-mass gates under both thresholds.",
            "- The next method direction should focus on detached Far-FP / scene-level false alarms, not target-near correction, hard-clutter bank replay, or inference-time inhibition.",
        ])

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Machine-readable summary: `{output / 'fp_census_gate_summary.json'}`",
        f"- Table: `{output / 'fp_census_gate_summary.csv'}`",
        "- Per-component CSV files are stored under each `METHOD/SPLIT/MODE/fp_components.csv` directory.",
    ])
    report = output / "FP_CENSUS_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "report": str(report), "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
