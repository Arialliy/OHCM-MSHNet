#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["metrics_at_threshold"]


def fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def add_metric_row(rows: list[dict], method: str, split: str, path: Path) -> None:
    if not path.exists():
        return
    metrics = load_metrics(path)
    rows.append(
        {
            "method": method,
            "split": split,
            "mIoU": metrics.get("mIoU", ""),
            "nIoU": metrics.get("nIoU", ""),
            "Pd": metrics.get("Pd", ""),
            "FA_ppm": metrics.get("FA_ppm", ""),
            "Precision": metrics.get("Precision", ""),
            "Recall": metrics.get("Recall", ""),
            "F1": metrics.get("F1", ""),
            "FP_components": metrics.get("FP_components", ""),
            "summary": str(path),
        }
    )


def row_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["method"], row["split"]): row for row in rows}


def as_float(row: dict | None, key: str):
    if row is None:
        return None
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OHCM-light/full equivalence audit.")
    parser.add_argument("--audit_root", required=True)
    parser.add_argument("--light_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/OHCM-light/NUDT-SIRST/seed_42")
    parser.add_argument("--lambda0_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step4_lambda0_sanity/20260615_step4_lambda0_3ca/OHCM-full-lambda0/NUDT-SIRST/seed_42")
    parser.add_argument("--proto_root", default="/home/ly/AAAI/OHCM-MSHNet/results/step4_ohcm_full_proto/20260614_step4_proto_3ca/OHCM-full/NUDT-SIRST/seed_42")
    parser.add_argument("--swap_root", required=True)
    parser.add_argument("--parity_summary", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    roots = [
        ("OHCM-light", Path(args.light_root)),
        ("OHCM-full lambda_proto=0 trained", Path(args.lambda0_root)),
        ("OHCM-full lambda_proto>0 trained", Path(args.proto_root)),
        ("Checkpoint-swap full path using OHCM-light ckpt", Path(args.swap_root)),
    ]
    for method, root in roots:
        add_metric_row(rows, method, "full", root / "eval_full" / "summary_metrics.json")
        add_metric_row(rows, method, "hcset", root / "eval_hcset" / "summary_metrics.json")

    metric_fields = ["method", "split", "mIoU", "nIoU", "Pd", "FA_ppm", "Precision", "Recall", "F1", "FP_components", "summary"]
    with (audit_root / "equivalence_metric_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)

    parity = json.loads(Path(args.parity_summary).read_text(encoding="utf-8"))
    idx = row_lookup(rows)
    light_hc = idx.get(("OHCM-light", "hcset"))
    swap_hc = idx.get(("Checkpoint-swap full path using OHCM-light ckpt", "hcset"))
    lambda0_hc = idx.get(("OHCM-full lambda_proto=0 trained", "hcset"))

    dm = None if not light_hc or not swap_hc else as_float(swap_hc, "mIoU") - as_float(light_hc, "mIoU")
    dfa = None if not light_hc or not swap_hc else as_float(swap_hc, "FA_ppm") - as_float(light_hc, "FA_ppm")
    dp = None if not light_hc or not swap_hc else as_float(swap_hc, "Precision") - as_float(light_hc, "Precision")
    swap_equiv = (
        dm is not None
        and abs(dm) <= 0.003
        and abs(dfa) <= 5.0
        and abs(dp) <= 0.01
    )

    if parity.get("pass_parity") and swap_equiv:
        decision = "FORWARD_AND_EVAL_PATH_EQUIVALENT"
        interpretation = (
            "Using the frozen OHCM-light checkpoint, the full path reproduces OHCM-light within the requested tolerance. "
            "The trained lambda_proto=0 degradation is therefore a training-trajectory/config reproducibility issue rather than a forward/eval formula mismatch."
        )
    elif parity.get("pass_parity") and not swap_equiv:
        decision = "FORWARD_EQUIVALENT_BUT_EVAL_EXPORT_MISMATCH"
        interpretation = "Forward parity passes, but checkpoint-swap metrics do not match; inspect export/evaluation, threshold, HC-list, or saved prediction path."
    else:
        decision = "FORWARD_NOT_EQUIVALENT"
        interpretation = "Same checkpoint gives different z_t/z_c/z_final or binary masks; inspect model_name-specific forward/config first."

    lines = [
        "# Full-Light Equivalence Audit",
        "",
        f"Decision: {decision}",
        "",
        "## HC-Set Metrics",
        "",
        "| Method | HC-mIoU | HC-FA ppm | HC-Precision | HC-Pd |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in [item[0] for item in roots]:
        row = idx.get((method, "hcset"))
        lines.append(
            f"| {method} | {fmt(as_float(row, 'mIoU'))} | {fmt(as_float(row, 'FA_ppm'))} | "
            f"{fmt(as_float(row, 'Precision'))} | {fmt(as_float(row, 'Pd'))} |"
        )
    lines.extend(
        [
            "",
            "## Forward Parity",
            "",
            f"- pass_parity: {parity.get('pass_parity')}",
            f"- num_images: {parity.get('num_images')}",
            f"- max z_t abs diff: {parity.get('max_values', {}).get('z_t_max_abs')}",
            f"- max z_c abs diff: {parity.get('max_values', {}).get('z_c_max_abs')}",
            f"- max z_final abs diff: {parity.get('max_values', {}).get('z_final_max_abs')}",
            f"- max binary diff pixels: {parity.get('max_values', {}).get('binary_diff_pixels')}",
            "",
            "## Checkpoint-Swap Tolerance",
            "",
            "- Required: HC-mIoU within +/-0.003, HC-FA within +/-5 ppm, HC-Precision within +/-0.01.",
            f"- Observed swap minus light HC-mIoU: {fmt(dm)}",
            f"- Observed swap minus light HC-FA ppm: {fmt(dfa)}",
            f"- Observed swap minus light HC-Precision: {fmt(dp)}",
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "## Next Rule",
            "",
            "Do not tune lambda_proto or run Proto-Retry until this audit is accepted and F0/F1/F2 are defined from an equivalent full-light base.",
            "",
        ]
    )
    (audit_root / "FULL_LIGHT_EQUIVALENCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {audit_root / 'equivalence_metric_table.csv'}")
    print(f"Wrote {audit_root / 'FULL_LIGHT_EQUIVALENCE_AUDIT.md'}")


if __name__ == "__main__":
    main()
