#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TARGET_LIKE = {
    "target-like clutter",
    "target_like_clutter",
    "target-like",
    "target_like",
    "targetlike",
}
TARGET_NEAR = {"target-near ambiguity", "target_near_ambiguity", "target-near", "target_near"}
ANNOTATION_AMBIGUITY = {"annotation ambiguity", "annotation_ambiguity"}
LARGE_BACKGROUND = {"large structured background", "large_structured_background"}
ORDINARY_BG = {"ordinary background", "ordinary_background", "background"}


def truthy(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "keep"}


def norm(value: str | None) -> str:
    return str(value or "").strip().lower()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check manually verified oracle clutter bank purity gates.")
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_oracle_size", type=int, default=300)
    parser.add_argument("--target_precision_min", type=float, default=0.90)
    parser.add_argument("--flat_ratio_max", type=float, default=0.05)
    parser.add_argument("--large_ratio_max", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_rows(Path(args.review_csv))
    verified = [row for row in rows if truthy(row.get("verified"))]
    keep_rows = [row for row in rows if truthy(row.get("keep_for_oracle"))]
    target_like_verified = [row for row in verified if norm(row.get("clutter_type")) in TARGET_LIKE]
    flat_ratio = safe_div(sum(int(float(row.get("flat_candidate", 0) or 0)) for row in keep_rows), len(keep_rows))
    large_ratio = safe_div(sum(int(float(row.get("large_region", 0) or 0)) for row in keep_rows), len(keep_rows))
    gt_leakage = sum(int(float(row.get("gt_leakage", 0) or 0)) for row in keep_rows)
    target_near = [row for row in keep_rows if norm(row.get("clutter_type")) in TARGET_NEAR]
    annotation_ambiguity = [row for row in keep_rows if norm(row.get("clutter_type")) in ANNOTATION_AMBIGUITY]
    non_target_keep = [row for row in keep_rows if norm(row.get("clutter_type")) not in TARGET_LIKE]
    target_like_keep = [row for row in keep_rows if norm(row.get("clutter_type")) in TARGET_LIKE]
    target_like_precision = safe_div(len(target_like_keep), len(keep_rows))

    gate_size = len(keep_rows) >= args.min_oracle_size
    gate_precision = bool(verified) and target_like_precision >= args.target_precision_min
    gate_flat = flat_ratio <= args.flat_ratio_max
    gate_gt = gt_leakage == 0
    gate_target_near = len(target_near) == 0
    gate_large = large_ratio <= args.large_ratio_max
    gate_non_target = len(non_target_keep) == 0
    gates = {
        "min_oracle_size": gate_size,
        "target_like_precision": gate_precision,
        "flat_candidate_ratio": gate_flat,
        "gt_leakage": gate_gt,
        "target_near_ambiguity": gate_target_near,
        "large_region_ratio": gate_large,
        "oracle_keep_all_target_like": gate_non_target,
    }
    status = "PASS" if all(gates.values()) else ("PENDING" if not verified else "FAIL")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["candidate_id"]
    write_csv(output_dir / "oracle_bank_verified.csv", keep_rows, fields)
    bank = {
        "source_review_csv": str(Path(args.review_csv).resolve()),
        "status": status,
        "records": [
            {
                "candidate_id": row.get("candidate_id"),
                "image_id": row.get("image_id"),
                "mask_path": row.get("mask_path"),
                "bbox": [int(float(row["y0"])), int(float(row["y1"])), int(float(row["x0"])), int(float(row["x1"]))],
                "area": int(float(row.get("area", 0) or 0)),
                "teacher_score": float(row.get("teacher_score", 0) or 0),
                "threshold_persistence": int(float(row.get("threshold_persistence", 0) or 0)),
                "local_contrast": float(row.get("local_contrast", 0) or 0),
                "clutter_type": row.get("clutter_type", ""),
                "verified": truthy(row.get("verified")),
            }
            for row in keep_rows
        ],
    }
    (output_dir / "oracle_bank.json").write_text(json.dumps(bank, indent=2), encoding="utf-8")
    summary = {
        "status": status,
        "review_rows": len(rows),
        "verified_rows": len(verified),
        "target_like_verified": len(target_like_verified),
        "keep_for_oracle": len(keep_rows),
        "target_like_precision": target_like_precision,
        "flat_candidate_ratio": flat_ratio,
        "gt_leakage": gt_leakage,
        "target_near_ambiguity": len(target_near),
        "annotation_ambiguity_kept": len(annotation_ambiguity),
        "large_region_ratio": large_ratio,
        "non_target_keep": len(non_target_keep),
        "gates": gates,
        "oracle_bank_json": str(output_dir / "oracle_bank.json"),
    }
    (output_dir / "oracle_bank_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Oracle Bank Gate",
        "",
        f"- Status: {status}",
        f"- Review rows: {len(rows)}",
        f"- Verified rows: {len(verified)}",
        f"- Keep for oracle: {len(keep_rows)}",
        f"- Target-like precision: {target_like_precision:.4f}",
        f"- Flat candidate ratio: {flat_ratio:.4f}",
        f"- GT leakage: {gt_leakage}",
        f"- Target-near ambiguity kept: {len(target_near)}",
        f"- Large-region ratio: {large_ratio:.4f}",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    (output_dir / "ORACLE_BANK_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
