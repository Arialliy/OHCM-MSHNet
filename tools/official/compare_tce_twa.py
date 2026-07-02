#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = ["mIoU", "nIoU", "Pd", "Precision", "FA_ppm", "F1", "FP_pixels", "FP_components"]


def load_metrics(path: str | Path) -> dict:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    return summary.get("metrics_at_threshold", summary)


def row_from_summary(label: str, path: str | Path) -> dict:
    metrics = load_metrics(path)
    row = {"label": label, "summary": str(Path(path))}
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "summary", *METRIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OHEM/TCE/TWA summary_metrics.json files.")
    parser.add_argument("--ohem", default=None)
    parser.add_argument("--tce", default=None)
    parser.add_argument("--twa", default=None)
    parser.add_argument("--twa_bn", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = []
    for label, path in [
        ("OHEM-400", args.ohem),
        ("TCE-4", args.tce),
        ("TWA", args.twa),
        ("TWA+BN", args.twa_bn),
    ]:
        if path:
            rows.append(row_from_summary(label, path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "compare_tce_twa.csv", rows)
    (output_dir / "compare_tce_twa.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
