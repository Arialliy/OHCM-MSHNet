#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HCVAL_IMAGE_LISTS = (
    PROJECT_ROOT / "docs/internal/hc_protocol/hcval_NUDT-SIRST.txt",
    Path("/home/AAAI/OHCM-MSHNet-1/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hcval_NUDT-SIRST.txt"),
    Path("/home/ly/AAAI/OHCM-MSHNet-1/results/aaai_p0_paired/20260617_aaai_p0_paired/hc_protocol/hcval_NUDT-SIRST.txt"),
)


def resolve_hcval_image_list(dataset_name: str, image_list: str | None) -> str:
    if image_list:
        return image_list
    candidates = [
        Path(str(path).replace("hcval_NUDT-SIRST.txt", f"hcval_{dataset_name}.txt"))
        for path in DEFAULT_HCVAL_IMAGE_LISTS
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"HC-Val image list not found for {dataset_name}. Searched: {searched}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard official evaluation wrapper for TWA checkpoints.")
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="full", choices=["full", "image_list", "hcval"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="TWAOHEM")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    args = parser.parse_args()

    if args.split == "image_list" and not args.image_list:
        raise ValueError("--split image_list requires --image_list")
    if args.split == "hcval":
        args.image_list = resolve_hcval_image_list(args.dataset_name, args.image_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_script = PROJECT_ROOT / "tools" / "official" / "evaluate_checkpoint_direct.py"
    cmd = [
        sys.executable,
        str(direct_script),
        "--dataset_dir",
        args.dataset_dir,
        "--dataset_name",
        args.dataset_name,
        "--model_name",
        args.model_name,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        str(output_dir),
        "--method",
        args.method,
        "--threshold",
        str(args.threshold),
        "--thresholds",
        args.thresholds,
        "--mshnet_export_head",
        "final",
    ]
    if args.train_dataset_name:
        cmd += ["--train_dataset_name", args.train_dataset_name]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.image_list:
        cmd += ["--image_list", args.image_list]

    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))
    fp_components = output_dir / "fp_components.csv"
    component_fp = output_dir / "component_fp_analysis.csv"
    if fp_components.exists() and not component_fp.exists():
        shutil.copyfile(fp_components, component_fp)
    meta = {
        "wrapper": "evaluate_twa_checkpoint",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_name": args.model_name,
        "dataset": args.dataset_name,
        "split": args.split,
        "direct_command": cmd,
        "outputs": {
            "summary_metrics": str(output_dir / "summary_metrics.json"),
            "threshold_curve": str(output_dir / "threshold_curve.csv"),
            "component_fp_analysis": str(component_fp),
        },
    }
    (output_dir / "twa_eval_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
