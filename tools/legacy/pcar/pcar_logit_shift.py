#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def fp_mask(prob: np.ndarray, gt: np.ndarray, threshold: float) -> np.ndarray:
    pred = prob > threshold
    out = np.zeros_like(pred, dtype=bool)
    label = measure.label(pred.astype(np.uint8), connectivity=2)
    for region in measure.regionprops(label):
        mask = np.zeros_like(pred, dtype=bool)
        mask[region.coords[:, 0], region.coords[:, 1]] = True
        if not np.logical_and(mask, gt).any():
            out |= mask
    return out


def stats(values: list[np.ndarray]) -> dict:
    if not values:
        return {"count": 0, "q10": 0.0, "median": 0.0, "q90": 0.0, "q99": 0.0, "mean": 0.0}
    arr = np.concatenate([v.reshape(-1) for v in values if v.size > 0])
    if arr.size == 0:
        return {"count": 0, "q10": 0.0, "median": 0.0, "q90": 0.0, "q99": 0.0, "mean": 0.0}
    return {
        "count": int(arr.size),
        "q10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
        "mean": float(arr.mean()),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Target/clutter logit shift diagnostics.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--ohem_exports", required=True)
    parser.add_argument("--method_exports", required=True)
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    if args.image_list:
        names = [line.strip() for line in Path(args.image_list).read_text().splitlines() if line.strip()]
    else:
        names = [line.strip() for line in (dataset_dir / "img_idx" / f"test_{args.dataset_name}.txt").read_text().splitlines() if line.strip()]

    exports = {"OHEM": Path(args.ohem_exports), args.method_name: Path(args.method_exports)}
    values = {method: {"target": [], "ohem_fp": [], "method_fp": [], "easy_bg": []} for method in exports}

    for name in names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        ohem_prob = np.load(exports["OHEM"] / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        method_prob = np.load(exports[args.method_name] / "probs" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
        ohem_fp = fp_mask(ohem_prob, gt, args.threshold)
        method_fp = fp_mask(method_prob, gt, args.threshold)
        easy_bg = (~gt) & (~ohem_fp) & (~method_fp) & (ohem_prob < 0.05) & (method_prob < 0.05)
        region_masks = {"target": gt, "ohem_fp": ohem_fp, "method_fp": method_fp, "easy_bg": easy_bg}
        for method, export_dir in exports.items():
            logit = np.load(export_dir / "logits" / f"{name}.npy").astype(np.float32)[: gt.shape[0], : gt.shape[1]]
            for region_name, mask in region_masks.items():
                if mask.any():
                    values[method][region_name].append(logit[mask])

    rows = []
    summary = {}
    for method in exports:
        method_summary = {}
        for region_name in ["target", "ohem_fp", "method_fp", "easy_bg"]:
            row = {"method": method, "region": region_name, **stats(values[method][region_name])}
            rows.append(row)
            method_summary[region_name] = row
        target_q10 = method_summary["target"]["q10"]
        own_fp = method_summary["method_fp" if method != "OHEM" else "ohem_fp"]
        summary[method] = {
            "target_q10": target_q10,
            "target_median": method_summary["target"]["median"],
            "fp_q90": own_fp["q90"],
            "fp_q99": own_fp["q99"],
            "separation": target_q10 - own_fp["q99"],
        }

    output_dir = Path(args.output_dir)
    fields = ["method", "region", "count", "q10", "median", "q90", "q99", "mean"]
    write_csv(output_dir / "logit_shift_regions.csv", rows, fields)
    write_csv(
        output_dir / "logit_shift_summary.csv",
        [{"method": method, **vals} for method, vals in summary.items()],
        ["method", "target_q10", "target_median", "fp_q90", "fp_q99", "separation"],
    )
    (output_dir / "logit_shift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
