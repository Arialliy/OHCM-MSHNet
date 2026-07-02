#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


def nearest_odd(value: float) -> int:
    rounded = max(1, int(round(float(value))))
    if rounded % 2 == 1:
        return rounded
    lower = max(1, rounded - 1)
    upper = rounded + 1
    return lower if abs(lower - value) <= abs(upper - value) else upper


def read_mask(path_png: Path, path_bmp: Path) -> np.ndarray:
    path = path_png if path_png.exists() else path_bmp
    mask = np.array(Image.open(path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 127.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TSR-OHEM target-scale windows from train-set masks.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    train_list_path = dataset_dir / "img_idx" / f"train_{args.dataset_name}.txt"
    names = [line.strip() for line in train_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    diameters = []
    areas = []
    for name in names:
        mask = read_mask(dataset_dir / "masks" / f"{name}.png", dataset_dir / "masks" / f"{name}.bmp")
        label = measure.label(mask.astype(np.uint8), connectivity=2)
        for region in measure.regionprops(label):
            area = float(region.area)
            if area <= 0:
                continue
            areas.append(area)
            diameters.append(2.0 * math.sqrt(area / math.pi))

    if not diameters:
        raise RuntimeError(f"No target components found in {train_list_path}")

    quantile_values = np.percentile(np.array(diameters, dtype=np.float64), [25, 50, 75])
    raw_scales = [nearest_odd(value) for value in quantile_values]
    scales = sorted(dict.fromkeys(raw_scales))
    median_scale = raw_scales[1]
    dilation_radius = max(3, int(math.ceil(float(median_scale) / 2.0)))

    output = {
        "dataset": args.dataset_name,
        "train_images": len(names),
        "target_components": len(diameters),
        "diameter_quantiles": {
            "q25": float(quantile_values[0]),
            "q50": float(quantile_values[1]),
            "q75": float(quantile_values[2]),
        },
        "area_quantiles": {
            "q25": float(np.percentile(np.array(areas, dtype=np.float64), 25)),
            "q50": float(np.percentile(np.array(areas, dtype=np.float64), 50)),
            "q75": float(np.percentile(np.array(areas, dtype=np.float64), 75)),
        },
        "raw_odd_scales": raw_scales,
        "target_scales": scales,
        "target_scales_arg": ",".join(str(scale) for scale in scales),
        "median_scale": int(median_scale),
        "safe_background_dilation_radius": int(dilation_radius),
        "definition": "d = 2 * sqrt(area / pi); scales are q25/q50/q75 rounded to nearest odd integer.",
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
