#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage import measure


def parse_args():
    parser = argparse.ArgumentParser(description="Audit ERD reliability pseudo labels before training.")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--min_neg_pixels_mean", type=float, default=1.0)
    parser.add_argument("--max_images_without_neg_ratio", type=float, default=0.50)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(Path(args.label_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError("no npz files in %s" % args.label_dir)

    pos_counts = []
    neg_counts = []
    valid_counts = []
    leakage_counts = []
    neg_component_counts = []
    neg_component_area_means = []
    ohem_far_candidate_counts = []
    no_neg = 0
    invalid_files = []

    for path in files:
        data = np.load(path)
        rel_label = data["rel_label"]
        rel_valid = data["rel_valid"] > 0
        neg_mask = (rel_label <= 0.5) & rel_valid
        pos = int(((rel_label > 0.5) & rel_valid).sum())
        neg = int(neg_mask.sum())
        valid = int(rel_valid.sum())
        pos_counts.append(pos)
        neg_counts.append(neg)
        valid_counts.append(valid)
        if neg == 0:
            no_neg += 1
        if valid == 0 or pos == 0:
            invalid_files.append(path.name)
        if "target_core" in data:
            target_core = data["target_core"] > 0
            leakage_counts.append(int((neg_mask & target_core).sum()))
        if "ohem_far_candidate" in data:
            ohem_far_candidate_counts.append(int((data["ohem_far_candidate"] > 0).sum()))

        regions = measure.regionprops(measure.label(neg_mask.astype(np.uint8), connectivity=2))
        neg_component_counts.append(len(regions))
        if regions:
            neg_component_area_means.append(float(np.mean([region.area for region in regions])))
        else:
            neg_component_area_means.append(0.0)

    target_leakage_neg_pixels = int(np.sum(leakage_counts)) if leakage_counts else -1
    ohem_far_total = int(np.sum(ohem_far_candidate_counts)) if ohem_far_candidate_counts else 0
    neg_total = int(np.sum(neg_counts))
    negative_overlap_ohem_fp_recall = float(neg_total) / float(ohem_far_total) if ohem_far_total else 0.0

    stats = {
        "num_images": len(files),
        "rel_pos_pixels_mean": float(np.mean(pos_counts)),
        "rel_neg_pixels_mean": float(np.mean(neg_counts)),
        "rel_valid_pixels_mean": float(np.mean(valid_counts)),
        "num_images_without_neg": int(no_neg),
        "images_without_neg_ratio": float(no_neg) / float(len(files)),
        "target_leakage_neg_pixels": target_leakage_neg_pixels,
        "negative_component_count_mean": float(np.mean(neg_component_counts)),
        "negative_component_area_mean": float(np.mean(neg_component_area_means)),
        "negative_overlap_ohem_fp_recall": negative_overlap_ohem_fp_recall,
        "invalid_files_with_no_valid_or_pos": len(invalid_files),
    }

    gate_pass = (
        stats["rel_pos_pixels_mean"] > 0
        and stats["rel_neg_pixels_mean"] >= args.min_neg_pixels_mean
        and stats["images_without_neg_ratio"] <= args.max_images_without_neg_ratio
        and stats["target_leakage_neg_pixels"] == 0
        and stats["invalid_files_with_no_valid_or_pos"] == 0
    )
    stats["gate_pass"] = bool(gate_pass)

    lines = ["%s: %s" % (key, value) for key, value in stats.items()]
    if invalid_files:
        lines.append("invalid_files_sample: %s" % ",".join(invalid_files[:20]))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")

    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
