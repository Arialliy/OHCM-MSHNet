#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import morphology


FORBIDDEN_SPLIT_TOKENS = ("test", "hc-test", "hctest", "blind", "external")
IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
PROB_EXTS = (".npy", ".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Build train-only ERD reliability pseudo labels.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--ohem_prob_dir", required=True)
    parser.add_argument("--tce_prob_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_dilate_radius", type=int, default=2)
    parser.add_argument("--far_dilate_radius", type=int, default=5)
    parser.add_argument("--tau_ohem_high", type=float, default=0.50)
    parser.add_argument("--tau_tce_low", type=float, default=0.25)
    return parser.parse_args()


def assert_train_split(path: str):
    name = os.path.basename(path).lower()
    if any(token in name for token in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("Reliability labels must be built from train split only: %s" % path)


def assert_no_forbidden_source_path(path: str):
    normalized = path.lower()
    if any(token in normalized for token in FORBIDDEN_SPLIT_TOKENS):
        raise ValueError("Reliability label sources must be train-only exports: %s" % path)


def parse_prob_dirs(value: str) -> list[Path]:
    dirs = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not dirs:
        raise ValueError("empty probability directory list")
    for path in dirs:
        assert_no_forbidden_source_path(str(path))
    return dirs


def load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr.astype(np.float32)


def find_existing(base: Path, image_id: str, exts=PROB_EXTS) -> Path:
    for ext in exts:
        path = base / (image_id + ext)
        if path.exists():
            return path
    raise FileNotFoundError(str(base / image_id))


def load_prob(base: Path, image_id: str) -> np.ndarray:
    path = find_existing(base, image_id)
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = np.squeeze(arr)
        return arr
    return load_gray(path)


def load_prob_ensemble(bases: list[Path], image_id: str) -> np.ndarray:
    probs = [load_prob(base, image_id) for base in bases]
    first_shape = probs[0].shape
    for prob in probs[1:]:
        if prob.shape != first_shape:
            raise ValueError("ensemble shape mismatch for %s: %s vs %s" % (image_id, first_shape, prob.shape))
    return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask > 0
    if radius <= 0:
        return mask
    return morphology.binary_dilation(mask, morphology.disk(int(radius)))


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_dir) / args.dataset_name
    split_path = (
        Path(args.train_split)
        if args.train_split
        else dataset_root / "img_idx" / ("train_%s.txt" % args.dataset_name)
    )
    assert_train_split(str(split_path))

    with split_path.open("r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ohem_prob_dir = Path(args.ohem_prob_dir)
    tce_prob_dirs = parse_prob_dirs(args.tce_prob_dir)
    assert_no_forbidden_source_path(str(ohem_prob_dir))

    summary = []
    for image_id in image_ids:
        mask_path = find_existing(dataset_root / "masks", image_id, exts=IMAGE_EXTS)
        gt = load_gray(mask_path)
        p_ohem = load_prob(ohem_prob_dir, image_id)
        p_tce = load_prob_ensemble(tce_prob_dirs, image_id)

        if p_ohem.shape != gt.shape or p_tce.shape != gt.shape:
            raise ValueError(
                "shape mismatch for %s: gt=%s ohem=%s tce=%s"
                % (image_id, gt.shape, p_ohem.shape, p_tce.shape)
            )

        gt_core = dilate(gt, args.target_dilate_radius)
        far_bg = ~dilate(gt, args.far_dilate_radius)

        rel_label = np.zeros_like(gt, dtype=np.float32)
        rel_valid = np.zeros_like(gt, dtype=np.float32)

        pos = gt_core
        ohem_far_candidate = far_bg & (p_ohem >= args.tau_ohem_high)
        neg = ohem_far_candidate & (p_tce <= args.tau_tce_low)

        rel_label[pos] = 1.0
        rel_valid[pos] = 1.0
        rel_label[neg] = 0.0
        rel_valid[neg] = 1.0

        leakage = int(((rel_label == 0) & (rel_valid > 0) & gt_core).sum())
        if leakage > 0:
            raise RuntimeError("target leakage in reliability negatives for %s: %d" % (image_id, leakage))

        np.savez_compressed(
            out_dir / (image_id + ".npz"),
            rel_label=rel_label.astype(np.float32),
            rel_valid=rel_valid.astype(np.float32),
            tce_prob=p_tce.astype(np.float32),
            target_core=gt_core.astype(np.uint8),
            far_bg=far_bg.astype(np.uint8),
            ohem_far_candidate=ohem_far_candidate.astype(np.uint8),
        )
        summary.append(
            (
                image_id,
                int(pos.sum()),
                int(neg.sum()),
                int(rel_valid.sum()),
                int(ohem_far_candidate.sum()),
            )
        )

    with (out_dir / "summary.csv").open("w", encoding="utf-8") as f:
        f.write("image_id,pos_pixels,neg_pixels,valid_pixels,ohem_far_candidate_pixels\n")
        for row in summary:
            f.write("%s,%d,%d,%d,%d\n" % row)

    metadata = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": args.dataset_name,
        "train_split": str(split_path.resolve()),
        "num_images": len(summary),
        "ohem_prob_dir": str(ohem_prob_dir.resolve()),
        "tce_prob_dirs": [str(path.resolve()) for path in tce_prob_dirs],
        "target_dilate_radius": int(args.target_dilate_radius),
        "far_dilate_radius": int(args.far_dilate_radius),
        "tau_ohem_high": float(args.tau_ohem_high),
        "tau_tce_low": float(args.tau_tce_low),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[build_reliability_labels] wrote %d files to %s" % (len(summary), out_dir))


if __name__ == "__main__":
    main()
