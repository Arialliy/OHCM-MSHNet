#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from utils import get_img_norm_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Audit train-only dense TCE soft labels.")
    parser.add_argument("--dataset", "--dataset_name", dest="dataset_name", default="NUDT-SIRST")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--tce_soft_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--far_radius", type=int, default=7)
    parser.add_argument("--absdiff_min", type=float, default=0.001)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def train_ids(dataset_dir: str, dataset_name: str) -> list[str]:
    path = Path(dataset_dir) / dataset_name / "img_idx" / f"train_{dataset_name}.txt"
    if not path.exists():
        fallback = Path(dataset_dir) / dataset_name / "img_idx" / "train.txt"
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(str(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * int(radius) + 1
    return (F.max_pool2d(tensor, kernel_size=k, stride=1, padding=int(radius))[0, 0].numpy() > 0)


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.split.lower() != "train":
        raise ValueError("TCE soft-label audit is train-only.")
    tce_soft_dir = Path(args.tce_soft_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = train_ids(args.dataset_dir, args.dataset_name)
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    dataset.test_list = ids
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

    rows = []
    all_p_mean = []
    all_p_std = []
    target_means = []
    background_means = []
    far_background_means = []
    absdiff_means = []
    valid_teacher_count = 0
    missing = []
    existing_npz = {path.stem for path in tce_soft_dir.glob("*.npz")}

    for idx, (_img, gt_mask, size, image_name) in enumerate(loader):
        h, w = size_to_int(size[0]), size_to_int(size[1])
        name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
        path = tce_soft_dir / f"{name}.npz"
        if not path.exists():
            missing.append(name)
            continue
        data = np.load(path)
        p_tce = data["teacher_prob"].astype(np.float32) if "teacher_prob" in data else data["p_tce"].astype(np.float32)
        if p_tce.shape != (h, w):
            raise ValueError("Shape mismatch for %s: p_tce=%s image=%s" % (name, p_tce.shape, (h, w)))
        gt = gt_mask[0, 0, :h, :w].numpy() > 0
        bg = ~gt
        far_bg = ~binary_dilate(gt, args.far_radius)
        if "student_prob" in data:
            p_ohem = data["student_prob"].astype(np.float32)
        elif "p_ohem_400" in data:
            p_ohem = data["p_ohem_400"].astype(np.float32)
        else:
            p_ohem = data["p_400"].astype(np.float32)
        absdiff = np.abs(p_tce - p_ohem)
        p_mean = float(p_tce.mean())
        p_std = float(p_tce.std())
        target_mean = float(p_tce[gt].mean()) if gt.any() else float("nan")
        bg_mean = float(p_tce[bg].mean()) if bg.any() else float("nan")
        far_bg_mean = float(p_tce[far_bg].mean()) if far_bg.any() else float("nan")
        absdiff_mean = float(absdiff.mean())
        valid_teacher = bool(np.isfinite(p_tce).all() and p_std > 1e-8 and 0.0 <= p_tce.min() <= p_tce.max() <= 1.0)
        valid_teacher_count += int(valid_teacher)
        all_p_mean.append(p_mean)
        all_p_std.append(p_std)
        target_means.append(target_mean)
        background_means.append(bg_mean)
        far_background_means.append(far_bg_mean)
        absdiff_means.append(absdiff_mean)
        rows.append(
            {
                "image_name": name,
                "p_tce_mean": p_mean,
                "p_tce_std": p_std,
                "p_tce_min": float(p_tce.min()),
                "p_tce_max": float(p_tce.max()),
                "target_p_tce_mean": target_mean,
                "background_p_tce_mean": bg_mean,
                "far_background_p_tce_mean": far_bg_mean,
                "teacher_student_absdiff_mean": absdiff_mean,
                "valid_teacher": int(valid_teacher),
            }
        )
        if (idx + 1) % 100 == 0:
            print("Audited TCE soft labels [%d/%d]" % (idx + 1, len(loader)), flush=True)

    expected = set(ids)
    extra_npz = sorted(existing_npz - expected - {"summary"})
    num_npz = len(existing_npz & expected)
    images_with_valid_teacher_ratio = valid_teacher_count / max(1, len(ids))
    p_tce_mean = safe_mean(all_p_mean)
    p_tce_std = safe_mean(all_p_std)
    target_p_tce_mean = safe_mean(target_means)
    background_p_tce_mean = safe_mean(background_means)
    far_background_p_tce_mean = safe_mean(far_background_means)
    teacher_student_absdiff_mean = safe_mean(absdiff_means)
    gate_checks = {
        "num_npz_eq_train_count": num_npz == len(ids),
        "missing_npz_eq_0": len(missing) == 0,
        "images_with_valid_teacher_ratio_eq_1": images_with_valid_teacher_ratio == 1.0,
        "p_tce_not_constant": p_tce_std > 1e-8,
        "target_gt_background": target_p_tce_mean > background_p_tce_mean,
        "global_absdiff_mean_is_diagnostic_only": True,
    }
    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "tce_soft_dir": str(tce_soft_dir),
        "train_image_count": len(ids),
        "num_npz": num_npz,
        "missing_npz": len(missing),
        "extra_npz": len(extra_npz),
        "p_tce_mean": p_tce_mean,
        "p_tce_std": p_tce_std,
        "target_p_tce_mean": target_p_tce_mean,
        "background_p_tce_mean": background_p_tce_mean,
        "far_background_p_tce_mean": far_background_p_tce_mean,
        "teacher_student_absdiff_mean": teacher_student_absdiff_mean,
        "previous_required_min_absdiff_mean": args.absdiff_min,
        "global_absdiff_mean_is_diagnostic_only": True,
        "num_images_with_valid_teacher": valid_teacher_count,
        "images_with_valid_teacher_ratio": images_with_valid_teacher_ratio,
        "gate_checks": gate_checks,
        "gate_pass": bool(all(gate_checks.values())),
        "outputs": {
            "per_image": str(output_dir / "per_image.csv"),
            "missing": str(output_dir / "missing.txt"),
        },
    }
    write_csv(
        output_dir / "per_image.csv",
        rows,
        [
            "image_name",
            "p_tce_mean",
            "p_tce_std",
            "p_tce_min",
            "p_tce_max",
            "target_p_tce_mean",
            "background_p_tce_mean",
            "far_background_p_tce_mean",
            "teacher_student_absdiff_mean",
            "valid_teacher",
        ],
    )
    (output_dir / "missing.txt").write_text("\n".join(missing), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
