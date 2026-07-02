from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REQUIRED_ECDV_GATE_B_CHECKS = (
    "target_dilate_overlap_pixels",
    "decoys_per_image_mean",
    "evidence_response_success_ratio",
    "mean_prob_gain",
    "area_in_target_range_ratio",
    "flat_artifact_ratio",
)


def check_ecdv_gate_b_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    failures = []
    if summary.get("gate_pass") is not True:
        failures.append("summary_gate_pass_not_true")
    checks = summary.get("checks", {})
    for key in REQUIRED_ECDV_GATE_B_CHECKS:
        if key not in checks:
            failures.append("missing_check_%s" % key)
        elif checks[key] is not True:
            failures.append("failed_check_%s" % key)
    if failures:
        raise SystemExit("ECDV Gate-B decoy bank failed; training is blocked:\n- " + "\n- ".join(failures))
    return summary


class ECDVDecoyBank:
    def __init__(self, bank_dir):
        self.bank_dir = Path(bank_dir)
        rows_path = self.bank_dir / "decoy_rows.csv"
        if not rows_path.exists():
            raise FileNotFoundError("Missing ECDV decoy rows: %s" % rows_path)
        self.by_image = {}
        with rows_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.by_image.setdefault(row["image_id"], []).append(row)
        if not self.by_image:
            raise ValueError("Empty ECDV decoy bank: %s" % rows_path)

    def sample(self, image_id):
        rows = self.by_image.get(str(image_id))
        if not rows:
            raise KeyError("No ECDV decoy found for image_id=%s" % image_id)
        row = rows[np.random.randint(0, len(rows))]
        residual = np.load(self.bank_dir / row["residual_path"]).astype(np.float32)
        mask = np.load(self.bank_dir / row["mask_path"]).astype(np.float32)
        return residual, mask, row

    def sample_batch(self, image_ids, image_tensor):
        residuals, masks, rows = [], [], []
        for image_id in image_ids:
            residual_np, mask_np, row = self.sample(image_id)
            residuals.append(torch.from_numpy(residual_np).unsqueeze(0))
            masks.append(torch.from_numpy(mask_np).unsqueeze(0))
            rows.append(row)
        residual = torch.stack(residuals, dim=0).to(device=image_tensor.device, dtype=image_tensor.dtype)
        mask = torch.stack(masks, dim=0).to(device=image_tensor.device, dtype=image_tensor.dtype)
        if residual.shape[-2:] != image_tensor.shape[-2:]:
            residual = F.interpolate(residual, size=image_tensor.shape[-2:], mode="bilinear", align_corners=True)
            mask = F.interpolate(mask, size=image_tensor.shape[-2:], mode="nearest")
        return residual, mask, rows


def apply_decoy_batch(image_tensor, residual, mask):
    return image_tensor + residual * (mask > 0).to(dtype=image_tensor.dtype)
