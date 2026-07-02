#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net import Net  # noqa: E402
from utils import Normalized, get_img_norm_cfg, seed_pytorch  # noqa: E402

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


def load_image(path: Path, img_norm_cfg) -> tuple[np.ndarray, torch.Tensor]:
    raw = np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    img = Normalized(raw, img_norm_cfg)
    tensor = torch.from_numpy(np.ascontiguousarray(img[None, None])).float()
    return raw, tensor


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0].numpy() > 0


def window_box(y: int, x: int, scale: int, h: int, w: int):
    half = scale // 2
    y0, y1 = y - half, y + half + 1
    x0, x1 = x - half, x + half + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    return int(y0), int(y1), int(x0), int(x1)


def topq_mean(arr: np.ndarray, q: float = 0.25) -> float:
    flat = arr.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(math.floor(q * flat.size)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


def box_iou(a, b) -> float:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    return float(inter) / float(area_a + area_b - inter + 1e-6)


def nms(candidates: list[dict], topk: int, iou_thr: float) -> list[dict]:
    selected = []
    for cand in sorted(
        candidates,
        key=lambda item: (int(bool(item.get("teacher_fp_flag", False))), item["teacher_score"]),
        reverse=True,
    ):
        if len(selected) >= topk:
            break
        if all(box_iou(cand["box"], kept["box"]) <= iou_thr for kept in selected):
            selected.append(cand)
    return selected


def connected_fp_mask(prob: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[np.ndarray, list[np.ndarray]]:
    pred = prob > threshold
    fp = np.zeros_like(pred, dtype=bool)
    components = []
    label = measure.label(pred.astype(np.uint8), connectivity=2)
    for region in measure.regionprops(label):
        mask = np.zeros_like(pred, dtype=bool)
        mask[region.coords[:, 0], region.coords[:, 1]] = True
        if np.logical_and(mask, gt).any():
            continue
        fp |= mask
        components.append(mask)
    return fp, components


def local_contrast(raw: np.ndarray, box) -> float:
    y0, y1, x0, x1 = box
    patch = raw[y0:y1, x0:x1]
    ry0, ry1 = max(0, y0 - 4), min(raw.shape[0], y1 + 4)
    rx0, rx1 = max(0, x0 - 4), min(raw.shape[1], x1 + 4)
    ring = raw[ry0:ry1, rx0:rx1].copy()
    ring[(y0 - ry0):(y1 - ry0), (x0 - rx0):(x1 - rx0)] = np.nan
    outside = ring[~np.isnan(ring)]
    if patch.size == 0 or outside.size == 0:
        return 0.0
    return float((patch.mean() - outside.mean()) / (outside.std() + 1e-6))


def score_candidates(prob1: np.ndarray, prob2: np.ndarray, raw: np.ndarray, gt: np.ndarray, scales: list[int], args) -> tuple[list[dict], list[np.ndarray]]:
    h, w = gt.shape
    safe = ~dilate_mask(gt, args.dilate_radius)
    fp_mask, fp_components = connected_fp_mask(prob1, gt, args.teacher_threshold)
    candidates = []
    p1 = torch.from_numpy(prob1[None, None].astype(np.float32))
    safe_t = torch.from_numpy(safe[None, None].astype(np.float32))

    def add_candidate(y: int, x: int, scale: int, teacher_fp_flag: bool | None = None) -> None:
        box = window_box(y, x, scale, h, w)
        if box is None:
            return
        y0, y1, x0, x1 = box
        if not safe[y0:y1, x0:x1].all():
            return
        rho1 = topq_mean(prob1[y0:y1, x0:x1], args.topq)
        rho2 = topq_mean(prob2[y0:y1, x0:x1], args.topq)
        consistency = 1.0 - abs(rho1 - rho2)
        stable_score = 0.5 * (rho1 + rho2) - args.consistency_alpha * abs(rho1 - rho2)
        if teacher_fp_flag is None:
            teacher_fp_flag = bool(fp_mask[y0:y1, x0:x1].any())
        candidates.append({
            "center_y": y,
            "center_x": x,
            "window_size": scale,
            "box": [y0, y1, x0, x1],
            "teacher_score": float(stable_score),
            "rho_view1": float(rho1),
            "rho_view2": float(rho2),
            "view_consistency": float(consistency),
            "local_contrast": local_contrast(raw, box),
            "teacher_fp_flag": bool(teacher_fp_flag),
        })

    for component in fp_components:
        coords = np.argwhere(component)
        if coords.size == 0:
            continue
        cy, cx = coords.mean(axis=0)
        area = float(coords.shape[0])
        diameter = 2.0 * math.sqrt(area / math.pi)
        scale = min(scales, key=lambda item: abs(float(item) - diameter))
        add_candidate(int(round(cy)), int(round(cx)), int(scale), teacher_fp_flag=True)

    for scale in scales:
        pad = scale // 2
        avg1 = F.avg_pool2d(p1, kernel_size=scale, stride=1, padding=pad)
        max1 = F.max_pool2d(p1, kernel_size=scale, stride=1, padding=pad)
        score_map = avg1 * max1
        local_max = score_map >= (F.max_pool2d(score_map, kernel_size=2 * scale + 1, stride=1, padding=scale) - 1e-12)
        kernel = torch.ones((1, 1, scale, scale), dtype=torch.float32)
        safe_count = F.conv2d(safe_t, kernel, stride=1, padding=pad)
        valid = (safe_count >= float(scale * scale - 1e-4)) & local_max
        ys, xs = torch.where(valid[0, 0])
        if args.preselect_per_scale > 0 and ys.numel() > args.preselect_per_scale:
            local_scores = score_map[0, 0, ys, xs]
            keep = torch.topk(local_scores, k=args.preselect_per_scale, largest=True).indices
            ys = ys[keep]
            xs = xs[keep]
        for y_t, x_t in zip(ys, xs):
            y, x = int(y_t), int(x_t)
            add_candidate(y, x, scale)
    return candidates, fp_components


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PCAR persistent clutter bank from a frozen OHEM teacher.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_scales", default="5,9")
    parser.add_argument("--max_regions_per_image", type=int, default=3)
    parser.add_argument("--top_percent", type=float, default=5.0)
    parser.add_argument("--teacher_threshold", type=float, default=0.5)
    parser.add_argument("--dilate_radius", type=int, default=5)
    parser.add_argument("--consistency_alpha", type=float, default=0.5)
    parser.add_argument("--topq", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.3)
    parser.add_argument("--preselect_per_scale", type=int, default=200)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    scales = [int(item.strip()) for item in args.target_scales.split(",") if item.strip()]
    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    names = [line.strip() for line in (dataset_dir / "img_idx" / f"train_{args.dataset_name}.txt").read_text().splitlines() if line.strip()]
    if args.max_images > 0:
        names = names[: args.max_images]
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)

    net = Net("MSHNetOHEM", mode="test", loss_cfg={"mshnet_in_channels": 1}).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()

    all_candidates = []
    fp_components_by_image = {}
    with torch.no_grad():
        for idx, name in enumerate(names):
            raw, img = load_image(find_file(dataset_dir / "images", name), img_norm_cfg)
            gt = load_mask(find_file(dataset_dir / "masks", name))
            img = img.to(device)
            logit1 = net.export_logits_features(img)["logit"]
            prob1 = torch.sigmoid(logit1)[0, 0].detach().cpu().numpy().astype(np.float32)[: gt.shape[0], : gt.shape[1]]
            flip_img = torch.flip(img, dims=[3])
            logit2 = net.export_logits_features(flip_img)["logit"]
            prob2 = torch.flip(torch.sigmoid(logit2), dims=[3])[0, 0].detach().cpu().numpy().astype(np.float32)[: gt.shape[0], : gt.shape[1]]
            candidates, fp_components = score_candidates(prob1, prob2, raw, gt, scales, args)
            fp_components_by_image[name] = fp_components
            for cand in candidates:
                cand["image_id"] = name
            all_candidates.extend(candidates)
            if (idx + 1) % 100 == 0:
                print(f"Bank mining [{idx + 1}/{len(names)}]", flush=True)

    scores = [cand["teacher_score"] for cand in all_candidates]
    score_cutoff = float(np.percentile(scores, 100.0 - args.top_percent)) if scores else 1.0
    by_image = {}
    for cand in all_candidates:
        if cand["teacher_fp_flag"] or cand["teacher_score"] >= score_cutoff:
            by_image.setdefault(cand["image_id"], []).append(cand)

    records = []
    covered_fp_components = 0
    total_fp_components = 0
    for name, candidates in by_image.items():
        selected = nms(candidates, args.max_regions_per_image, args.nms_iou)
        if not selected:
            continue
        max_score = max(cand["teacher_score"] for cand in selected)
        weights = [math.exp(cand["teacher_score"] - max_score) for cand in selected]
        denom = sum(weights) + 1e-12
        for cand, weight in zip(selected, weights):
            row = dict(cand)
            row["weight"] = float(weight / denom)
            records.append(row)
    records_by_image = {}
    for row in records:
        records_by_image.setdefault(row["image_id"], []).append(row)
    for name, fp_components in fp_components_by_image.items():
        total_fp_components += len(fp_components)
        selected = records_by_image.get(name, [])
        for fp_mask in fp_components:
            hit = False
            for row in selected:
                y0, y1, x0, x1 = row["box"]
                if fp_mask[y0:y1, x0:x1].any():
                    hit = True
                    break
            covered_fp_components += int(hit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = {
        "dataset": args.dataset_name,
        "teacher_checkpoint": str(Path(args.checkpoint).resolve()),
        "target_scales": scales,
        "max_regions_per_image": args.max_regions_per_image,
        "top_percent": args.top_percent,
        "preselect_per_scale": args.preselect_per_scale,
        "score_cutoff": score_cutoff,
        "records": records,
    }
    (output_dir / "persistent_clutter_bank.json").write_text(json.dumps(bank, indent=2), encoding="utf-8")
    csv_rows = []
    for row in records:
        y0, y1, x0, x1 = row["box"]
        csv_rows.append({
            "image": row["image_id"],
            "image_id": row["image_id"],
            "rank": 0,
            "scale": row["window_size"],
            "score": row["teacher_score"],
            "teacher_score": row["teacher_score"],
            "view_consistency": row["view_consistency"],
            "local_contrast": row["local_contrast"],
            "teacher_fp_flag": int(row["teacher_fp_flag"]),
            "weight": row["weight"],
            "center_y": row["center_y"],
            "center_x": row["center_x"],
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
        })
    write_csv(
        output_dir / "persistent_clutter_bank_candidates.csv",
        csv_rows,
        ["image", "image_id", "rank", "scale", "score", "teacher_score", "view_consistency", "local_contrast", "teacher_fp_flag", "weight", "center_y", "center_x", "y0", "y1", "x0", "x1"],
    )
    active_images = len({row["image_id"] for row in records})
    summary = {
        "images": len(names),
        "active_images": active_images,
        "active_image_ratio": active_images / max(1, len(names)),
        "records": len(records),
        "avg_records_per_active_image": len(records) / max(1, active_images),
        "score_cutoff": score_cutoff,
        "preselect_per_scale": args.preselect_per_scale,
        "teacher_fp_windows": int(sum(1 for row in records if row["teacher_fp_flag"])),
        "teacher_fp_component_total": int(total_fp_components),
        "covered_teacher_fp_components": int(covered_fp_components),
        "fp_recall_at_k": covered_fp_components / max(1, total_fp_components),
        "candidate_precision_at_k": (
            sum(1 for row in records if row["teacher_fp_flag"]) / max(1, len(records))
        ),
        "candidate_activation_ratio": active_images / max(1, len(names)),
        "gt_leakage": 0,
        "bank_json": str(output_dir / "persistent_clutter_bank.json"),
        "candidate_csv": str(output_dir / "persistent_clutter_bank_candidates.csv"),
    }
    (output_dir / "persistent_clutter_bank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
