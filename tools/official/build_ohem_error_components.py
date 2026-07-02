#!/usr/bin/env python3
"""Build OHEM train-split error components from a frozen checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import measure
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


@dataclass
class ComponentRecord:
    component_id: int
    component_type: str
    area: int
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    centroid_y: float
    centroid_x: float
    max_prob: float
    mean_prob: float
    sum_prob: float
    prob_contrast: float
    image_contrast: float
    ring_mean_prob: float
    ring_mean_intensity: float
    component_mean_intensity: float
    compactness: float
    aspect_ratio: float
    distance_to_nearest_target: float
    is_target_like_area: int
    is_target_like_area_loose: int
    is_nonflat: int
    is_detached_far_fp: int
    is_boundary_excess: int
    is_detached_near_fp: int
    target_leakage_pixels: int


def parse_args():
    parser = argparse.ArgumentParser(description="Build OHEM false-positive connected components.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_dilate_radius", type=int, default=5)
    parser.add_argument("--near_radius", type=float, default=12.0)
    parser.add_argument("--far_radius", type=float, default=24.0)
    parser.add_argument("--min_component_area", type=int, default=1)
    parser.add_argument("--prob_contrast_min", type=float, default=0.05)
    parser.add_argument("--image_contrast_min", type=float, default=0.50)
    parser.add_argument("--ring_inner_radius", type=int, default=1)
    parser.add_argument("--ring_outer_radius", type=int, default=5)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))[0, 0].numpy() > 0


def connected_components(mask: np.ndarray):
    labeled = measure.label(mask.astype(np.uint8), connectivity=2)
    return labeled, int(labeled.max())


def connected_regions(mask: np.ndarray):
    return measure.regionprops(measure.label(mask.astype(np.uint8), connectivity=2))


def split_ids(args, dataset) -> tuple[list[str], str]:
    if args.split == "train":
        path = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"train_{args.dataset_name}.txt"
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], str(path)
    return list(dataset.test_list), "test"


def torch_load_checkpoint(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(net: Net, checkpoint_path: str, device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def target_component_areas(gt: np.ndarray) -> list[int]:
    return [int(region.area) for region in connected_regions(gt.astype(bool))]


def area_stats_from_values(values: list[int]) -> dict:
    if not values:
        return {
            "target_area_p01": 1.0,
            "target_area_p05": 1.0,
            "target_area_p25": 1.0,
            "target_area_median": 1.0,
            "target_area_p75": 1.0,
            "target_area_p95": float("inf"),
            "target_area_p99": float("inf"),
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "target_area_p01": float(np.percentile(arr, 1)),
        "target_area_p05": float(np.percentile(arr, 5)),
        "target_area_p25": float(np.percentile(arr, 25)),
        "target_area_median": float(np.median(arr)),
        "target_area_p75": float(np.percentile(arr, 75)),
        "target_area_p95": float(np.percentile(arr, 95)),
        "target_area_p99": float(np.percentile(arr, 99)),
    }


def classify_component_type(
    component: np.ndarray,
    gt: np.ndarray,
    gt_dilated: np.ndarray,
    dist_to_gt: np.ndarray,
    near_radius: float,
    far_radius: float,
) -> tuple[str, float]:
    if bool((component & gt).any()):
        return "target_hit_or_overlap", 0.0
    if bool((component & gt_dilated).any()):
        return "boundary_excess", 0.0
    distance = float(dist_to_gt[component].min()) if component.any() else float("inf")
    if distance <= near_radius:
        return "detached_near_fp", distance
    if distance >= far_radius:
        return "detached_far_fp", distance
    return "ambiguous_mid_fp", distance


def classify_components(
    pred: np.ndarray,
    gt: np.ndarray,
    prob: np.ndarray | None = None,
    image: np.ndarray | None = None,
    target_area_stats: dict | None = None,
    target_dilate_radius: int = 5,
    near_radius: float = 12.0,
    far_radius: float = 24.0,
    min_component_area: int = 1,
    prob_contrast_min: float = 0.05,
    image_contrast_min: float = 0.50,
    ring_inner_radius: int = 1,
    ring_outer_radius: int = 5,
) -> list[ComponentRecord]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if prob is None:
        prob = pred.astype(np.float32)
    if image is None:
        image = np.zeros_like(prob, dtype=np.float32)
    if target_area_stats is None:
        target_area_stats = area_stats_from_values(target_component_areas(gt))

    gt_dilated = binary_dilate(gt, target_dilate_radius)
    dist_to_gt = distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf, dtype=np.float32)
    labeled, n_components = connected_components(pred)
    records: list[ComponentRecord] = []
    for component_id in range(1, n_components + 1):
        component = labeled == component_id
        area = int(component.sum())
        if area < min_component_area:
            continue
        component_type, distance = classify_component_type(
            component, gt, gt_dilated, dist_to_gt, near_radius=near_radius, far_radius=far_radius
        )
        ys, xs = np.where(component)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_area = max(1, bbox_h * bbox_w)
        aspect_ratio = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))

        ring = binary_dilate(component, ring_outer_radius) & (~binary_dilate(component, ring_inner_radius)) & (~gt)
        ring_prob = prob[ring] if ring.any() else np.asarray([0.0], dtype=np.float32)
        ring_img = image[ring] if ring.any() else np.asarray([0.0], dtype=np.float32)
        component_prob = prob[component]
        component_img = image[component]
        prob_contrast = float(component_prob.max() - ring_prob.mean())
        image_contrast = float(abs(component_img.mean() - ring_img.mean()) / (ring_img.std() + 1e-6))
        is_nonflat = int(prob_contrast >= prob_contrast_min or image_contrast >= image_contrast_min)
        is_target_like_area = int(
            target_area_stats["target_area_p05"] <= area <= target_area_stats["target_area_p95"]
        )
        is_target_like_area_loose = int(
            target_area_stats["target_area_p01"] <= area <= target_area_stats["target_area_p99"]
        )
        records.append(
            ComponentRecord(
                component_id=int(component_id),
                component_type=component_type,
                area=area,
                bbox_y0=int(ys.min()),
                bbox_x0=int(xs.min()),
                bbox_y1=int(ys.max()),
                bbox_x1=int(xs.max()),
                centroid_y=float(ys.mean()),
                centroid_x=float(xs.mean()),
                max_prob=float(component_prob.max()),
                mean_prob=float(component_prob.mean()),
                sum_prob=float(component_prob.sum()),
                prob_contrast=prob_contrast,
                image_contrast=image_contrast,
                ring_mean_prob=float(ring_prob.mean()),
                ring_mean_intensity=float(ring_img.mean()),
                component_mean_intensity=float(component_img.mean()),
                compactness=safe_div(area, bbox_area),
                aspect_ratio=aspect_ratio,
                distance_to_nearest_target=distance,
                is_target_like_area=is_target_like_area,
                is_target_like_area_loose=is_target_like_area_loose,
                is_nonflat=is_nonflat,
                is_detached_far_fp=int(component_type == "detached_far_fp"),
                is_boundary_excess=int(component_type == "boundary_excess"),
                is_detached_near_fp=int(component_type == "detached_near_fp"),
                target_leakage_pixels=int((component & gt).sum()),
            )
        )
    return records


def build_components_for_image(
    prob: np.ndarray,
    gt: np.ndarray,
    image: np.ndarray | None = None,
    target_area_stats: dict | None = None,
    threshold: float = 0.5,
    target_dilate_radius: int = 5,
    near_radius: float = 12.0,
    far_radius: float = 24.0,
    min_component_area: int = 1,
    prob_contrast_min: float = 0.05,
    image_contrast_min: float = 0.50,
    ring_inner_radius: int = 1,
    ring_outer_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ComponentRecord]]:
    pred = prob >= threshold
    gt_bool = gt.astype(bool)
    records = classify_components(
        pred,
        gt_bool,
        prob=prob,
        image=image,
        target_area_stats=target_area_stats,
        target_dilate_radius=target_dilate_radius,
        near_radius=near_radius,
        far_radius=far_radius,
        min_component_area=min_component_area,
        prob_contrast_min=prob_contrast_min,
        image_contrast_min=image_contrast_min,
        ring_inner_radius=ring_inner_radius,
        ring_outer_radius=ring_outer_radius,
    )
    fp_component_mask = np.zeros_like(pred, dtype=bool)
    boundary_excess_mask = np.zeros_like(pred, dtype=bool)
    detached_far_fp_mask = np.zeros_like(pred, dtype=bool)
    labeled, _ = connected_components(pred)
    for record in records:
        component = labeled == record.component_id
        if record.component_type != "target_hit_or_overlap":
            fp_component_mask |= component
        if record.component_type == "boundary_excess":
            boundary_excess_mask |= component
        if record.component_type == "detached_far_fp":
            detached_far_fp_mask |= component
    return fp_component_mask, boundary_excess_mask, detached_far_fp_mask, records


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def direct_probability(net: Net, img: torch.Tensor, h: int, w: int) -> np.ndarray:
    logit = net.export_logits_features(img)["logit"][:, :, :h, :w]
    return foreground_probability(logit)[0, 0].detach().cpu().numpy().astype(np.float32)


def collect_target_area_stats(dataset, args) -> dict:
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    areas: list[int] = []
    for _img, gt_mask, size, _name in loader:
        h, w = size_to_int(size[0]), size_to_int(size[1])
        gt = gt_mask[0, 0, :h, :w].numpy() > 0
        areas.extend(target_component_areas(gt))
    return area_stats_from_values(areas)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    per_image_dir = out_dir / "per_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    image_ids, split_source = split_ids(args, dataset)
    dataset.test_list = image_ids
    target_area_stats = collect_target_area_stats(dataset, args)
    loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = load_checkpoint(net, args.checkpoint, device)

    all_rows = []
    image_rows = []
    with torch.no_grad():
        for idx, (img, gt_mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = gt_mask[0, 0, :h, :w].numpy() > 0
            image_np = img[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            prob = direct_probability(net, img, h, w)
            fp_mask, boundary_mask, detached_mask, records = build_components_for_image(
                prob=prob,
                gt=gt,
                image=image_np,
                target_area_stats=target_area_stats,
                threshold=args.threshold,
                target_dilate_radius=args.target_dilate_radius,
                near_radius=args.near_radius,
                far_radius=args.far_radius,
                min_component_area=args.min_component_area,
                prob_contrast_min=args.prob_contrast_min,
                image_contrast_min=args.image_contrast_min,
                ring_inner_radius=args.ring_inner_radius,
                ring_outer_radius=args.ring_outer_radius,
            )
            for record in records:
                row = asdict(record)
                row["image_id"] = name
                all_rows.append(row)
            image_rows.append(
                {
                    "image_id": name,
                    "component_count": len(records),
                    "detached_far_fp_components": sum(1 for r in records if r.component_type == "detached_far_fp"),
                    "boundary_excess_components": sum(1 for r in records if r.component_type == "boundary_excess"),
                    "fp_component_pixels": int(fp_mask.sum()),
                    "boundary_excess_pixels": int(boundary_mask.sum()),
                    "detached_far_fp_pixels": int(detached_mask.sum()),
                    "target_pixels": int(gt.sum()),
                    "far_background_pixels": int((~binary_dilate(gt, args.target_dilate_radius)).sum()),
                    "ohem_budget_pixels": max(1, int(np.ceil((~binary_dilate(gt, args.target_dilate_radius)).sum() * 0.01))),
                }
            )
            np.savez_compressed(
                per_image_dir / f"{name}.npz",
                fp_component_mask=fp_mask,
                boundary_excess_mask=boundary_mask,
                detached_far_fp_mask=detached_mask,
                anchor_prob=prob.astype(np.float16),
                pred_mask=(prob >= args.threshold),
                gt_mask=gt,
            )
            if (idx + 1) % 100 == 0:
                print(f"Built [{idx + 1}/{len(loader)}]", flush=True)

    component_fields = [
        "image_id",
        "component_id",
        "component_type",
        "area",
        "bbox_y0",
        "bbox_x0",
        "bbox_y1",
        "bbox_x1",
        "centroid_y",
        "centroid_x",
        "max_prob",
        "mean_prob",
        "sum_prob",
        "prob_contrast",
        "image_contrast",
        "ring_mean_prob",
        "ring_mean_intensity",
        "component_mean_intensity",
        "compactness",
        "aspect_ratio",
        "distance_to_nearest_target",
        "is_target_like_area",
        "is_target_like_area_loose",
        "is_nonflat",
        "is_detached_far_fp",
        "is_boundary_excess",
        "is_detached_near_fp",
        "target_leakage_pixels",
    ]
    image_fields = [
        "image_id",
        "component_count",
        "detached_far_fp_components",
        "boundary_excess_components",
        "fp_component_pixels",
        "boundary_excess_pixels",
        "detached_far_fp_pixels",
        "target_pixels",
        "far_background_pixels",
        "ohem_budget_pixels",
    ]
    write_csv(out_dir / "error_components.csv", all_rows, component_fields)
    write_csv(out_dir / "image_level_counts.csv", image_rows, image_fields)

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "split_source": split_source,
        "model_name": args.model_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "output_dir": str(out_dir),
        "num_images": len(image_ids),
        "num_written": len(image_rows),
        "component_count_total": len(all_rows),
        "num_images_with_components": sum(1 for row in image_rows if int(row["component_count"]) > 0),
        "target_hit_components": sum(1 for row in all_rows if row["component_type"] == "target_hit_or_overlap"),
        "target_leakage_components": sum(
            1
            for row in all_rows
            if row["component_type"] != "target_hit_or_overlap" and int(row["target_leakage_pixels"]) > 0
        ),
        "target_leakage_pixels_total": sum(
            int(row["target_leakage_pixels"])
            for row in all_rows
            if row["component_type"] != "target_hit_or_overlap"
        ),
        "target_area_stats": target_area_stats,
        "params": {
            "threshold": args.threshold,
            "target_dilate_radius": args.target_dilate_radius,
            "near_radius": args.near_radius,
            "far_radius": args.far_radius,
            "min_component_area": args.min_component_area,
            "prob_contrast_min": args.prob_contrast_min,
            "image_contrast_min": args.image_contrast_min,
            "ring_inner_radius": args.ring_inner_radius,
            "ring_outer_radius": args.ring_outer_radius,
        },
        "outputs": {
            "error_components": str(out_dir / "error_components.csv"),
            "image_level_counts": str(out_dir / "image_level_counts.csv"),
            "per_image": str(per_image_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
