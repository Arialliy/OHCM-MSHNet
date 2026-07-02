#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, morphology


IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
EPS = 1e-8


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")


def load_gray(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array


def load_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def prob_to_rgb(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    red = (prob * 255).astype(np.uint8)
    blue = ((1.0 - prob) * 255).astype(np.uint8)
    green = (np.minimum(prob, 1.0 - prob) * 2.0 * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask = mask.astype(bool)
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_mask_rgb(rgb: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    mask = mask.astype(bool)
    out[mask] = 0.45 * out[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def component_info(gt_mask: np.ndarray):
    gt_label = measure.label(gt_mask.astype(np.uint8), connectivity=2)
    regions = measure.regionprops(gt_label)
    areas = {region.label: int(region.area) for region in regions}
    centroids = {region.label: np.asarray(region.centroid, dtype=np.float32) for region in regions}
    return gt_label, regions, areas, centroids


def min_center_distance(centroid: np.ndarray, gt_centroids: dict[int, np.ndarray]) -> float:
    if not gt_centroids:
        return math.inf
    return float(min(np.linalg.norm(centroid - gt_centroid) for gt_centroid in gt_centroids.values()))


def nearest_gt_area(centroid: np.ndarray, gt_areas: dict[int, int], gt_centroids: dict[int, np.ndarray]) -> int:
    if not gt_centroids:
        return 0
    nearest_label = min(gt_centroids, key=lambda label: np.linalg.norm(centroid - gt_centroids[label]))
    return int(gt_areas[nearest_label])


def max_region_iou(coords: np.ndarray, area: int, gt_label: np.ndarray, gt_areas: dict[int, int]) -> float:
    labels, counts = np.unique(gt_label[coords[:, 0], coords[:, 1]], return_counts=True)
    best = 0.0
    for label, intersection in zip(labels, counts):
        label = int(label)
        if label == 0:
            continue
        union = area + gt_areas[label] - int(intersection)
        best = max(best, safe_div(intersection, union))
    return best


def scale_similarity(area: int, gt_area_median: float) -> float:
    if area <= 0 or gt_area_median <= 0:
        return 0.0
    return float(math.exp(-abs(math.log((float(area) + EPS) / (gt_area_median + EPS)))))


def local_contrast(image: np.ndarray, component_mask: np.ndarray, ring_radius: int):
    footprint = morphology.disk(max(1, int(ring_radius)))
    dilated = morphology.dilation(component_mask, footprint)
    ring = np.logical_and(dilated, ~component_mask)
    inside_values = image[component_mask]
    ring_values = image[ring]
    if inside_values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    inside_mean = float(inside_values.mean())
    if ring_values.size == 0:
        return inside_mean, 0.0, 0.0, 0.0
    ring_mean = float(ring_values.mean())
    ring_std = float(ring_values.std())
    delta = inside_mean - ring_mean
    z_score = delta / (ring_std + EPS)
    return inside_mean, ring_mean, float(delta), float(z_score)


def multiscale_response_proxy(prob: np.ndarray, component_mask: np.ndarray) -> float:
    means = []
    h, w = prob.shape
    component = component_mask.astype(np.uint8)
    for scale in (1, 2, 4):
        if scale == 1:
            smooth_prob = prob
        else:
            small_w = max(1, w // scale)
            small_h = max(1, h // scale)
            pil = Image.fromarray((prob * 255.0).clip(0, 255).astype(np.uint8))
            pil = pil.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            smooth_prob = np.asarray(pil, dtype=np.float32) / 255.0
        values = smooth_prob[component.astype(bool)]
        means.append(float(values.mean()) if values.size else 0.0)
    return float(min(means))


def classify_component(
    area: int,
    mean_prob: float,
    local_contrast_z: float,
    distance_to_target: float,
    max_iou: float,
    sim: float,
    args,
) -> str:
    if max_iou > args.iou_eps or distance_to_target <= args.center_radius:
        return "target_near_confusion"
    if area <= args.noise_area:
        return "sensor_noise_hot_pixel"
    if (
        mean_prob >= args.high_prob_threshold
        and sim >= args.scale_similarity_threshold
        and local_contrast_z >= args.local_contrast_threshold
    ):
        return "target_like_hard_clutter"
    if args.gt_area_median > 0 and area >= 4.0 * args.gt_area_median:
        return "large_structured_clutter"
    return "weak_background_fp"


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_boxes(image: Image.Image, rows, color, width=2):
    draw = ImageDraw.Draw(image)
    for row in rows:
        draw.rectangle(
            [int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]) - 1, int(row["bbox_y2"]) - 1],
            outline=color,
            width=width,
        )


def make_image_visualization(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, prob: np.ndarray, fp_rows, path: Path):
    gray = to_uint8(image)
    raw_panel = np.stack([gray, gray, gray], axis=-1)
    gt_panel = overlay_mask(gray, gt, (0, 255, 0))
    prob_panel = prob_to_rgb(prob)
    pred_panel = overlay_mask(gray, pred, (255, 0, 0))

    pil_prob = Image.fromarray(prob_panel)
    pil_pred = Image.fromarray(pred_panel)
    draw_boxes(pil_prob, fp_rows, (0, 255, 255), width=2)
    draw_boxes(pil_pred, fp_rows, (0, 255, 255), width=2)

    vis = np.concatenate([raw_panel, gt_panel, np.asarray(pil_prob), np.asarray(pil_pred)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(path)


def save_component_crop(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, row: dict, path: Path, margin=16):
    h, w = image.shape
    x1 = max(0, int(row["bbox_x1"]) - margin)
    y1 = max(0, int(row["bbox_y1"]) - margin)
    x2 = min(w, int(row["bbox_x2"]) + margin)
    y2 = min(h, int(row["bbox_y2"]) + margin)
    gray = to_uint8(image[y1:y2, x1:x2])
    panel = np.stack([gray, gray, gray], axis=-1)
    panel = overlay_mask_rgb(panel, pred[y1:y2, x1:x2], (255, 0, 0))
    panel = overlay_mask_rgb(panel, gt[y1:y2, x1:x2], (0, 255, 0))
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    draw.rectangle(
        [
            int(row["bbox_x1"]) - x1,
            int(row["bbox_y1"]) - y1,
            int(row["bbox_x2"]) - x1 - 1,
            int(row["bbox_y2"]) - y1 - 1,
        ],
        outline=(0, 255, 255),
        width=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def histogram_rows(values, label, bins):
    counts, edges = np.histogram(np.asarray(values, dtype=np.float32), bins=bins, range=(0.0, 1.0))
    rows = []
    for idx, count in enumerate(counts):
        rows.append(
            {
                "distribution": label,
                "bin_left": float(edges[idx]),
                "bin_right": float(edges[idx + 1]),
                "count": int(count),
            }
        )
    return rows


def save_histogram_png(target_values, fp_values, path: Path, bins=20):
    width, height = 760, 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 30, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    target_counts, _ = np.histogram(target_values, bins=bins, range=(0.0, 1.0))
    fp_counts, _ = np.histogram(fp_values, bins=bins, range=(0.0, 1.0))
    max_count = max(int(target_counts.max()) if target_counts.size else 0, int(fp_counts.max()) if fp_counts.size else 0, 1)
    bar_w = plot_w / bins

    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(0, 0, 0), width=1)
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(0, 0, 0), width=1)
    for idx in range(bins):
        x0 = margin_l + idx * bar_w
        x_mid = x0 + bar_w / 2.0
        target_h = plot_h * safe_div(target_counts[idx], max_count)
        fp_h = plot_h * safe_div(fp_counts[idx], max_count)
        draw.rectangle((x0 + 2, margin_t + plot_h - target_h, x_mid - 1, margin_t + plot_h), fill=(40, 170, 80))
        draw.rectangle((x_mid + 1, margin_t + plot_h - fp_h, x0 + bar_w - 2, margin_t + plot_h), fill=(220, 70, 70))

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = margin_l + tick * plot_w
        draw.line((x, margin_t + plot_h, x, margin_t + plot_h + 5), fill=(0, 0, 0), width=1)
        draw.text((x - 10, margin_t + plot_h + 10), f"{tick:.2f}", fill=(0, 0, 0))
    draw.text((margin_l, 8), "Mean probability distribution: target components vs FP clutter components", fill=(0, 0, 0))
    draw.rectangle((margin_l + 5, height - 25, margin_l + 20, height - 10), fill=(40, 170, 80))
    draw.text((margin_l + 25, height - 27), "target", fill=(0, 0, 0))
    draw.rectangle((margin_l + 105, height - 25, margin_l + 120, height - 10), fill=(220, 70, 70))
    draw.text((margin_l + 125, height - 27), "FP clutter", fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Step1 hard-clutter diagnosis from Step0 exported predictions.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--exports_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou_eps", type=float, default=0.0)
    parser.add_argument("--center_radius", type=float, default=3.0)
    parser.add_argument("--ring_radius", type=int, default=6)
    parser.add_argument("--top_k_vis", type=int, default=80)
    parser.add_argument("--high_prob_threshold", type=float, default=0.5)
    parser.add_argument("--scale_similarity_threshold", type=float, default=0.35)
    parser.add_argument("--local_contrast_threshold", type=float, default=0.5)
    parser.add_argument("--noise_area", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) / args.dataset_name
    exports_dir = Path(args.exports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_list_path = dataset_dir / "img_idx" / f"test_{args.dataset_name}.txt"
    image_names = [line.strip() for line in test_list_path.read_text().splitlines() if line.strip()]

    all_gt_areas = []
    for name in image_names:
        gt = load_mask(find_file(dataset_dir / "masks", name))
        _, gt_regions, _, _ = component_info(gt)
        all_gt_areas.extend([int(region.area) for region in gt_regions])
    args.gt_area_median = float(np.median(all_gt_areas)) if all_gt_areas else 0.0

    fp_rows = []
    target_response_values = []
    fp_response_values = []
    total_pred_components = 0
    total_gt_components = 0
    target_near_components = 0
    image_fp_rows = {}

    for image_idx, name in enumerate(image_names):
        image = load_gray(find_file(dataset_dir / "images", name))
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)
        logit_path = exports_dir / "logits" / f"{name}.npy"
        logit = np.load(logit_path).astype(np.float32) if logit_path.exists() else np.zeros_like(prob)

        h = min(image.shape[0], gt.shape[0], prob.shape[0])
        w = min(image.shape[1], gt.shape[1], prob.shape[1])
        image = image[:h, :w]
        gt = gt[:h, :w]
        prob = prob[:h, :w]
        logit = logit[:h, :w]
        pred = prob > args.threshold

        gt_label, gt_regions, gt_areas, gt_centroids = component_info(gt)
        total_gt_components += len(gt_regions)
        for gt_region in gt_regions:
            coords = gt_region.coords
            values = prob[coords[:, 0], coords[:, 1]]
            target_response_values.append(float(values.mean()) if values.size else 0.0)

        pred_label = measure.label(pred.astype(np.uint8), connectivity=2)
        pred_regions = measure.regionprops(pred_label)
        total_pred_components += len(pred_regions)
        image_fp_rows[name] = []

        for component_idx, region in enumerate(pred_regions):
            area = int(region.area)
            coords = region.coords
            centroid = np.asarray(region.centroid, dtype=np.float32)
            max_iou = max_region_iou(coords, area, gt_label, gt_areas)
            distance = min_center_distance(centroid, gt_centroids)

            if max_iou > args.iou_eps or distance <= args.center_radius:
                target_near_components += 1
                continue

            component_mask = pred_label == region.label
            inside_mean, ring_mean, contrast_delta, contrast_z = local_contrast(image, component_mask, args.ring_radius)
            mean_prob = float(prob[coords[:, 0], coords[:, 1]].mean())
            max_prob = float(prob[coords[:, 0], coords[:, 1]].max())
            mean_logit = float(logit[coords[:, 0], coords[:, 1]].mean())
            max_logit = float(logit[coords[:, 0], coords[:, 1]].max())
            nearest_area = nearest_gt_area(centroid, gt_areas, gt_centroids)
            sim = scale_similarity(area, args.gt_area_median)
            multi_scale_response = multiscale_response_proxy(prob, component_mask)
            clutter_type = classify_component(area, mean_prob, contrast_z, distance, max_iou, sim, args)
            min_row, min_col, max_row, max_col = [int(v) for v in region.bbox]

            row = {
                "dataset": args.dataset_name,
                "seed": args.seed if args.seed is not None else "",
                "image_index": image_idx,
                "image_name": name,
                "component_id": component_idx,
                "area": area,
                "bbox_x1": min_col,
                "bbox_y1": min_row,
                "bbox_x2": max_col,
                "bbox_y2": max_row,
                "centroid_x": float(centroid[1]),
                "centroid_y": float(centroid[0]),
                "mean_prob": mean_prob,
                "max_prob": max_prob,
                "mean_logit": mean_logit,
                "max_logit": max_logit,
                "local_inside_mean": inside_mean,
                "local_ring_mean": ring_mean,
                "local_contrast": contrast_delta,
                "local_contrast_z": contrast_z,
                "distance_to_target": distance,
                "nearest_gt_area": nearest_area,
                "gt_area_median": args.gt_area_median,
                "scale_similarity": sim,
                "multi_scale_response": multi_scale_response,
                "max_iou_to_gt": max_iou,
                "clutter_type": clutter_type,
                "manual_label": "",
                "is_high_response": int(mean_prob >= args.high_prob_threshold),
            }
            fp_rows.append(row)
            image_fp_rows[name].append(row)
            fp_response_values.append(mean_prob)

    fp_fieldnames = [
        "dataset",
        "seed",
        "image_index",
        "image_name",
        "component_id",
        "area",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "centroid_x",
        "centroid_y",
        "mean_prob",
        "max_prob",
        "mean_logit",
        "max_logit",
        "local_inside_mean",
        "local_ring_mean",
        "local_contrast",
        "local_contrast_z",
        "distance_to_target",
        "nearest_gt_area",
        "gt_area_median",
        "scale_similarity",
        "multi_scale_response",
        "max_iou_to_gt",
        "clutter_type",
        "manual_label",
        "is_high_response",
    ]
    write_csv(output_dir / "step1_fp_components.csv", fp_rows, fp_fieldnames)

    type_counter = Counter(row["clutter_type"] for row in fp_rows)
    type_rows = []
    total_fp = len(fp_rows)
    for clutter_type, count in sorted(type_counter.items()):
        type_rows.append(
            {
                "dataset": args.dataset_name,
                "seed": args.seed if args.seed is not None else "",
                "clutter_type": clutter_type,
                "count": count,
                "fraction_of_fp": safe_div(count, total_fp),
            }
        )
    write_csv(output_dir / "step1_fp_type_stats.csv", type_rows, ["dataset", "seed", "clutter_type", "count", "fraction_of_fp"])

    hist_rows = []
    hist_rows.extend(histogram_rows(target_response_values, "target_component_mean_prob", bins=20))
    hist_rows.extend(histogram_rows(fp_response_values, "fp_component_mean_prob", bins=20))
    write_csv(output_dir / "step1_prob_distributions.csv", hist_rows, ["distribution", "bin_left", "bin_right", "count"])
    save_histogram_png(target_response_values, fp_response_values, output_dir / "vis" / "target_vs_fp_prob_histogram.png", bins=20)

    ranked_rows = sorted(
        fp_rows,
        key=lambda row: (
            row["clutter_type"] == "target_like_hard_clutter",
            float(row["mean_prob"]),
            float(row["scale_similarity"]),
            float(row["local_contrast_z"]),
        ),
        reverse=True,
    )
    selected = ranked_rows[: args.top_k_vis]
    selected_names = sorted({row["image_name"] for row in selected})
    selected_by_name = {name: [row for row in selected if row["image_name"] == name] for name in selected_names}

    for name in selected_names:
        image = load_gray(find_file(dataset_dir / "images", name))
        gt = load_mask(find_file(dataset_dir / "masks", name))
        prob = np.load(exports_dir / "probs" / f"{name}.npy").astype(np.float32)
        h = min(image.shape[0], gt.shape[0], prob.shape[0])
        w = min(image.shape[1], gt.shape[1], prob.shape[1])
        image = image[:h, :w]
        gt = gt[:h, :w]
        prob = prob[:h, :w]
        pred = prob > args.threshold
        make_image_visualization(image, gt, pred, prob, selected_by_name[name], output_dir / "vis" / "fp_components" / f"{name}.png")

    for rank, row in enumerate(selected[: min(60, len(selected))], start=1):
        image = load_gray(find_file(dataset_dir / "images", row["image_name"]))
        gt = load_mask(find_file(dataset_dir / "masks", row["image_name"]))
        prob = np.load(exports_dir / "probs" / f"{row['image_name']}.npy").astype(np.float32)
        h = min(image.shape[0], gt.shape[0], prob.shape[0])
        w = min(image.shape[1], gt.shape[1], prob.shape[1])
        save_component_crop(
            image[:h, :w],
            gt[:h, :w],
            prob[:h, :w] > args.threshold,
            row,
            output_dir / "vis" / "top_fp_crops" / f"{rank:03d}_{row['image_name']}_c{row['component_id']}.png",
        )

    summary = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "threshold": args.threshold,
        "num_images": len(image_names),
        "total_gt_components": total_gt_components,
        "total_pred_components": total_pred_components,
        "false_positive_components": total_fp,
        "target_near_components_excluded": target_near_components,
        "high_response_fp_components": int(sum(int(row["is_high_response"]) for row in fp_rows)),
        "target_like_hard_clutter_components": int(type_counter.get("target_like_hard_clutter", 0)),
        "hard_clutter_fraction_of_fp": safe_div(type_counter.get("target_like_hard_clutter", 0), total_fp),
        "high_response_fraction_of_fp": safe_div(sum(int(row["is_high_response"]) for row in fp_rows), total_fp),
        "gt_area_median": args.gt_area_median,
        "clutter_type_counts": dict(type_counter),
        "outputs": {
            "fp_components": str(output_dir / "step1_fp_components.csv"),
            "fp_type_stats": str(output_dir / "step1_fp_type_stats.csv"),
            "prob_distributions": str(output_dir / "step1_prob_distributions.csv"),
            "histogram": str(output_dir / "vis" / "target_vs_fp_prob_histogram.png"),
            "fp_visualizations": str(output_dir / "vis" / "fp_components"),
            "top_fp_crops": str(output_dir / "vis" / "top_fp_crops"),
        },
    }
    (output_dir / "step1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
