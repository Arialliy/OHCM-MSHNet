#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure


IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
EPS = 1e-8


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem} in {directory}")


def load_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def load_feature(exports_dir: Path, image_name: str) -> np.ndarray:
    data = np.load(exports_dir / "features" / f"{image_name}.npz")
    feature = data["decoder_feature"].astype(np.float32)
    if feature.ndim != 3:
        raise ValueError(f"Bad feature shape for {image_name}: {feature.shape}")
    return feature


def load_prob(exports_dir: Path | None, image_name: str) -> np.ndarray | None:
    if exports_dir is None:
        return None
    path = exports_dir / "probs" / f"{image_name}.npy"
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def pooled_vector(feature: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h = min(feature.shape[1], mask.shape[0])
    w = min(feature.shape[2], mask.shape[1])
    mask = mask[:h, :w].astype(bool)
    if not mask.any():
        return None
    vec = feature[:, :h, :w][:, mask].mean(axis=1)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= EPS:
        return None
    return (vec / norm).astype(np.float32)


def exact_or_bbox_component_mask(row: dict, reference_prob: np.ndarray | None, threshold: float, shape: tuple[int, int]) -> np.ndarray:
    y1 = int(float(row["bbox_y1"]))
    x1 = int(float(row["bbox_x1"]))
    y2 = int(float(row["bbox_y2"]))
    x2 = int(float(row["bbox_x2"]))
    h, w = shape
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    mask = np.zeros((h, w), dtype=bool)
    if y2 <= y1 or x2 <= x1:
        return mask

    if reference_prob is not None:
        rh = min(h, reference_prob.shape[0])
        rw = min(w, reference_prob.shape[1])
        pred = reference_prob[:rh, :rw] > threshold
        labels = measure.label(pred.astype(np.uint8), connectivity=2)
        for region in measure.regionprops(labels):
            min_row, min_col, max_row, max_col = [int(v) for v in region.bbox]
            if (min_row, min_col, max_row, max_col) == (y1, x1, y2, x2):
                coords = region.coords
                mask[coords[:, 0], coords[:, 1]] = True
                return mask

    mask[y1:y2, x1:x2] = True
    return mask


def read_hc_rows(path: Path, dataset: str, image_set: set[str] | None) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row_dataset = row.get("hcset_dataset") or row.get("dataset")
            if row_dataset != dataset:
                continue
            if image_set is not None and row["image_name"] not in image_set:
                continue
            if row.get("clutter_type", "") != "target_like_hard_clutter":
                continue
            rows.append(row)
    return rows


def read_image_set(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def collect_vectors(
    method: str,
    exports_dir: Path,
    dataset_dir: Path,
    dataset: str,
    hc_rows: list[dict],
    image_set: set[str] | None,
    reference_exports_dir: Path | None,
    threshold: float,
) -> list[dict]:
    image_names = sorted({row["image_name"] for row in hc_rows})
    if image_set is not None:
        image_names = [name for name in image_names if name in image_set]

    hc_by_image: dict[str, list[dict]] = {}
    for row in hc_rows:
        hc_by_image.setdefault(row["image_name"], []).append(row)

    vectors = []
    for image_name in image_names:
        feature = load_feature(exports_dir, image_name)
        gt = load_mask(find_file(dataset_dir / dataset / "masks", image_name))
        h = min(feature.shape[1], gt.shape[0])
        w = min(feature.shape[2], gt.shape[1])
        gt = gt[:h, :w]

        gt_labels = measure.label(gt.astype(np.uint8), connectivity=2)
        for idx, region in enumerate(measure.regionprops(gt_labels)):
            mask = gt_labels == region.label
            vec = pooled_vector(feature, mask)
            if vec is not None:
                vectors.append(
                    {
                        "method": method,
                        "label": "target",
                        "image_name": image_name,
                        "component_id": idx,
                        "area": int(region.area),
                        "vector": vec,
                    }
                )

        ref_prob = load_prob(reference_exports_dir, image_name)
        for row in hc_by_image.get(image_name, []):
            mask = exact_or_bbox_component_mask(row, ref_prob, threshold, (h, w))
            vec = pooled_vector(feature, mask)
            if vec is not None:
                vectors.append(
                    {
                        "method": method,
                        "label": "hard_clutter",
                        "image_name": image_name,
                        "component_id": int(float(row["component_id"])),
                        "area": int(float(row["area"])),
                        "vector": vec,
                    }
                )
    return vectors


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(np.matmul(a, b.T), -1.0, 1.0)


def silhouette_binary(target: np.ndarray, clutter: np.ndarray) -> float:
    all_vecs = np.concatenate([target, clutter], axis=0)
    labels = np.asarray([0] * len(target) + [1] * len(clutter), dtype=np.int32)
    if len(target) < 2 or len(clutter) < 2:
        return 0.0
    dist = cosine_distance_matrix(all_vecs, all_vecs)
    scores = []
    for idx in range(len(all_vecs)):
        same = labels == labels[idx]
        other = ~same
        same[idx] = False
        a = float(dist[idx, same].mean()) if same.any() else 0.0
        b = float(dist[idx, other].mean()) if other.any() else 0.0
        scores.append(safe_div(b - a, max(a, b)))
    return float(np.mean(scores)) if scores else 0.0


def separation_metrics(method: str, rows: list[dict]) -> dict:
    target = np.stack([row["vector"] for row in rows if row["label"] == "target"], axis=0)
    clutter = np.stack([row["vector"] for row in rows if row["label"] == "hard_clutter"], axis=0)
    target_centroid = target.mean(axis=0)
    clutter_centroid = clutter.mean(axis=0)
    target_centroid /= np.linalg.norm(target_centroid) + EPS
    clutter_centroid /= np.linalg.norm(clutter_centroid) + EPS

    centroid_cosine = float(np.dot(target_centroid, clutter_centroid))
    centroid_distance = 1.0 - centroid_cosine
    intra_target = float(cosine_distance_matrix(target, target).mean()) if len(target) else 0.0
    intra_clutter = float(cosine_distance_matrix(clutter, clutter).mean()) if len(clutter) else 0.0
    inter = float(cosine_distance_matrix(target, clutter).mean()) if len(target) and len(clutter) else 0.0
    fisher = safe_div(centroid_distance, intra_target + intra_clutter + EPS)
    margin = inter - 0.5 * (intra_target + intra_clutter)
    return {
        "method": method,
        "target_vectors": len(target),
        "hard_clutter_vectors": len(clutter),
        "target_clutter_centroid_cosine": centroid_cosine,
        "target_clutter_centroid_distance": centroid_distance,
        "mean_intra_target_distance": intra_target,
        "mean_intra_clutter_distance": intra_clutter,
        "mean_inter_class_distance": inter,
        "inter_minus_intra_margin": margin,
        "fisher_separation": fisher,
        "silhouette_cosine": silhouette_binary(target, clutter),
    }


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    return np.matmul(centered, components)


def save_projection_png(rows: list[dict], path: Path) -> None:
    width, height = 640, 480
    margin = 52
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(0, 0, 0), width=1)

    xs = np.asarray([float(row["pc1"]) for row in rows], dtype=np.float32)
    ys = np.asarray([float(row["pc2"]) for row in rows], dtype=np.float32)
    if xs.size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if abs(x_max - x_min) < EPS:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < EPS:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return margin + safe_div(value - x_min, x_max - x_min) * (width - 2 * margin)

    def sy(value: float) -> float:
        return height - margin - safe_div(value - y_min, y_max - y_min) * (height - 2 * margin)

    colors = {"target": (35, 150, 70), "hard_clutter": (210, 65, 65)}
    for row in rows:
        x = sx(float(row["pc1"]))
        y = sy(float(row["pc2"]))
        color = colors.get(row["label"], (60, 60, 60))
        r = 4 if row["label"] == "target" else 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(20, 20, 20))

    draw.text((margin, 18), "PCA feature projection: target vs fixed hard clutter", fill=(0, 0, 0))
    draw.rectangle((margin + 4, height - 36, margin + 18, height - 22), fill=colors["target"])
    draw.text((margin + 24, height - 39), "target", fill=(0, 0, 0))
    draw.rectangle((margin + 104, height - 36, margin + 118, height - 22), fill=colors["hard_clutter"])
    draw.text((margin + 124, height - 39), "hard clutter", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_method(items: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --method NAME=EXPORTS_DIR, got {item}")
        name, path = item.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Step4 feature separability analysis for target vs fixed hard clutter.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--hc_components_csv", required=True)
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--reference_exports_dir", default=None)
    parser.add_argument("--method", action="append", required=True, help="NAME=EXPORTS_DIR. May be repeated.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    image_set = read_image_set(Path(args.image_list)) if args.image_list else None
    hc_rows = read_hc_rows(Path(args.hc_components_csv), args.dataset_name, image_set)
    reference_exports_dir = Path(args.reference_exports_dir) if args.reference_exports_dir else None

    all_rows = []
    metric_rows = []
    projection_rows = []
    for method, exports_dir in parse_method(args.method):
        vectors = collect_vectors(
            method=method,
            exports_dir=exports_dir,
            dataset_dir=dataset_dir,
            dataset=args.dataset_name,
            hc_rows=hc_rows,
            image_set=image_set,
            reference_exports_dir=reference_exports_dir,
            threshold=args.threshold,
        )
        if not vectors:
            continue
        target_count = sum(row["label"] == "target" for row in vectors)
        clutter_count = sum(row["label"] == "hard_clutter" for row in vectors)
        if target_count == 0 or clutter_count == 0:
            continue

        metric_rows.append(separation_metrics(method, vectors))
        vector_matrix = np.stack([row["vector"] for row in vectors], axis=0)
        coords = pca_2d(vector_matrix)
        method_projection_rows = []
        for idx, row in enumerate(vectors):
            out_row = {
                "method": row["method"],
                "label": row["label"],
                "image_name": row["image_name"],
                "component_id": row["component_id"],
                "area": row["area"],
                "pc1": float(coords[idx, 0]),
                "pc2": float(coords[idx, 1]),
            }
            method_projection_rows.append(out_row)
            projection_rows.append(out_row)
            all_rows.append({**out_row, **{f"f{i}": float(v) for i, v in enumerate(row["vector"])}})
        save_projection_png(method_projection_rows, output_dir / "vis" / f"{method}_pca.png")

    metric_fields = [
        "method",
        "target_vectors",
        "hard_clutter_vectors",
        "target_clutter_centroid_cosine",
        "target_clutter_centroid_distance",
        "mean_intra_target_distance",
        "mean_intra_clutter_distance",
        "mean_inter_class_distance",
        "inter_minus_intra_margin",
        "fisher_separation",
        "silhouette_cosine",
    ]
    write_csv(output_dir / "feature_separation_metrics.csv", metric_rows, metric_fields)
    write_csv(
        output_dir / "feature_projection.csv",
        projection_rows,
        ["method", "label", "image_name", "component_id", "area", "pc1", "pc2"],
    )
    if all_rows:
        vector_fields = ["method", "label", "image_name", "component_id", "area", "pc1", "pc2"]
        vector_fields.extend([key for key in all_rows[0] if key.startswith("f")])
        write_csv(output_dir / "feature_vectors.csv", all_rows, vector_fields)

    summary = {
        "dataset": args.dataset_name,
        "threshold": args.threshold,
        "hc_components": len(hc_rows),
        "methods": [row["method"] for row in metric_rows],
        "metrics": metric_rows,
        "outputs": {
            "metrics": str(output_dir / "feature_separation_metrics.csv"),
            "projection": str(output_dir / "feature_projection.csv"),
            "vectors": str(output_dir / "feature_vectors.csv"),
            "visualizations": str(output_dir / "vis"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_separation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
