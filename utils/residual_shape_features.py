from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - exercised only when scipy is absent.
    ndi = None


EPS = 1e-6


@dataclass(frozen=True)
class ResidualShapeWeights:
    compactness: float = 1.0
    fill_ratio: float = 0.5
    anisotropy: float = 0.15
    center_surround: float = 0.5
    radial_symmetry: float = 0.5
    dog_peakness: float = 0.5


def _as_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("component coords must have shape [N, 2]")
    return coords


def _component_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    if ndi is not None:
        return ndi.binary_dilation(mask.astype(bool), structure=np.ones((3, 3), dtype=bool), iterations=iterations)
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    if ndi is not None:
        return ndi.binary_erosion(mask.astype(bool), structure=np.ones((3, 3), dtype=bool), iterations=1)
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    return (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )


def _gaussian_filter(values: np.ndarray, sigma: float) -> np.ndarray:
    if ndi is not None:
        return ndi.gaussian_filter(values, sigma=sigma)
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=padded)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=tmp)


def compactness_from_mask(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = float(
        ((center) & (~padded[:-2, 1:-1])).sum()
        + ((center) & (~padded[2:, 1:-1])).sum()
        + ((center) & (~padded[1:-1, :-2])).sum()
        + ((center) & (~padded[1:-1, 2:])).sum()
    )
    if perimeter <= 0:
        return 0.0
    return float(min(1.0, (4.0 * math.pi * area) / (perimeter * perimeter + EPS)))


def bbox_fill_ratio(coords: np.ndarray) -> float:
    coords = _as_coords(coords)
    if coords.size == 0:
        return 0.0
    height = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    width = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    return float(coords.shape[0] / max(1, height * width))


def coordinate_anisotropy(coords: np.ndarray) -> float:
    coords = _as_coords(coords).astype(np.float64)
    if coords.shape[0] < 3:
        return 1.0
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return float((eigvals[-1] + EPS) / (eigvals[0] + EPS))


def center_surround_contrast(residual: np.ndarray, mask: np.ndarray, ring_radius: int = 3) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    mask = mask.astype(bool)
    if not mask.any():
        return 0.0
    ring = _binary_dilate(mask, ring_radius) & (~mask)
    center_mean = float(residual[mask].mean())
    ring_mean = float(residual[ring].mean()) if ring.any() else center_mean
    return center_mean - ring_mean


def radial_symmetry(residual: np.ndarray, coords: np.ndarray) -> float:
    coords = _as_coords(coords)
    if coords.size == 0:
        return 0.0
    residual = np.asarray(residual, dtype=np.float64)
    center = coords.mean(axis=0)
    values = []
    y_high = coords[:, 0] >= center[0]
    x_high = coords[:, 1] >= center[1]
    for y_flag, x_flag in ((False, False), (False, True), (True, False), (True, True)):
        keep = (y_high == y_flag) & (x_high == x_flag)
        if keep.any():
            yy, xx = coords[keep, 0], coords[keep, 1]
            values.append(float(residual[yy, xx].mean()))
    if not values:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if abs(mean) < EPS:
        return 0.0
    return float(np.clip(1.0 - (values.std() / (abs(mean) + EPS)), 0.0, 1.0))


def dog_peakness(residual: np.ndarray, coords: np.ndarray, sigma_small: float = 1.0, sigma_large: float = 2.0) -> float:
    coords = _as_coords(coords)
    if coords.size == 0:
        return 0.0
    residual = np.asarray(residual, dtype=np.float64)
    dog = _gaussian_filter(residual, sigma_small) - _gaussian_filter(residual, sigma_large)
    yy, xx = coords[:, 0], coords[:, 1]
    scale = float(residual.std()) + EPS
    return float(dog[yy, xx].mean() / scale)


def residual_shape_features(
    residual: np.ndarray,
    coords: np.ndarray,
    ring_radius: int = 3,
    sigma_small: float = 1.0,
    sigma_large: float = 2.0,
    weights: ResidualShapeWeights | None = None,
) -> dict:
    residual = np.asarray(residual, dtype=np.float64)
    coords = _as_coords(coords)
    weights = weights or ResidualShapeWeights()
    mask = _component_mask(coords, residual.shape)
    compactness = compactness_from_mask(mask)
    fill = bbox_fill_ratio(coords)
    anisotropy = coordinate_anisotropy(coords)
    csr = center_surround_contrast(residual, mask, ring_radius=ring_radius)
    symmetry = radial_symmetry(residual, coords)
    peakness = dog_peakness(residual, coords, sigma_small=sigma_small, sigma_large=sigma_large)
    score = (
        weights.compactness * compactness
        + weights.fill_ratio * fill
        - weights.anisotropy * math.log1p(max(0.0, anisotropy - 1.0))
        + weights.center_surround * csr
        + weights.radial_symmetry * symmetry
        + weights.dog_peakness * peakness
    )
    return {
        "area": int(coords.shape[0]),
        "compactness": float(compactness),
        "bbox_fill_ratio": float(fill),
        "anisotropy": float(anisotropy),
        "center_surround": float(csr),
        "radial_symmetry": float(symmetry),
        "dog_peakness": float(peakness),
        "shape_score": float(score),
    }


def parse_shape_weights(value: str | None) -> ResidualShapeWeights:
    if value is None or str(value).strip() == "":
        return ResidualShapeWeights()
    parts = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parts) != 6:
        raise ValueError("shape weights must contain 6 comma-separated numbers")
    return ResidualShapeWeights(*parts)
