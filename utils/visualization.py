import os

import numpy as np
from PIL import Image


def normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def make_overlay(image, pred, target=None, threshold=0.5):
    base = normalize_to_uint8(image)
    if base.ndim == 2:
        overlay = np.stack([base, base, base], axis=-1)
    else:
        overlay = base.copy()

    pred_mask = np.asarray(pred) > threshold
    overlay[pred_mask] = np.array([255, 64, 64], dtype=np.uint8)

    if target is not None:
        target_mask = np.asarray(target) > threshold
        overlay[target_mask] = np.array([64, 255, 64], dtype=np.uint8)
        overlap = pred_mask & target_mask
        overlay[overlap] = np.array([255, 220, 64], dtype=np.uint8)
    return overlay


def save_overlay(image, pred, save_path, target=None, threshold=0.5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    overlay = make_overlay(image, pred, target=target, threshold=threshold)
    Image.fromarray(overlay).save(save_path)

