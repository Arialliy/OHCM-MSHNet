import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.residual_shape_features import residual_shape_features


def test_residual_shape_features_prefer_compact_blob_over_line():
    residual = np.full((20, 20), 0.1, dtype=np.float32)
    blob_y, blob_x = np.mgrid[8:11, 8:11]
    blob_coords = np.column_stack([blob_y.reshape(-1), blob_x.reshape(-1)])
    line_coords = np.column_stack([np.full(7, 15), np.arange(4, 11)])
    residual[blob_coords[:, 0], blob_coords[:, 1]] = 3.0
    residual[line_coords[:, 0], line_coords[:, 1]] = 3.0

    blob = residual_shape_features(residual, blob_coords)
    line = residual_shape_features(residual, line_coords)

    assert blob["compactness"] > line["compactness"]
    assert line["anisotropy"] > blob["anisotropy"]
    assert blob["shape_score"] > line["shape_score"]


def test_residual_shape_features_are_finite_for_single_pixel():
    residual = np.ones((5, 5), dtype=np.float32)
    features = residual_shape_features(residual, np.asarray([[2, 2]]))

    for value in features.values():
        assert np.isfinite(value)
    assert features["area"] == 1
    assert features["anisotropy"] == 1.0
