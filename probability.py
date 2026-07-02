from __future__ import annotations

import torch


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """Convert model output logits to foreground probability.

    Supported outputs:
    - Bx1xHxW binary logits
    - Bx2xHxW two-class logits where channel 1 is foreground
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected BCHW logits, got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    raise ValueError(f"Unsupported output channel count: {logits.shape[1]}")
