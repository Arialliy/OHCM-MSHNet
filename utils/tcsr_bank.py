from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class TCSRBankItem:
    image_id: str
    neg_weight: torch.Tensor
    protect_weight: torch.Tensor
    ref_prob: Optional[torch.Tensor] = None


class TCSRBank:
    """Load train-only trajectory-consensus sparse reliability maps."""

    def __init__(self, bank_root: str | Path):
        self.bank_root = Path(bank_root)
        if not self.bank_root.exists():
            raise FileNotFoundError(f"TCSR bank not found: {self.bank_root}")

    @staticmethod
    def _normalize_image_id(image_id) -> str:
        if isinstance(image_id, bytes):
            image_id = image_id.decode("utf-8")
        return Path(str(image_id)).stem

    @staticmethod
    def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected CHW or HW tensor, got shape={tuple(x.shape)}")
        return x.float()

    def get(self, image_id: str, device=None, size=None) -> TCSRBankItem:
        key = self._normalize_image_id(image_id)
        path = self.bank_root / f"{key}.pt"
        if not path.exists():
            raise KeyError(f"TCSR bank item missing for image_id={image_id}, path={path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")

        neg_weight = self._ensure_chw(data["neg_weight"])
        protect_weight = self._ensure_chw(data["protect_weight"])
        ref_prob = data.get("ref_prob", None)
        if ref_prob is not None:
            ref_prob = self._ensure_chw(ref_prob)

        if size is not None:
            neg_weight = F.interpolate(neg_weight[None], size=size, mode="nearest")[0]
            protect_weight = F.interpolate(protect_weight[None], size=size, mode="nearest")[0]
            if ref_prob is not None:
                ref_prob = F.interpolate(ref_prob[None], size=size, mode="bilinear", align_corners=False)[0]

        if device is not None:
            neg_weight = neg_weight.to(device)
            protect_weight = protect_weight.to(device)
            if ref_prob is not None:
                ref_prob = ref_prob.to(device)

        return TCSRBankItem(
            image_id=key,
            neg_weight=neg_weight,
            protect_weight=protect_weight,
            ref_prob=ref_prob,
        )

    def batch(self, image_ids, device, size) -> Dict[str, torch.Tensor | None]:
        items = [self.get(image_id, device=device, size=size) for image_id in image_ids]
        ref_prob = None
        if items and items[0].ref_prob is not None:
            ref_prob = torch.stack([item.ref_prob for item in items], dim=0)
        return {
            "neg_weight": torch.stack([item.neg_weight for item in items], dim=0),
            "protect_weight": torch.stack([item.protect_weight for item in items], dim=0),
            "ref_prob": ref_prob,
        }
