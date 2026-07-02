import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evidence_conditioned_decoy import generate_evidence_conditioned_decoy


class BrightnessEvidence(torch.nn.Module):
    def export_logits_features(self, image):
        return {"logit": image * 5.0}


def test_decoy_increases_evidence_response():
    image = torch.zeros(1, 1, 32, 32)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 14:18, 14:18] = 1.0
    model = BrightnessEvidence()

    _image_aug, pseudo_mask, stats = generate_evidence_conditioned_decoy(
        image,
        target,
        model,
        center=(4, 4),
        patch_radius=3,
        steps=8,
        lr=0.2,
        target_dilate_radius=5,
        max_delta=0.5,
        response_threshold=0.60,
        min_gain=0.10,
        topk=4,
    )

    assert stats["accepted"] is True
    assert stats["prob_after_max"] > stats["prob_before_max"] + 0.10
    assert int(pseudo_mask.sum().item()) > 0
    assert stats["target_dilate_overlap_pixels"] == 0
