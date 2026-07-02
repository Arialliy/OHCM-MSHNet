import pytest
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tcsr_bank import TCSRBank


def test_tcsr_bank_loads_item_and_resizes(tmp_path):
    bank_root = tmp_path / "bank"
    bank_root.mkdir()
    torch.save(
        {
            "neg_weight": torch.ones(2, 3),
            "protect_weight": torch.zeros(1, 2, 3),
            "ref_prob": torch.arange(6, dtype=torch.float32).view(2, 3),
        },
        bank_root / "000001.pt",
    )

    bank = TCSRBank(bank_root)
    item = bank.get("000001.png", size=(4, 6))

    assert item.image_id == "000001"
    assert item.neg_weight.shape == (1, 4, 6)
    assert item.protect_weight.shape == (1, 4, 6)
    assert item.ref_prob.shape == (1, 4, 6)
    assert item.neg_weight.dtype == torch.float32


def test_tcsr_bank_batches_items(tmp_path):
    bank_root = tmp_path / "bank"
    bank_root.mkdir()
    for image_id, value in [("a", 1.0), ("b", 2.0)]:
        torch.save(
            {
                "neg_weight": torch.full((1, 2, 2), value),
                "protect_weight": torch.zeros(1, 2, 2),
                "ref_prob": torch.full((1, 2, 2), value / 2.0),
            },
            bank_root / f"{image_id}.pt",
        )

    batch = TCSRBank(bank_root).batch(["a", "b"], device=torch.device("cpu"), size=(2, 2))

    assert batch["neg_weight"].shape == (2, 1, 2, 2)
    assert batch["protect_weight"].shape == (2, 1, 2, 2)
    assert batch["ref_prob"].shape == (2, 1, 2, 2)
    assert torch.allclose(batch["neg_weight"][1], torch.full((1, 2, 2), 2.0))


def test_tcsr_bank_missing_item_raises_keyerror(tmp_path):
    bank_root = tmp_path / "bank"
    bank_root.mkdir()
    bank = TCSRBank(bank_root)

    with pytest.raises(KeyError):
        bank.get("missing")
