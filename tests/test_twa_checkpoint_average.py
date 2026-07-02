from collections import OrderedDict

import pytest
import torch

from tools.official.build_twa_checkpoint import average_states


def test_twa_average_floating_tensors_and_keep_final_nonfloating():
    state_a = OrderedDict(
        [
            ("weight", torch.tensor([1.0, 3.0], dtype=torch.float32)),
            ("bn.num_batches_tracked", torch.tensor(2, dtype=torch.long)),
        ]
    )
    state_b = OrderedDict(
        [
            ("weight", torch.tensor([3.0, 5.0], dtype=torch.float32)),
            ("bn.num_batches_tracked", torch.tensor(8, dtype=torch.long)),
        ]
    )

    averaged = average_states([state_a, state_b])

    assert torch.allclose(averaged["weight"], torch.tensor([2.0, 4.0]))
    assert averaged["bn.num_batches_tracked"].item() == 8


def test_twa_average_rejects_key_mismatch():
    state_a = OrderedDict([("weight", torch.ones(1))])
    state_b = OrderedDict([("other", torch.ones(1))])

    with pytest.raises(ValueError, match="keys do not match"):
        average_states([state_a, state_b])


def test_twa_average_rejects_shape_mismatch():
    state_a = OrderedDict([("weight", torch.ones(1))])
    state_b = OrderedDict([("weight", torch.ones(2))])

    with pytest.raises(ValueError, match="Shape mismatch"):
        average_states([state_a, state_b])
