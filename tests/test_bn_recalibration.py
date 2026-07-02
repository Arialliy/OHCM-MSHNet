import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.official.recalibrate_bn import bn_state_changed, bn_state_snapshot, recalibrate_bn, reset_bn_stats


class TinyBNNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, x, epoch=0):
        return self.bn(x)


def test_reset_bn_stats_resets_running_statistics():
    model = TinyBNNet()
    model.bn.running_mean.fill_(3.0)
    model.bn.running_var.fill_(5.0)

    count = reset_bn_stats(model)

    assert count == 1
    assert torch.allclose(model.bn.running_mean, torch.zeros_like(model.bn.running_mean))
    assert torch.allclose(model.bn.running_var, torch.ones_like(model.bn.running_var))


def test_recalibrate_bn_updates_running_statistics():
    model = TinyBNNet()
    reset_bn_stats(model)
    before = bn_state_snapshot(model)
    data = torch.ones(8, 1, 4, 4) * 4.0
    masks = torch.zeros(8, 1, 4, 4)
    loader = DataLoader(TensorDataset(data, masks), batch_size=4, drop_last=True)

    batches = recalibrate_bn(model, loader, torch.device("cpu"), num_batches=2, epoch=1)

    assert batches == 2
    assert bn_state_changed(before, model)
    assert model.bn.running_mean.item() > 0.0
