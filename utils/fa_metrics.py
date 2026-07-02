import numpy as np
import torch


def to_numpy(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask)


def false_alarm_rate(pred, target, threshold=0.5):
    pred = to_numpy(pred) > threshold
    target = to_numpy(target) > threshold
    false_alarm = np.logical_and(pred, np.logical_not(target)).sum()
    pixels = np.prod(target.shape)
    return float(false_alarm / max(pixels, 1))


class FalseAlarmMetric(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def update(self, pred, target):
        pred = to_numpy(pred) > self.threshold
        target = to_numpy(target) > self.threshold
        self.false_alarm += np.logical_and(pred, np.logical_not(target)).sum()
        self.total_pixels += np.prod(target.shape)

    def get(self):
        return float(self.false_alarm / max(self.total_pixels, 1))

    def reset(self):
        self.false_alarm = 0
        self.total_pixels = 0

