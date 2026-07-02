import numpy as np
import torch
from skimage import measure


def _to_tensor(value):
    if torch.is_tensor(value):
        return value.detach()
    return torch.from_numpy(np.asarray(value))


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_4d_tensor(value):
    value = _to_tensor(value).float()
    if value.dim() == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.dim() == 3:
        value = value.unsqueeze(1)
    elif value.dim() != 4:
        raise ValueError("Unknown target dimension")
    return value


def _binary_tensor(value, threshold=0.0):
    return (_as_4d_tensor(value) > threshold).float()


def _size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def _safe_div(numerator, denominator):
    return numerator / (denominator + np.spacing(1))


class mIoU():
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pix_acc = _safe_div(self.total_correct, self.total_label)
        iou = _safe_div(self.total_inter, self.total_union)
        return float(pix_acc), float(np.mean(iou))

    def reset(self):
        self.total_inter = np.array([0.0])
        self.total_union = np.array([0.0])
        self.total_correct = 0.0
        self.total_label = 0.0


class nIoU():
    def __init__(self, score_thresh=0.5):
        super(nIoU, self).__init__()
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        pred = _binary_tensor(preds, self.score_thresh).cpu().numpy().astype(np.int64)
        target = _binary_tensor(labels, 0.0).cpu().numpy().astype(np.int64)
        if pred.shape != target.shape:
            raise AssertionError("Predict and Label Shape Don't Match")

        for idx in range(pred.shape[0]):
            intersection = np.logical_and(pred[idx] > 0, target[idx] > 0).sum()
            union = np.logical_or(pred[idx] > 0, target[idx] > 0).sum()
            self.total_inter = np.append(self.total_inter, intersection)
            self.total_union = np.append(self.total_union, union)

    def get(self):
        iou = _safe_div(self.total_inter, self.total_union)
        return iou, float(np.mean(iou))

    def reset(self):
        self.total_inter = np.array([], dtype=np.float64)
        self.total_union = np.array([], dtype=np.float64)


class F1():
    def __init__(self, score_thresh=0.5):
        super(F1, self).__init__()
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        pred = _binary_tensor(preds, self.score_thresh)
        target = _binary_tensor(labels, 0.0)
        if pred.shape != target.shape:
            raise AssertionError("Predict and Label Shape Don't Match")

        self.tp += float((pred * target).sum().item())
        self.fp += float((pred * (1.0 - target)).sum().item())
        self.fn += float(((1.0 - pred) * target).sum().item())

    def get(self):
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        return float(precision), float(recall), float(f1)

    def value(self):
        return self.get()[-1]

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0


class BinaryMetricsGPU():
    def __init__(self, threshold=0.5, device=None, eps=1e-8):
        self.threshold = threshold
        self.device = device
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_inter = None
        self.total_union = None
        self.total_tp = None
        self.total_fp = None
        self.total_fn = None
        self.total_pixels = None
        self.niou_sum = None
        self.niou_count = 0

    def _new_scalar(self, value=0.0, device=None):
        if device is None:
            device = self.device
        return torch.tensor(float(value), device=device)

    def _ensure_state(self, device):
        if self.total_inter is not None:
            return
        self.total_inter = self._new_scalar(device=device)
        self.total_union = self._new_scalar(device=device)
        self.total_tp = self._new_scalar(device=device)
        self.total_fp = self._new_scalar(device=device)
        self.total_fn = self._new_scalar(device=device)
        self.total_pixels = self._new_scalar(device=device)
        self.niou_sum = self._new_scalar(device=device)

    def update(self, pred, target):
        pred = _as_4d_tensor(pred)
        target = _as_4d_tensor(target).to(pred.device)
        self._ensure_state(pred.device)

        pred = (pred > self.threshold).float()
        target = (target > 0).float()
        if pred.shape != target.shape:
            raise AssertionError("Predict and Label Shape Don't Match")

        tp_map = pred * target
        fp_map = pred * (1.0 - target)
        fn_map = (1.0 - pred) * target
        union_map = (pred + target - tp_map).clamp_min(0.0)

        tp = tp_map.sum()
        fp = fp_map.sum()
        fn = fn_map.sum()
        union = union_map.sum()

        self.total_inter += tp
        self.total_union += union
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_pixels += torch.tensor(target.numel(), device=target.device, dtype=target.dtype)

        sample_inter = tp_map.flatten(1).sum(dim=1)
        sample_union = union_map.flatten(1).sum(dim=1)
        self.niou_sum += (sample_inter / (sample_union + self.eps)).sum()
        self.niou_count += int(pred.shape[0])

    def get(self):
        if self.total_inter is None:
            return {
                'pixAcc': 0.0,
                'mIoU': 0.0,
                'nIoU': 0.0,
                'PixelRecall': 0.0,
                'Fa': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
            }

        precision = self.total_tp / (self.total_tp + self.total_fp + self.eps)
        recall = self.total_tp / (self.total_tp + self.total_fn + self.eps)
        f1 = 2.0 * precision * recall / (precision + recall + self.eps)
        miou = self.total_inter / (self.total_union + self.eps)
        niou = self.niou_sum / max(self.niou_count, 1)
        fa = self.total_fp / (self.total_pixels + self.eps)

        return {
            'pixAcc': float(recall.detach().cpu()),
            'mIoU': float(miou.detach().cpu()),
            'nIoU': float(niou.detach().cpu()),
            'PixelRecall': float(recall.detach().cpu()),
            'Fa': float(fa.detach().cpu()),
            'Precision': float(precision.detach().cpu()),
            'Recall': float(recall.detach().cpu()),
            'F1': float(f1.detach().cpu()),
        }


class PD_FA():
    def __init__(self, distance_threshold=3):
        super(PD_FA, self).__init__()
        self.distance_threshold = distance_threshold
        self.reset()

    def update(self, preds, labels, size=None):
        pred = (_to_numpy(preds) > 0).astype(np.int64).squeeze()
        target = (_to_numpy(labels) > 0).astype(np.int64).squeeze()
        if pred.shape != target.shape:
            raise AssertionError("Predict and Label Shape Don't Match")

        pred_regions = measure.regionprops(measure.label(pred, connectivity=2))
        target_regions = measure.regionprops(measure.label(target, connectivity=2))

        matched_pred = np.zeros(pred.shape, dtype=np.float32)
        used_pred = set()
        matched_targets = 0

        for target_region in target_regions:
            target_centroid = np.array(target_region.centroid)
            for pred_idx, pred_region in enumerate(pred_regions):
                if pred_idx in used_pred:
                    continue
                pred_centroid = np.array(pred_region.centroid)
                distance = np.linalg.norm(pred_centroid - target_centroid)
                if distance < self.distance_threshold:
                    used_pred.add(pred_idx)
                    matched_targets += 1
                    matched_pred[pred_region.coords[:, 0], pred_region.coords[:, 1]] = 1
                    break

        false_alarm_pixels = np.logical_and(pred > 0, matched_pred <= 0).sum()
        if size is None:
            image_pixels = pred.shape[-2] * pred.shape[-1]
        else:
            image_pixels = _size_to_int(size[0]) * _size_to_int(size[1])

        self.dismatch_pixel += float(false_alarm_pixels)
        self.all_pixel += float(image_pixels)
        self.PD += float(matched_targets)
        self.target += float(len(target_regions))

    def get(self):
        final_pd = _safe_div(self.PD, self.target)
        final_fa = _safe_div(self.dismatch_pixel, self.all_pixel)
        return float(final_pd), float(final_fa)

    def reset(self):
        self.dismatch_pixel = 0.0
        self.all_pixel = 0.0
        self.PD = 0.0
        self.target = 0.0


class Pd(PD_FA):
    def get(self):
        return super(Pd, self).get()[0]


class Fa(PD_FA):
    def get(self):
        return super(Fa, self).get()[1]


def batch_pix_accuracy(output, target):
    predict = _binary_tensor(output, 0.0)
    target = _binary_tensor(target, 0.0)
    if predict.shape != target.shape:
        raise AssertionError("Predict and Label Shape Don't Match")

    pixel_labeled = float((target > 0).sum().item())
    pixel_correct = float(((predict == target).float() * (target > 0).float()).sum().item())
    if pixel_correct > pixel_labeled:
        raise AssertionError("Correct area should be smaller than Labeled")
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    predict = _binary_tensor(output, 0.0).cpu().numpy().astype(np.int64)
    target = _binary_tensor(target, 0.0).cpu().numpy().astype(np.int64)
    if predict.shape != target.shape:
        raise AssertionError("Predict and Label Shape Don't Match")

    intersection = np.logical_and(predict > 0, target > 0).sum()
    area_pred = (predict > 0).sum()
    area_lab = (target > 0).sum()
    area_union = area_pred + area_lab - intersection
    if intersection > area_union:
        raise AssertionError("Error: Intersection area should be smaller than Union area")
    return np.array([float(intersection)]), np.array([float(area_union)])


def Params(model, trainable_only=False):
    parameters = model.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return int(sum(param.numel() for param in parameters))


def FLOPs(model, input_size=(1, 1, 256, 256), device=None):
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("FLOPs requires thop. Install it or use cal_params.py in the prepared environment.") from exc

    was_training = model.training
    if device is None:
        device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    model.eval()
    with torch.no_grad():
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    model.train(was_training)
    return int(flops)


def model_complexity(model, input_size=(1, 1, 256, 256), device=None):
    return {
        "Params": Params(model),
        "FLOPs": FLOPs(model, input_size=input_size, device=device),
    }


def format_params(params):
    return "{:.6f}M".format(params / 1e6)


def format_flops(flops):
    return "{:.6f}GFLOPs".format(flops / 1e9)
