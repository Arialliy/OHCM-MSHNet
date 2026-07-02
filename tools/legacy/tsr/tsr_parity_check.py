#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import TrainSetLoader  # noqa: E402
from net import Net  # noqa: E402
from utils import seed_pytorch  # noqa: E402


def max_abs_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().cpu())


def compare_predictions(pred_a, pred_b) -> dict:
    masks_a, final_a = pred_a
    masks_b, final_b = pred_b
    diffs = {"final_logit_max_abs": max_abs_tensor(final_a, final_b)}
    for idx, (mask_a, mask_b) in enumerate(zip(masks_a, masks_b)):
        diffs[f"mask{idx}_max_abs"] = max_abs_tensor(mask_a, mask_b)
    diffs["forward_max_abs"] = max(diffs.values()) if diffs else 0.0
    return diffs


def grad_map(net: Net) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in net.named_parameters():
        if param.grad is not None:
            out[name] = param.grad.detach().clone()
    return out


def compare_grads(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    keys = sorted(set(a) | set(b))
    max_diff = 0.0
    worst_key = ""
    missing = []
    for key in keys:
        if key not in a or key not in b:
            missing.append(key)
            continue
        diff = float((a[key] - b[key]).abs().max().cpu())
        if diff > max_diff:
            max_diff = diff
            worst_key = key
    return {"grad_max_abs": max_diff, "grad_worst_key": worst_key, "grad_missing_keys": missing}


def make_seed_worker(seed: int):
    def _seed_worker(worker_id: int):
        import random
        import numpy as np

        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _seed_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="First-batch parity check for MSHNetOHEM vs TSR wrapper with lambda_region=0.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=120)
    parser.add_argument("--target_scales", default="3,5,7")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_set = TrainSetLoader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        patch_size=args.patchSize,
        img_norm_cfg=None,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset=train_set,
        num_workers=args.threads,
        batch_size=args.batchSize,
        shuffle=True,
        worker_init_fn=make_seed_worker(args.seed),
        generator=generator,
    )
    img, gt_mask = next(iter(loader))
    img = img.to(device)
    gt_mask = gt_mask.to(device)

    base_cfg = {
        "mshnet_warm_epoch": 5,
        "mshnet_in_channels": 1,
        "lambda_variant": 0.2,
        "ohem_ratio": 0.01,
    }
    tsr_cfg = dict(base_cfg)
    tsr_cfg.update({
        "tsr_lambda_region": 0.0,
        "tsr_region_start_epoch": 60,
        "tsr_region_end_epoch": 100,
        "tsr_target_scales": args.target_scales,
        "tsr_region_loss_mode": "rank",
    })

    seed_pytorch(args.seed)
    base = Net(model_name="MSHNetOHEM", mode="train", loss_cfg=base_cfg).to(device)
    tsr = Net(model_name="MSHNetOHEM", mode="train", loss_cfg=tsr_cfg).to(device)
    tsr.load_state_dict(base.state_dict())
    base.train()
    tsr.train()

    pred_base = base(img, epoch=args.epoch)
    pred_tsr = tsr(img, epoch=args.epoch)
    forward_diffs = compare_predictions(pred_base, pred_tsr)

    loss_base = base.loss(pred_base, gt_mask, epoch=args.epoch)
    loss_tsr = tsr.loss(pred_tsr, gt_mask, epoch=args.epoch)
    loss_base["total"].backward()
    loss_tsr["total"].backward()
    grad_diffs = compare_grads(grad_map(base), grad_map(tsr))

    result = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "epoch": args.epoch,
        "device": str(device),
        "target_scales": args.target_scales,
        **forward_diffs,
        "base_total_loss": float(loss_base["total"].detach().cpu()),
        "tsr_total_loss": float(loss_tsr["total"].detach().cpu()),
        "total_loss_abs_diff": float(abs(loss_base["total"].detach().cpu() - loss_tsr["total"].detach().cpu())),
        "base_variant_loss": float(loss_base["variant_loss"].detach().cpu()),
        "tsr_variant_loss": float(loss_tsr["variant_loss"].detach().cpu()),
        "tsr_region_loss": float(loss_tsr["region_loss"].detach().cpu()),
        "tsr_lambda_region": float(loss_tsr["lambda_region"].detach().cpu()),
        **grad_diffs,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
