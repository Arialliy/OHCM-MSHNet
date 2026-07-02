#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TrainSetLoader
from net import Net
from tools.official.build_twa_checkpoint import torch_load
from utils import get_img_norm_cfg, seed_pytorch


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def reset_bn_stats(module: torch.nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, _BatchNorm):
            child.reset_running_stats()
            count += 1
    return count


def bn_state_snapshot(module: torch.nn.Module) -> dict:
    snapshot = {}
    for name, child in module.named_modules():
        if isinstance(child, _BatchNorm):
            snapshot[name] = {
                "running_mean": child.running_mean.detach().cpu().clone(),
                "running_var": child.running_var.detach().cpu().clone(),
            }
    return snapshot


def bn_state_changed(before: dict, module: torch.nn.Module) -> bool:
    after = bn_state_snapshot(module)
    for name, item in before.items():
        if name not in after:
            continue
        if not torch.equal(item["running_mean"], after[name]["running_mean"]):
            return True
        if not torch.equal(item["running_var"], after[name]["running_var"]):
            return True
    return False


def recalibrate_bn(model: Net, loader: DataLoader, device: torch.device, num_batches: int, epoch: int) -> int:
    model.train()
    batches = 0
    with torch.no_grad():
        for batch in loader:
            img = batch[0].to(device)
            if img.shape[0] <= 1:
                continue
            _ = model(img, epoch=epoch)
            batches += 1
            if batches >= num_batches:
                break
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibrate BatchNorm statistics for a TWA checkpoint using train split only.")
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--lambda_variant", type=float, default=0.2)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--no_reset_bn", action="store_true")
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    dataset = TrainSetLoader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        patch_size=args.patch_size,
        img_norm_cfg=img_norm_cfg,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.threads,
        shuffle=True,
        drop_last=True,
    )

    model = Net(args.model_name, mode="test", loss_cfg=vars(args)).to(device)
    checkpoint = torch_load(args.checkpoint)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    bn_layers = 0 if args.no_reset_bn else reset_bn_stats(model)
    before = bn_state_snapshot(model)
    batches = recalibrate_bn(model, loader, device, args.num_batches, epoch=args.mshnet_warm_epoch + 1)
    changed = bn_state_changed(before, model)

    out = {
        "state_dict": model.state_dict(),
        "twa_bn_meta": {
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset": args.dataset_name,
            "num_batches_requested": args.num_batches,
            "num_batches_used": batches,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "reset_bn": not args.no_reset_bn,
            "bn_layers": bn_layers if not args.no_reset_bn else len(before),
            "bn_stats_changed": changed,
        },
    }
    if isinstance(checkpoint, dict) and "twa_meta" in checkpoint:
        out["twa_meta"] = checkpoint["twa_meta"]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(json.dumps(out["twa_bn_meta"], indent=2), flush=True)


if __name__ == "__main__":
    main()
