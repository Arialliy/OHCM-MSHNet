#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TrainSetLoader
from net import Net
from utils import get_img_norm_cfg, get_optimizer, seed_pytorch


def group_name(param_name: str) -> str:
    if param_name.startswith("model.backbone."):
        return "backbone_decoder_target"
    if param_name.startswith("model.clutter_head."):
        return "clutter_head"
    if param_name.startswith("model."):
        return "other_model"
    if param_name.startswith("cal_loss."):
        return "loss_buffers_or_params"
    return "other"


def diff_state(a: dict, b: dict) -> dict[str, float]:
    groups: dict[str, float] = {
        "backbone_decoder_target": 0.0,
        "clutter_head": 0.0,
        "other_model": 0.0,
        "loss_buffers_or_params": 0.0,
        "z_final_related_params": 0.0,
    }
    for key, tensor_a in a.items():
        if key not in b or not torch.is_tensor(tensor_a) or not torch.is_tensor(b[key]):
            continue
        if tensor_a.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            continue
        d = float((tensor_a.detach().float() - b[key].detach().float()).abs().max().item())
        g = group_name(key)
        groups[g] = max(groups.get(g, 0.0), d)
    return groups


def optimizer_rows(net: Net, optimizer: torch.optim.Optimizer) -> list[dict]:
    id_to_group = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        for param in group["params"]:
            id_to_group[id(param)] = {
                "group_idx": group_idx,
                "lr": group.get("lr", ""),
                "weight_decay": group.get("weight_decay", 0.0),
            }
    rows = []
    for name, param in net.named_parameters():
        opt_info = id_to_group.get(id(param), {})
        rows.append(
            {
                "name": name,
                "group": group_name(name),
                "requires_grad": bool(param.requires_grad),
                "numel": int(param.numel()),
                "in_optimizer": bool(opt_info),
                "optimizer_group": opt_info.get("group_idx", ""),
                "lr": opt_info.get("lr", ""),
                "weight_decay": opt_info.get("weight_decay", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.detach() - b.detach()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def grad_norms(net: Net) -> dict[str, float]:
    sums: dict[str, float] = {}
    for name, param in net.named_parameters():
        if param.grad is None:
            continue
        g = group_name(name)
        sums[g] = sums.get(g, 0.0) + float(param.grad.detach().pow(2).sum().item())
    return {key: value ** 0.5 for key, value in sums.items()}


def state_group_diff(net_a: Net, net_b: Net) -> dict[str, float]:
    return diff_state(net_a.state_dict(), net_b.state_dict())


def make_net(model_name: str, cfg: dict, device: torch.device) -> Net:
    net = Net(model_name=model_name, mode="train", loss_cfg=cfg).to(device)
    net.train()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-parity probe for OHCM-light vs OHCMMSHNetFull-F0.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohcm_warm_epoch", type=int, default=60)
    parser.add_argument("--ohcm_tau", type=float, default=0.5)
    parser.add_argument("--ohcm_dilate_radius", type=int, default=5)
    parser.add_argument("--ohcm_topk", type=int, default=3)
    parser.add_argument("--ohcm_gamma_max", type=float, default=0.3)
    parser.add_argument("--ohcm_gamma_ramp_epochs", type=int, default=60)
    parser.add_argument("--ohcm_margin_m", type=float, default=0.1)
    parser.add_argument("--ohcm_margin_delta", type=float, default=0.5)
    parser.add_argument("--ohcm_gt_area_median", type=float, default=20.0)
    parser.add_argument("--ohcm_mining_mode", default="cc_area_lc_ms")
    parser.add_argument("--lambda_clu", type=float, default=0.2)
    parser.add_argument("--lambda_sup", type=float, default=0.5)
    parser.add_argument("--lambda_margin", type=float, default=0.1)
    parser.add_argument("--lambda_proto", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = vars(args).copy()
    cfg["ohcm_force_no_proto"] = True
    cfg["ohcm_use_proto"] = False
    cfg["optimizer_settings"] = {"lr": 0.05}
    cfg["scheduler_settings"] = {"epochs": 400, "min_lr": 1e-5}

    seed_pytorch(args.seed)
    light = make_net("OHCMMSHNet", cfg, device)
    init_light = {
        "epoch": 0,
        "model_name": "OHCMMSHNet",
        "state_dict": light.state_dict(),
        "seed": args.seed,
        "config": cfg,
    }
    torch.save(init_light, output_dir / "init_light.pth")

    seed_pytorch(args.seed)
    full_direct = make_net("OHCMMSHNetFull", cfg, device)
    direct_init_diff = diff_state(light.state_dict(), full_direct.state_dict())

    full = make_net("OHCMMSHNetFull", cfg, device)
    full.load_state_dict(init_light["state_dict"])
    loaded_init_diff = diff_state(light.state_dict(), full.state_dict())
    torch.save(
        {
            "epoch": 0,
            "model_name": "OHCMMSHNetFull",
            "state_dict": full.state_dict(),
            "seed": args.seed,
            "config": cfg,
            "source": "shared weights loaded from init_light.pth",
        },
        output_dir / "init_light_as_full_f0.pth.tar",
    )

    optimizer_settings = {"lr": 0.05}
    scheduler_settings = {"epochs": 400, "min_lr": 1e-5}
    light_opt, _ = get_optimizer(light, "Adagrad", "CosineAnnealingLR", optimizer_settings.copy(), scheduler_settings.copy())
    full_opt, _ = get_optimizer(full, "Adagrad", "CosineAnnealingLR", optimizer_settings.copy(), scheduler_settings.copy())
    write_csv(
        output_dir / "optimizer_light.csv",
        optimizer_rows(light, light_opt),
        ["name", "group", "requires_grad", "numel", "in_optimizer", "optimizer_group", "lr", "weight_decay"],
    )
    write_csv(
        output_dir / "optimizer_full_f0.csv",
        optimizer_rows(full, full_opt),
        ["name", "group", "requires_grad", "numel", "in_optimizer", "optimizer_group", "lr", "weight_decay"],
    )

    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)
    train_set = TrainSetLoader(args.dataset_dir, args.dataset_name, args.patchSize, img_norm_cfg=img_norm_cfg)
    loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batchSize, shuffle=False)
    img, gt_mask = next(iter(loader))
    img, gt_mask = Variable(img).to(device), Variable(gt_mask).to(device)

    parity_rows = []
    for epoch in (1, 100):
        for net in (light, full):
            net.zero_grad(set_to_none=True)
        light_out = light(img, epoch=epoch)
        full_out = full(img, epoch=epoch)
        row = {"epoch": epoch}
        for key in ("target_logit", "clutter_logit", "final_logit"):
            d = tensor_diff(light_out[key], full_out[key])
            row[f"{key}_max_abs"] = d["max_abs"]
            row[f"{key}_mean_abs"] = d["mean_abs"]
        p_diff = tensor_diff(torch.sigmoid(light_out["final_logit"]), torch.sigmoid(full_out["final_logit"]))
        row["p_final_max_abs"] = p_diff["max_abs"]
        row["p_final_mean_abs"] = p_diff["mean_abs"]

        light_loss = light.loss_with_image(light_out, gt_mask, img, epoch=epoch)
        full_loss = full.loss_with_image(full_out, gt_mask, img, epoch=epoch)
        for key in ("total", "sls", "clu", "sup", "margin", "proto", "hard_pixels", "hard_components", "hard_score_mean", "gamma"):
            lv = float(light_loss[key].detach().cpu())
            fv = float(full_loss[key].detach().cpu())
            row[f"{key}_light"] = lv
            row[f"{key}_full"] = fv
            row[f"{key}_abs_diff"] = abs(lv - fv)

        light_loss["total"].backward()
        full_loss["total"].backward()
        light_grads = grad_norms(light)
        full_grads = grad_norms(full)
        for group in sorted(set(light_grads) | set(full_grads)):
            lv = light_grads.get(group, 0.0)
            fv = full_grads.get(group, 0.0)
            row[f"grad_{group}_light"] = lv
            row[f"grad_{group}_full"] = fv
            row[f"grad_{group}_abs_diff"] = abs(lv - fv)

        before_step = state_group_diff(light, full)
        light_opt.step()
        full_opt.step()
        after_step = state_group_diff(light, full)
        for group, value in before_step.items():
            row[f"param_{group}_before_step_max_abs"] = value
        for group, value in after_step.items():
            row[f"param_{group}_after_step_max_abs"] = value
        parity_rows.append(row)

        # Restore from init for next epoch probe.
        light.load_state_dict(init_light["state_dict"])
        full.load_state_dict(init_light["state_dict"])
        light_opt, _ = get_optimizer(light, "Adagrad", "CosineAnnealingLR", optimizer_settings.copy(), scheduler_settings.copy())
        full_opt, _ = get_optimizer(full, "Adagrad", "CosineAnnealingLR", optimizer_settings.copy(), scheduler_settings.copy())

    fieldnames = sorted({key for row in parity_rows for key in row})
    write_csv(output_dir / "first_batch_parity.csv", parity_rows, fieldnames)

    pass_first_batch = all(
        row.get("target_logit_max_abs", 1.0) <= 1e-7
        and row.get("clutter_logit_max_abs", 1.0) <= 1e-7
        and row.get("final_logit_max_abs", 1.0) <= 1e-7
        and row.get("total_abs_diff", 1.0) <= 1e-7
        and max(value for key, value in row.items() if key.endswith("_after_step_max_abs")) <= 1e-7
        for row in parity_rows
    )
    summary = {
        "decision": "PASS_FIRST_BATCH_PARITY" if pass_first_batch else "FAIL_FIRST_BATCH_PARITY",
        "seed": args.seed,
        "direct_init_diff": direct_init_diff,
        "loaded_init_diff": loaded_init_diff,
        "first_batch_epochs": [row["epoch"] for row in parity_rows],
        "pass_first_batch": pass_first_batch,
        "outputs": {
            "init_light": str(output_dir / "init_light.pth"),
            "init_light_as_full_f0": str(output_dir / "init_light_as_full_f0.pth.tar"),
            "optimizer_light": str(output_dir / "optimizer_light.csv"),
            "optimizer_full_f0": str(output_dir / "optimizer_full_f0.csv"),
            "first_batch_parity": str(output_dir / "first_batch_parity.csv"),
        },
    }
    (output_dir / "train_parity_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
