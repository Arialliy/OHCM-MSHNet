#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import TestSetLoader
from net import Net
from probability import foreground_probability
from utils import get_img_norm_cfg


FORBIDDEN_SPLITS = {"test", "full", "hcval", "hctest", "hc-test", "blind", "external"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build train-only dense TCE soft labels from OHEM checkpoints.")
    parser.add_argument("--dataset", "--dataset_name", dest="dataset_name", default="NUDT-SIRST")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_dir", default="/home/AAAI/OHCM-MSHNet/datasets")
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--student_checkpoint", default=None)
    parser.add_argument("--mshnet_head", default="final")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--mshnet_warm_epoch", type=int, default=5)
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--ohem_ratio", type=float, default=0.01)
    parser.add_argument("--save_member_probs", action="store_true", default=True)
    return parser.parse_args()


def size_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def checkpoint_epoch(path: str, fallback: int) -> str:
    match = re.search(r"_(\d+)\.pth\.tar$", Path(path).name)
    return match.group(1) if match else str(fallback)


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def assert_teacher_student_not_identical(teacher_ckpts: list[str], student_ckpt: str) -> dict:
    student_hash = sha256_file(student_ckpt)
    teacher_hashes = [sha256_file(path) for path in teacher_ckpts]
    unique_teacher_hashes = sorted(set(teacher_hashes))
    if len(unique_teacher_hashes) < 2:
        raise RuntimeError(
            "TCE teacher checkpoints are not diverse. Expected at least two distinct teacher checkpoint hashes."
        )
    if all(h == student_hash for h in teacher_hashes):
        raise RuntimeError(
            "All TCE teacher checkpoints are identical to the student checkpoint. "
            "Soft-label distillation would be a no-op."
        )
    return {
        "student_hash": student_hash,
        "teacher_hashes": teacher_hashes,
        "unique_teacher_hash_count": len(unique_teacher_hashes),
    }


def load_checkpoint(net: Net, checkpoint_path: str, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()
    return checkpoint


def forward_prob(net: Net, img: torch.Tensor, h: int, w: int) -> np.ndarray:
    export = net.export_logits_features(img)
    logit = export["logit"][:, :, :h, :w]
    return foreground_probability(logit)[0, 0].detach().cpu().numpy().astype(np.float32)


def train_ids(dataset_dir: str, dataset_name: str) -> list[str]:
    path = Path(dataset_dir) / dataset_name / "img_idx" / f"train_{dataset_name}.txt"
    if not path.exists():
        fallback = Path(dataset_dir) / dataset_name / "img_idx" / "train.txt"
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(str(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    if args.split.lower() != "train" or any(token in args.split.lower() for token in FORBIDDEN_SPLITS):
        raise ValueError("TCE soft labels may only be generated for train split.")
    if len(args.checkpoints) < 2:
        raise ValueError("--checkpoints must contain at least two OHEM checkpoints.")
    student_checkpoint = args.student_checkpoint or args.checkpoints[-1]
    hash_meta = assert_teacher_student_not_identical(args.checkpoints, student_checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = train_ids(args.dataset_dir, args.dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset_name = args.train_dataset_name or args.dataset_name
    img_norm_cfg = get_img_norm_cfg(train_dataset_name, args.dataset_dir)
    dataset = TestSetLoader(args.dataset_dir, train_dataset_name, args.dataset_name, img_norm_cfg)
    dataset.test_list = ids
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

    nets = []
    checkpoint_meta = []
    net_by_path = {}
    for idx, checkpoint_path in enumerate(args.checkpoints):
        resolved = str(Path(checkpoint_path).resolve())
        net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
        checkpoint = load_checkpoint(net, checkpoint_path, device)
        epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else checkpoint_epoch(checkpoint_path, idx)
        checkpoint_meta.append(
            {
                "path": resolved,
                "epoch": int(epoch),
                "sha256": hash_meta["teacher_hashes"][idx],
            }
        )
        nets.append((str(epoch), net))
        net_by_path[resolved] = net

    student_resolved = str(Path(student_checkpoint).resolve())
    student_net = net_by_path.get(student_resolved)
    if student_net is None:
        student_net = Net(model_name=args.model_name, mode="test", loss_cfg=vars(args)).to(device)
        load_checkpoint(student_net, student_checkpoint, device)

    written = []
    with torch.no_grad():
        for idx, (img, mask, size, image_name) in enumerate(loader):
            img = img.to(device)
            h, w = size_to_int(size[0]), size_to_int(size[1])
            name = image_name[0] if isinstance(image_name, (list, tuple)) else str(image_name)
            gt = (mask[0, 0, :h, :w].numpy() > 0).astype(np.uint8)
            member_probs = {}
            member_stats = {}
            probs = []
            for epoch, net in nets:
                prob = forward_prob(net, img, h, w)
                member_probs[f"p_{epoch}"] = prob
                member_stats[f"p_{epoch}_mean"] = float(prob.mean())
                member_stats[f"p_{epoch}_std"] = float(prob.std())
                probs.append(prob)
            teacher_prob = np.stack(probs, axis=0).mean(axis=0).astype(np.float32)
            student_prob = forward_prob(student_net, img, h, w)
            absdiff = np.abs(teacher_prob - student_prob)
            payload = {
                "teacher_prob": teacher_prob,
                "student_prob": student_prob,
                "gt_mask": gt,
                "p_tce": teacher_prob,
                "p_ohem_400": student_prob,
                "image_id": np.asarray(name),
                "teacher_ckpts": np.asarray([str(Path(path).resolve()) for path in args.checkpoints]),
                "student_ckpt": np.asarray(student_resolved),
                "teacher_epoch": np.asarray([int(item["epoch"]) for item in checkpoint_meta], dtype=np.int32),
                "probability_fn": np.asarray("foreground_probability"),
                "head": np.asarray(args.mshnet_head),
                "teacher_prob_mean": np.asarray(float(teacher_prob.mean()), dtype=np.float32),
                "student_prob_mean": np.asarray(float(student_prob.mean()), dtype=np.float32),
                "teacher_student_absdiff_mean": np.asarray(float(absdiff.mean()), dtype=np.float32),
            }
            if args.save_member_probs:
                payload.update(member_probs)
            np.savez_compressed(output_dir / f"{name}.npz", **payload)
            written.append(name)
            if (idx + 1) % 100 == 0:
                print("Built TCE soft labels [%d/%d]" % (idx + 1, len(loader)), flush=True)

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "num_images": len(ids),
        "num_npz": len(written),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "student_checkpoint": student_resolved,
        "student_sha256": hash_meta["student_hash"],
        "teacher_unique_hash_count": hash_meta["unique_teacher_hash_count"],
        "probability_fn": "foreground_probability",
        "head": args.mshnet_head,
        "checkpoints": checkpoint_meta,
        "train_only": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
