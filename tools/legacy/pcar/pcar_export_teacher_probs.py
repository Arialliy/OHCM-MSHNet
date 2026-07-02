#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net import Net  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg, seed_pytorch  # noqa: E402

IMAGE_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def find_file(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def load_image(path: Path, img_norm_cfg) -> tuple[torch.Tensor, tuple[int, int]]:
    raw = np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    h, w = raw.shape
    img = Normalized(raw, img_norm_cfg)
    img = PadImg(img)
    tensor = torch.from_numpy(np.ascontiguousarray(img[None, None])).float()
    return tensor, (h, w)


def read_names(dataset_dir: Path, dataset_name: str, split: str, image_list: str | None, max_images: int) -> list[str]:
    if image_list:
        names = [line.strip() for line in Path(image_list).read_text().splitlines() if line.strip()]
    else:
        list_path = dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt"
        names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if max_images > 0:
        names = names[:max_images]
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen OHEM teacher probabilities for PCAR diagnostics.")
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--image_list", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="MSHNetOHEM")
    parser.add_argument("--mshnet_in_channels", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_dir) / args.dataset_name
    names = read_names(dataset_root, args.dataset_name, args.split, args.image_list, args.max_images)
    img_norm_cfg = get_img_norm_cfg(args.dataset_name, args.dataset_dir)

    output_dir = Path(args.output_dir)
    prob_dir = output_dir / "probs"
    prob_dir.mkdir(parents=True, exist_ok=True)

    net = Net(args.model_name, mode="test", loss_cfg={"mshnet_in_channels": args.mshnet_in_channels}).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()

    with torch.no_grad():
        for idx, name in enumerate(names):
            img, (h, w) = load_image(find_file(dataset_root / "images", name), img_norm_cfg)
            logit = net.export_logits_features(img.to(device))["logit"]
            prob = torch.sigmoid(logit)[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32)
            np.save(prob_dir / f"{name}.npy", prob)
            if (idx + 1) % 100 == 0:
                print(f"Exported teacher probs [{idx + 1}/{len(names)}]", flush=True)

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "image_list": args.image_list,
        "images": len(names),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "prob_dir": str(prob_dir),
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
