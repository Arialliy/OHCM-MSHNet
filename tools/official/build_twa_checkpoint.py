#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch


def torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_state(path: str | Path, key: str = "state_dict"):
    checkpoint = torch_load(path)
    if isinstance(checkpoint, dict) and key in checkpoint:
        return checkpoint[key], checkpoint
    return checkpoint, checkpoint


def _check_tensor_compatibility(key: str, values: list):
    tensor_flags = [torch.is_tensor(value) for value in values]
    if len(set(tensor_flags)) > 1:
        raise ValueError(f"Tensor/non-tensor mismatch at {key}")
    if not tensor_flags[0]:
        return
    shapes = [tuple(value.shape) for value in values]
    if len(set(shapes)) > 1:
        raise ValueError(f"Shape mismatch at {key}: {shapes}")


def average_states(states: list[dict]) -> OrderedDict:
    if not states:
        raise ValueError("No checkpoints were provided.")
    keys = list(states[0].keys())
    for idx, state in enumerate(states[1:], start=1):
        if list(state.keys()) != keys:
            missing = sorted(set(keys) - set(state.keys()))
            extra = sorted(set(state.keys()) - set(keys))
            raise ValueError(f"Checkpoint keys do not match at index {idx}: missing={missing[:8]} extra={extra[:8]}")

    averaged = OrderedDict()
    for key in keys:
        values = [state[key] for state in states]
        _check_tensor_compatibility(key, values)
        first = values[0]
        if torch.is_tensor(first) and first.is_floating_point():
            avg = sum(value.detach().cpu().float() for value in values) / float(len(values))
            averaged[key] = avg.to(dtype=first.dtype)
        elif torch.is_tensor(first):
            averaged[key] = values[-1].detach().cpu().clone()
        else:
            averaged[key] = values[-1]
    return averaged


def checkpoint_epoch(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("epoch")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TWA-OHEM checkpoint by uniform weight averaging.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_key", default="state_dict")
    args = parser.parse_args()

    states = []
    meta = []
    for path in args.checkpoints:
        state, checkpoint = load_state(path, args.model_key)
        states.append(state)
        meta.append({"path": str(Path(path).resolve()), "epoch": checkpoint_epoch(checkpoint)})

    averaged = average_states(states)
    output = {
        "state_dict": averaged,
        "twa_meta": {
            "checkpoints": meta,
            "num_checkpoints": len(args.checkpoints),
            "method": "uniform_weight_average",
            "model_key": args.model_key,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(json.dumps(output["twa_meta"], indent=2), flush=True)


if __name__ == "__main__":
    main()
