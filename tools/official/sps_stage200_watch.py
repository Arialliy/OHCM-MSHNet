#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


RUNS = (
    "SPS-rerank-alpha100-gain1005-hybridfb00001",
    "TwoViewOHEM-rerank-gain1005-hybridfb00001",
    "ConfidenceOnly-rerank-alpha100-gain1005-hybridfb00001",
    "GlobalConsistency-rerankctrl-gain1005-hybridfb00001",
)


def parse_run_names(value: str | None) -> tuple[str, ...]:
    if not value:
        return RUNS
    names = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            _label, run_name = item.split("=", 1)
        else:
            run_name = item
        names.append(run_name.strip())
    if not names:
        raise ValueError("--runs did not contain any valid run names.")
    return tuple(names)


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")
    print(f"[{stamp}] {message}", flush=True)


def checkpoints_ready(run_root: Path, runs: tuple[str, ...], dataset: str, seed: int, epoch: int, min_age_sec: int) -> tuple[bool, list[Path]]:
    missing = []
    now = time.time()
    for run in runs:
        path = (
            run_root
            / run
            / f"seed_{seed}"
            / "checkpoints"
            / dataset
            / f"MSHNetSPSOHEM_{epoch}.pth.tar"
        )
        if not path.exists():
            missing.append(path)
            continue
        age = now - path.stat().st_mtime
        if age < min_age_sec:
            missing.append(path)
    return len(missing) == 0, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch SPS Stage-200 checkpoints and evaluate ready epochs.")
    parser.add_argument("--run_root", default="results/sps_ohem/20260625_sps_rerank_stage200")
    parser.add_argument("--dataset_name", default="NUDT-SIRST")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", default="50,100,150,200")
    parser.add_argument("--interval_sec", type=int, default=300)
    parser.add_argument("--min_checkpoint_age_sec", type=int, default=60)
    parser.add_argument("--output_dir", default="results/sps_ohem/20260625_sps_rerank_stage200/gate")
    parser.add_argument("--log_path", default="results/sps_ohem/20260625_sps_rerank_stage200/stage200_watch.log")
    parser.add_argument("--runs", default=None, help="Comma-separated run_name or label=run_name entries.")
    parser.add_argument("--primary_label", default="SPS")
    parser.add_argument("--control_labels", default="TwoViewOHEM,ConfidenceOnly,GlobalConsistency")
    parser.add_argument("--baseline_full", default=None)
    parser.add_argument("--baseline_hcval", default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_path)
    epochs = [int(item) for item in args.epochs.split(",") if item.strip()]
    runs = parse_run_names(args.runs)
    done_dir = output_dir / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)

    while True:
        all_done = True
        for epoch in epochs:
            sentinel = done_dir / f"epoch_{epoch}.done"
            if sentinel.exists():
                continue
            all_done = False
            ready, missing = checkpoints_ready(
                run_root,
                runs,
                args.dataset_name,
                args.seed,
                epoch,
                args.min_checkpoint_age_sec,
            )
            if not ready:
                log(log_path, f"epoch {epoch}: pending checkpoints={len(missing)}")
                continue

            log(log_path, f"epoch {epoch}: all checkpoints ready; launching Full/HC-Val evaluation")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            cmd = [
                sys.executable,
                "tools/official/sps_stage200_eval.py",
                "--run_root",
                str(run_root),
                "--seed",
                str(args.seed),
                "--dataset_name",
                args.dataset_name,
                "--epochs",
                ",".join(str(item) for item in epochs if item <= epoch),
                "--output_dir",
                str(output_dir),
            ]
            if args.runs:
                cmd.extend(["--runs", args.runs])
            if args.baseline_full:
                cmd.extend(["--baseline_full", args.baseline_full])
            if args.baseline_hcval:
                cmd.extend(["--baseline_hcval", args.baseline_hcval])
            cmd.extend(["--primary_label", args.primary_label, "--control_labels", args.control_labels])
            subprocess.run(cmd, check=True, env=env)
            sentinel.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
            log(log_path, f"epoch {epoch}: evaluation finished")

        if all_done:
            log(log_path, "all requested epochs evaluated; watcher exiting")
            return
        if args.once:
            log(log_path, "once mode complete; watcher exiting")
            return
        time.sleep(max(10, args.interval_sec))


if __name__ == "__main__":
    main()
