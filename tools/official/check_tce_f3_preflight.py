#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_TCE_EPOCHS = [250, 300, 350, 400]


def resolve_existing_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
        text = str(path)
        if text.startswith("/home/ly/AAAI/"):
            candidates.append(Path("/home/AAAI") / text[len("/home/ly/AAAI/"):])
        if text.startswith("/home/AAAI/"):
            candidates.append(Path("/home/ly/AAAI") / text[len("/home/AAAI/"):])
    else:
        candidates.extend([PROJECT_ROOT / path, Path.cwd() / path, path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def load_json(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_existing_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {resolved}")
    return data


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sha256_file(path: str | Path) -> str:
    resolved = resolve_existing_path(path)
    h = hashlib.sha256()
    with resolved.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_gate_pass(summary: Dict[str, Any], name: str) -> None:
    if summary.get("gate_pass") is not True:
        raise SystemExit(f"{name} is not PASS: gate_pass={summary.get('gate_pass')}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def validate_frozen_plan(plan: Dict[str, Any]) -> None:
    method = plan.get("method", {})
    require(plan.get("decision") == "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE", "TCE-4 freeze decision missing.")
    require(method.get("name") == "TCE-4-OHEM", f"Unexpected frozen method: {method.get('name')}")
    require(method.get("base") == "MSHNetOHEM", f"Unexpected frozen base: {method.get('base')}")
    require(method.get("checkpoints") == EXPECTED_TCE_EPOCHS, "Frozen checkpoints must be [250,300,350,400].")
    require(float(method.get("threshold", -1.0)) == 0.5, "Frozen threshold must be 0.5.")
    require(method.get("training") == "no_new_training", "Frozen method must require no new training.")
    require(int(method.get("inference_forward_count", -1)) == 4, "Frozen method must use four forwards.")


def validate_manifest(manifest: Dict[str, Any]) -> None:
    require(manifest.get("gate") == "Gate-TCE-F3-blind-external-once", f"Unexpected gate: {manifest.get('gate')}")
    require(manifest.get("candidate") == "TCE-4-OHEM", f"Unexpected candidate: {manifest.get('candidate')}")
    require(manifest.get("baseline") == "MSHNetOHEM-400", f"Unexpected baseline: {manifest.get('baseline')}")
    require(float(manifest.get("threshold", -1.0)) == 0.5, f"F3 threshold must be 0.5, got {manifest.get('threshold')}")
    require(list(manifest.get("seeds", [])) == EXPECTED_SEEDS, f"F3 seeds must be [42,43,44], got {manifest.get('seeds')}")
    require(
        list(manifest.get("tce_epochs", [])) == EXPECTED_TCE_EPOCHS,
        f"F3 TCE epochs must be [250,300,350,400], got {manifest.get('tce_epochs')}",
    )
    require(manifest.get("status") == "LOCKED_BEFORE_BLIND_EXTERNAL", "F3 manifest must be locked before external evaluation.")

    splits = manifest.get("splits", [])
    require(bool(splits) and all(isinstance(item, str) and item for item in splits), "F3 manifest must define non-empty splits.")
    summary_paths = manifest.get("summary_paths")
    require(isinstance(summary_paths, dict), "F3 manifest missing summary_paths.")
    split_datasets = manifest.get("split_datasets")
    require(isinstance(split_datasets, dict), "F3 manifest missing split_datasets.")

    for split in splits:
        require(split in split_datasets, f"Missing dataset mapping for split={split}")
        require(split in summary_paths, f"Missing summary_paths for split={split}")
        for seed in ["42", "43", "44"]:
            pair = summary_paths[split].get(seed)
            require(isinstance(pair, dict), f"Missing summary pair for split={split}, seed={seed}")
            require("ohem" in pair and "tce4" in pair, f"Missing ohem/tce4 summary paths for split={split}, seed={seed}")

    checkpoint_paths = manifest.get("checkpoint_paths")
    require(isinstance(checkpoint_paths, dict), "F3 manifest missing checkpoint_paths.")
    for seed in ["42", "43", "44"]:
        seed_paths = checkpoint_paths.get(seed)
        require(isinstance(seed_paths, dict), f"Missing checkpoint_paths for seed={seed}")
        for epoch in ["250", "300", "350", "400"]:
            ckpt = seed_paths.get(epoch)
            require(isinstance(ckpt, str) and ckpt, f"Missing checkpoint path for seed={seed}, epoch={epoch}")
            resolved = resolve_existing_path(ckpt)
            require(resolved.exists(), f"Checkpoint does not exist: seed={seed}, epoch={epoch}, path={ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-TCE-F3 blind/external once preflight checker.")
    parser.add_argument("--f0_summary", required=True)
    parser.add_argument("--f1_summary", required=True)
    parser.add_argument("--f2_summary", required=True)
    parser.add_argument("--frozen_method_plan", required=True)
    parser.add_argument("--f3_manifest", required=True)
    parser.add_argument("--once_lock", required=True)
    parser.add_argument("--final_report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    final_report = Path(args.final_report)
    if final_report.exists():
        raise SystemExit(f"F3 final report already exists. Do not rerun blind/external: {final_report}")

    f0 = load_json(args.f0_summary)
    f1 = load_json(args.f1_summary)
    f2 = load_json(args.f2_summary)
    plan = load_json(args.frozen_method_plan)
    manifest = load_json(args.f3_manifest)

    require_gate_pass(f0, "Gate-TCE-F0")
    require_gate_pass(f1, "Gate-TCE-F1")
    require_gate_pass(f2, "Gate-TCE-F2")
    validate_frozen_plan(plan)
    validate_manifest(manifest)

    manifest_sha = sha256_file(args.f3_manifest)
    lock_path = Path(args.once_lock)
    if lock_path.exists():
        lock = load_json(lock_path)
        if lock.get("status") == "COMPLETED":
            raise SystemExit("F3 once lock is already completed. Do not rerun.")
        if lock.get("manifest_sha256") != manifest_sha:
            raise SystemExit("Existing F3 lock manifest hash differs. Do not continue after manifest change.")
        status = "PREEXISTING_LOCK_RESUME_ALLOWED_FOR_MISSING_OUTPUTS_ONLY"
    else:
        lock = {
            "gate": "Gate-TCE-F3-blind-external-once",
            "status": "STARTED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": manifest.get("candidate"),
            "baseline": manifest.get("baseline"),
            "threshold": manifest.get("threshold"),
            "seeds": manifest.get("seeds"),
            "tce_epochs": manifest.get("tce_epochs"),
            "splits": manifest.get("splits"),
            "split_datasets": manifest.get("split_datasets"),
            "frozen_method_plan_sha256": sha256_file(args.frozen_method_plan),
            "f0_sha256": sha256_file(args.f0_summary),
            "f1_sha256": sha256_file(args.f1_summary),
            "f2_sha256": sha256_file(args.f2_summary),
            "manifest_sha256": manifest_sha,
        }
        save_json(lock, lock_path)
        status = "NEW_ONCE_LOCK_CREATED"

    out = {
        "gate": "Gate-TCE-F3-preflight",
        "gate_pass": True,
        "status": status,
        "next_allowed_action": "RUN_BLIND_EXTERNAL_ONCE",
        "once_lock": str(lock_path),
        "final_report": str(final_report),
        "candidate": manifest.get("candidate"),
        "baseline": manifest.get("baseline"),
        "threshold": manifest.get("threshold"),
        "seeds": manifest.get("seeds"),
        "tce_epochs": manifest.get("tce_epochs"),
        "splits": manifest.get("splits"),
        "forbidden_after_start": [
            "threshold_search",
            "seed_search",
            "checkpoint_search",
            "method_change",
            "rerun_after_result",
        ],
    }
    save_json(out, args.output)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
