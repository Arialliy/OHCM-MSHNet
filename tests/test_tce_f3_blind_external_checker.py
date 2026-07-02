import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_TOOL = PROJECT_ROOT / "tools" / "official" / "check_tce_f3_preflight.py"
REPORT_TOOL = PROJECT_ROOT / "tools" / "official" / "check_tce_f3_blind_external_report.py"


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_text(path: Path, text: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metrics(miou, fa_ppm, precision=0.9, pd=0.98, fp_components=10):
    return {
        "metrics_at_threshold": {
            "mIoU": miou,
            "FA_ppm": fa_ppm,
            "Precision": precision,
            "Pd": pd,
            "FP_components": fp_components,
        }
    }


def valid_plan():
    return {
        "decision": "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE",
        "method": {
            "name": "TCE-4-OHEM",
            "base": "MSHNetOHEM",
            "checkpoints": [250, 300, 350, 400],
            "aggregation": "existing_official_tce_aggregation",
            "threshold": 0.5,
            "training": "no_new_training",
            "inference_forward_count": 4,
        },
    }


def valid_manifest(tmp_path, threshold=0.5, seeds=None, epochs=None):
    seeds = seeds or [42, 43, 44]
    epochs = epochs or [250, 300, 350, 400]
    splits = ["blind", "external"]
    manifest = {
        "gate": "Gate-TCE-F3-blind-external-once",
        "candidate": "TCE-4-OHEM",
        "baseline": "MSHNetOHEM-400",
        "status": "LOCKED_BEFORE_BLIND_EXTERNAL",
        "threshold": threshold,
        "seeds": seeds,
        "tce_epochs": epochs,
        "splits": splits,
        "split_datasets": {"blind": "BLIND", "external": "EXTERNAL"},
        "summary_paths": {},
        "checkpoint_paths": {},
    }
    for split in splits:
        manifest["summary_paths"][split] = {}
        for seed in seeds:
            manifest["summary_paths"][split][str(seed)] = {
                "ohem": str(tmp_path / "summaries" / split / str(seed) / "ohem.json"),
                "tce4": str(tmp_path / "summaries" / split / str(seed) / "tce4.json"),
            }
    for seed in seeds:
        manifest["checkpoint_paths"][str(seed)] = {}
        for epoch in epochs:
            ckpt = tmp_path / "checkpoints" / str(seed) / f"e{epoch}.pth.tar"
            write_text(ckpt)
            manifest["checkpoint_paths"][str(seed)][str(epoch)] = str(ckpt)
    return manifest


def run_preflight(tmp_path, manifest, f0_gate=True, final_exists=False):
    f0 = tmp_path / "f0.json"
    f1 = tmp_path / "f1.json"
    f2 = tmp_path / "f2.json"
    plan = tmp_path / "plan.json"
    manifest_path = tmp_path / "manifest.json"
    lock = tmp_path / "lock.json"
    final = tmp_path / "final.json"
    out = tmp_path / "preflight.json"
    write_json(f0, {"gate_pass": f0_gate})
    write_json(f1, {"gate_pass": True})
    write_json(f2, {"gate_pass": True})
    write_json(plan, valid_plan())
    write_json(manifest_path, manifest)
    if final_exists:
        write_json(final, {"already": "exists"})
    return subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_TOOL),
            "--f0_summary",
            str(f0),
            "--f1_summary",
            str(f1),
            "--f2_summary",
            str(f2),
            "--frozen_method_plan",
            str(plan),
            "--f3_manifest",
            str(manifest_path),
            "--once_lock",
            str(lock),
            "--final_report",
            str(final),
            "--output",
            str(out),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def build_report_fixture(tmp_path, ohem_metric, tce_metric):
    manifest = valid_manifest(tmp_path)
    for split in manifest["splits"]:
        for seed in [42, 43, 44]:
            pair = manifest["summary_paths"][split][str(seed)]
            write_json(Path(pair["ohem"]), ohem_metric)
            write_json(Path(pair["tce4"]), tce_metric)
    manifest_path = tmp_path / "manifest.json"
    lock_path = tmp_path / "lock.json"
    out_path = tmp_path / "report.json"
    write_json(manifest_path, manifest)
    write_json(lock_path, {"status": "STARTED"})
    return manifest_path, lock_path, out_path


def run_report(manifest_path, lock_path, out_path):
    return subprocess.run(
        [
            sys.executable,
            str(REPORT_TOOL),
            "--manifest",
            str(manifest_path),
            "--once_lock",
            str(lock_path),
            "--output",
            str(out_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def test_f3_preflight_passes_and_creates_once_lock(tmp_path):
    manifest = valid_manifest(tmp_path)
    proc = run_preflight(tmp_path, manifest)

    assert proc.returncode == 0
    out = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    lock = json.loads((tmp_path / "lock.json").read_text(encoding="utf-8"))
    assert out["gate_pass"] is True
    assert out["status"] == "NEW_ONCE_LOCK_CREATED"
    assert lock["status"] == "STARTED"


def test_f3_preflight_fails_if_previous_gate_not_passed(tmp_path):
    manifest = valid_manifest(tmp_path)
    proc = run_preflight(tmp_path, manifest, f0_gate=False)

    assert proc.returncode != 0
    assert "Gate-TCE-F0 is not PASS" in proc.stderr


def test_f3_preflight_fails_if_threshold_changes(tmp_path):
    manifest = valid_manifest(tmp_path, threshold=0.6)
    proc = run_preflight(tmp_path, manifest)

    assert proc.returncode != 0
    assert "F3 threshold must be 0.5" in proc.stderr


def test_f3_preflight_fails_if_final_report_exists(tmp_path):
    manifest = valid_manifest(tmp_path)
    proc = run_preflight(tmp_path, manifest, final_exists=True)

    assert proc.returncode != 0
    assert "F3 final report already exists" in proc.stderr


def test_f3_report_strong_pass(tmp_path):
    manifest_path, lock_path, out_path = build_report_fixture(
        tmp_path,
        metrics(0.80, 100.0, 0.90, 0.98),
        metrics(0.81, 90.0, 0.91, 0.98, fp_components=8),
    )

    proc = run_report(manifest_path, lock_path, out_path)

    assert proc.returncode == 0
    out = json.loads(out_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert out["gate_pass"] is True
    assert out["verdict"] == "F3_PASS_STRONG"
    assert lock["status"] == "COMPLETED"


def test_f3_report_mixed_reportable_pass(tmp_path):
    manifest_path, lock_path, out_path = build_report_fixture(
        tmp_path,
        metrics(0.80, 100.0, 0.90, 0.98),
        metrics(0.8001, 100.0, 0.89, 0.98),
    )

    proc = run_report(manifest_path, lock_path, out_path)

    assert proc.returncode == 0
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["gate_pass"] is True
    assert out["verdict"] == "F3_PASS_MIXED_REPORTABLE"


def test_f3_report_fail_stops_without_rescue(tmp_path):
    manifest_path, lock_path, out_path = build_report_fixture(
        tmp_path,
        metrics(0.80, 100.0, 0.90, 0.98),
        metrics(0.801, 101.0, 0.91, 0.98),
    )

    proc = run_report(manifest_path, lock_path, out_path)

    assert proc.returncode != 0
    out = json.loads(out_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert out["gate_pass"] is False
    assert out["verdict"] == "F3_FAIL_NO_REDESIGN"
    assert lock["status"] == "COMPLETED"


def test_f3_report_completed_lock_refuses_rerun(tmp_path):
    manifest_path, lock_path, out_path = build_report_fixture(
        tmp_path,
        metrics(0.80, 100.0),
        metrics(0.81, 90.0),
    )
    write_json(lock_path, {"status": "COMPLETED"})

    proc = run_report(manifest_path, lock_path, out_path)

    assert proc.returncode != 0
    assert "already completed" in proc.stderr
