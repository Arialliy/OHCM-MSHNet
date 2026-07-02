import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.twa_gate_utils import get_metric


SCRIPT = Path("tools/official/check_twa_gate_e_mechanism.py")


def _write_summary(path: Path, *, miou: float, fa: float, precision: float, pd: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metrics": {"mIoU": miou, "FA_ppm": fa, "Precision": precision, "Pd": pd}}),
        encoding="utf-8",
    )
    return path


def _write_gate_d(path: Path, *, gate_pass: bool = True, next_allowed_gate: str = "Gate-TWA-E") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "gate": "Gate-TWA-D",
                "candidate": "twa_without_bn",
                "gate_pass": gate_pass,
                "next_allowed_gate": next_allowed_gate,
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_files(tmp_path: Path) -> dict[str, Path]:
    return {
        "gate_d": _write_gate_d(tmp_path / "gate_d.json"),
        "ohem_full": _write_summary(tmp_path / "ohem_full.json", miou=0.83, fa=61.0, precision=0.90, pd=0.98),
        "ohem_hc": _write_summary(tmp_path / "ohem_hc.json", miou=0.60, fa=386.0, precision=0.66, pd=0.833333),
        "twa4_full": _write_summary(tmp_path / "twa4_full.json", miou=0.835, fa=59.0, precision=0.905, pd=0.98),
        "twa4_hc": _write_summary(tmp_path / "twa4_hc.json", miou=0.6291, fa=327.508, precision=0.69545, pd=0.833333),
        "tce4_hc": _write_summary(tmp_path / "tce4_hc.json", miou=0.66, fa=286.0, precision=0.72, pd=0.833333),
        "single250": _write_summary(tmp_path / "single250.json", miou=0.61, fa=360.0, precision=0.67, pd=0.833333),
        "single300": _write_summary(tmp_path / "single300.json", miou=0.62, fa=340.0, precision=0.68, pd=0.833333),
        "single350": _write_summary(tmp_path / "single350.json", miou=0.625, fa=330.0, precision=0.69, pd=0.833333),
        "single400": _write_summary(tmp_path / "single400.json", miou=0.60, fa=386.0, precision=0.66, pd=0.833333),
        "twa2_hc": _write_summary(tmp_path / "twa2_hc.json", miou=0.621, fa=340.0, precision=0.685, pd=0.833333),
        "twa3_hc": _write_summary(tmp_path / "twa3_hc.json", miou=0.626, fa=333.0, precision=0.690, pd=0.833333),
        "twa2_full": _write_summary(tmp_path / "twa2_full.json", miou=0.832, fa=60.0, precision=0.902, pd=0.98),
        "twa3_full": _write_summary(tmp_path / "twa3_full.json", miou=0.834, fa=59.5, precision=0.904, pd=0.98),
    }


def _cmd(files: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--gate_d_summary",
        str(files["gate_d"]),
        "--ohem_full",
        str(files["ohem_full"]),
        "--ohem_hcval",
        str(files["ohem_hc"]),
        "--twa4_full",
        str(files["twa4_full"]),
        "--twa4_hcval",
        str(files["twa4_hc"]),
        "--tce4_hcval",
        str(files["tce4_hc"]),
        "--single_late",
        f"ep250={files['single250']}",
        "--single_late",
        f"ep300={files['single300']}",
        "--single_late",
        f"ep350={files['single350']}",
        "--single_late",
        f"ep400={files['single400']}",
        "--twa_variant_hcval",
        f"TWA-2={files['twa2_hc']}",
        "--twa_variant_hcval",
        f"TWA-3={files['twa3_hc']}",
        "--twa_variant_hcval",
        f"TWA-4={files['twa4_hc']}",
        "--twa_variant_full",
        f"TWA-2={files['twa2_full']}",
        "--twa_variant_full",
        f"TWA-3={files['twa3_full']}",
        "--twa_variant_full",
        f"TWA-4={files['twa4_full']}",
        "--output",
        str(output),
    ]


def test_gate_e_passes_for_consistent_mechanism(tmp_path):
    files = _base_files(tmp_path)
    output = tmp_path / "gate_e.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["gate_pass"] is True
    assert summary["next_allowed_gate"] == "Gate-TWA-F-seed43-seed44-paired-Full-HCVal"


def test_gate_e_fails_if_gate_d_does_not_allow_gate_e(tmp_path):
    files = _base_files(tmp_path)
    files["gate_d"] = _write_gate_d(tmp_path / "bad_gate_d.json", gate_pass=True, next_allowed_gate="STOP")
    output = tmp_path / "gate_e.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["gate_d_passed_and_allows_gate_e"] is False


def test_gate_e_fails_when_best_single_beats_twa4(tmp_path):
    files = _base_files(tmp_path)
    files["single350"] = _write_summary(tmp_path / "single350.json", miou=0.64, fa=320.0, precision=0.70, pd=0.833333)
    output = tmp_path / "gate_e.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["twa4_not_worse_than_best_single_late_hcval"] is False


def test_gate_e_fails_when_tce_retention_is_too_low(tmp_path):
    files = _base_files(tmp_path)
    files["tce4_hc"] = _write_summary(tmp_path / "tce4_hc.json", miou=0.80, fa=80.0, precision=0.90, pd=0.833333)
    output = tmp_path / "gate_e.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["twa4_retains_tce_hard_split_gain"] is False


def test_gate_e_fails_when_twa4_full_regresses_pd(tmp_path):
    files = _base_files(tmp_path)
    files["twa4_full"] = _write_summary(tmp_path / "twa4_full.json", miou=0.835, fa=59.0, precision=0.905, pd=0.979)
    output = tmp_path / "gate_e.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["twa4_full_nonregression_vs_ohem"] is False


def test_gate_e_fails_when_twa_variant_trend_is_missing(tmp_path):
    files = _base_files(tmp_path)
    output = tmp_path / "gate_e.json"
    cmd = _cmd(files, output)
    filtered = []
    skip_next = False
    for token in cmd:
        if skip_next:
            skip_next = False
            continue
        if token == "--twa_variant_hcval":
            skip_next = True
            continue
        filtered.append(token)
    filtered.extend(["--twa_variant_hcval", f"TWA-4={files['twa4_hc']}"])

    result = subprocess.run(filtered, text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["twa2_twa3_twa4_trend_reasonable"] is False


def test_gate_e_metric_reader_prefers_fa_ppm_over_fa():
    summary = {"metrics_at_threshold": {"FA": 0.0002, "FA_ppm": 200.0}}

    assert get_metric(summary, "FA_ppm") == 200.0
