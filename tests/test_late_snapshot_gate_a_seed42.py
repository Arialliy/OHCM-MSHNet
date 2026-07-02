import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/official/check_late_snapshot_gate_a_seed42.py")


def _write_summary(path: Path, *, miou: float, fa: float, precision: float, pd: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metrics": {"mIoU": miou, "FA_ppm": fa, "Precision": precision, "Pd": pd}}),
        encoding="utf-8",
    )
    return path


def _metric_payload(*, miou: float, fa: float, precision: float, pd: float) -> dict:
    return {"mIoU": miou, "FA_ppm": fa, "Precision": precision, "Pd": pd}


def _write_gate_e(
    path: Path,
    *,
    failed_conditions=None,
    best_name: str = "ep250",
    ep250_miou: float = 0.710648,
    ep300_miou: float = 0.650000,
) -> Path:
    if failed_conditions is None:
        failed_conditions = ["twa4_not_worse_than_best_single_late_hcval"]

    all_condition_names = [
        "gate_d_passed_and_allows_gate_e",
        "twa4_full_nonregression_vs_ohem",
        "twa4_hcval_improvement_vs_ohem",
        "twa4_not_worse_than_best_single_late_hcval",
        "twa4_retains_tce_hard_split_gain",
        "twa2_twa3_twa4_trend_reasonable",
    ]
    conditions = {name: name not in failed_conditions for name in all_condition_names}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "gate": "Gate-TWA-E",
                "gate_pass": False,
                "decision": "STOP_TWA_NO_BN_AT_GATE_E",
                "next_allowed_gate": "STOP_TWA_NO_BN_AT_GATE_E",
                "conditions": conditions,
                "best_single_late_checkpoint": {
                    "name": best_name,
                    "metrics": _metric_payload(miou=ep250_miou, fa=210.524, precision=0.781241, pd=0.833333),
                    "all_single_late_hcval": {
                        "ep250": _metric_payload(miou=ep250_miou, fa=210.524, precision=0.781241, pd=0.833333),
                        "ep300": _metric_payload(miou=ep300_miou, fa=290.0, precision=0.720000, pd=0.833333),
                        "ep350": _metric_payload(miou=0.640000, fa=310.0, precision=0.700000, pd=0.833333),
                        "ep400": _metric_payload(miou=0.604791, fa=386.000, precision=0.660000, pd=0.833333),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_files(tmp_path: Path) -> dict[str, Path]:
    return {
        "gate_e": _write_gate_e(tmp_path / "gate_e.json"),
        "ohem_full": _write_summary(tmp_path / "ohem_full.json", miou=0.833393, fa=63.449, precision=0.906277, pd=0.979894),
        "ohem_hcval": _write_summary(tmp_path / "ohem_hcval.json", miou=0.604791, fa=386.000, precision=0.660000, pd=0.833333),
        "snapshot_full": _write_summary(tmp_path / "snapshot_full.json", miou=0.835000, fa=62.000, precision=0.910000, pd=0.979894),
        "snapshot_hcval": _write_summary(tmp_path / "snapshot_hcval.json", miou=0.710648, fa=210.524, precision=0.781241, pd=0.833333),
        "twa4_hcval": _write_summary(tmp_path / "twa4_hcval.json", miou=0.633891, fa=327.508, precision=0.695450, pd=0.833333),
    }


def _cmd(files: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--gate_e_summary",
        str(files["gate_e"]),
        "--ohem_full",
        str(files["ohem_full"]),
        "--ohem_hcval",
        str(files["ohem_hcval"]),
        "--snapshot_full",
        str(files["snapshot_full"]),
        "--snapshot_hcval",
        str(files["snapshot_hcval"]),
        "--twa4_hcval",
        str(files["twa4_hcval"]),
        "--snapshot_name",
        "ep250",
        "--output",
        str(output),
    ]


def test_late_snapshot_gate_a_passes_for_ep250_rescue(tmp_path):
    files = _base_files(tmp_path)
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["gate_pass"] is True
    assert summary["next_allowed_gate"] == "Gate-LS-B-seed43-seed44-paired-Full-HCVal"


def test_late_snapshot_gate_a_fails_if_gate_e_failed_for_extra_reason(tmp_path):
    files = _base_files(tmp_path)
    files["gate_e"] = _write_gate_e(
        tmp_path / "gate_e_extra_fail.json",
        failed_conditions=[
            "twa4_not_worse_than_best_single_late_hcval",
            "twa4_retains_tce_hard_split_gain",
        ],
    )
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["gate_e_failed_only_because_best_single_won"] is False


def test_late_snapshot_gate_a_fails_if_best_single_is_not_ep250(tmp_path):
    files = _base_files(tmp_path)
    files["gate_e"] = _write_gate_e(tmp_path / "gate_e_ep300.json", best_name="ep300")
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["best_single_is_frozen_snapshot"] is False


def test_late_snapshot_gate_a_fails_if_ep250_is_not_unique_enough(tmp_path):
    files = _base_files(tmp_path)
    files["gate_e"] = _write_gate_e(tmp_path / "gate_e_not_unique.json", ep250_miou=0.710648, ep300_miou=0.708500)
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["best_single_is_frozen_snapshot"] is False


def test_late_snapshot_gate_a_fails_if_snapshot_full_regresses(tmp_path):
    files = _base_files(tmp_path)
    files["snapshot_full"] = _write_summary(tmp_path / "snapshot_full_bad.json", miou=0.820000, fa=70.000, precision=0.890000, pd=0.979894)
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["snapshot_full_nonregression_vs_ohem"] is False


def test_late_snapshot_gate_a_fails_if_snapshot_hcval_does_not_beat_ohem(tmp_path):
    files = _base_files(tmp_path)
    files["snapshot_hcval"] = _write_summary(tmp_path / "snapshot_hcval_bad_ohem.json", miou=0.606000, fa=380.000, precision=0.661000, pd=0.833333)
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["snapshot_hcval_improvement_vs_ohem"] is False


def test_late_snapshot_gate_a_fails_if_snapshot_hcval_does_not_beat_twa4(tmp_path):
    files = _base_files(tmp_path)
    files["snapshot_hcval"] = _write_summary(tmp_path / "snapshot_hcval_bad_twa4.json", miou=0.635000, fa=326.000, precision=0.696000, pd=0.833333)
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["snapshot_hcval_improvement_vs_twa4"] is False
