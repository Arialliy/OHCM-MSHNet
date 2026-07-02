import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.official.check_twa_gate_e2_fullsafe_single_control import (  # noqa: E402
    build_snapshot_record,
    choose_best_eligible,
    metrics_from_summary,
)


def summary(miou, precision, pd, fa_ppm):
    return {"metrics": {"mIoU": miou, "Precision": precision, "Pd": pd, "FA_ppm": fa_ppm}}


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_record(epoch, full, hc, ohem_full, ohem_hc):
    return build_snapshot_record(
        epoch=epoch,
        full_summary=full,
        hcval_summary=hc,
        ohem_full=ohem_full,
        ohem_hcval=ohem_hc,
        min_full_delta_miou=0.0,
        min_full_delta_precision=0.0,
        min_full_delta_pd=0.0,
        max_full_delta_fa_ppm=0.0,
        min_hcval_delta_miou=0.005,
        min_hcval_fa_reduction_ppm=10.0,
        min_hcval_delta_precision=0.0,
        min_hcval_delta_pd=0.0,
    )


def test_ep250_hc_strong_but_full_regression_is_ineligible():
    ohem_full = metrics_from_summary(summary(0.834393, 0.900000, 0.980000, 63.449))
    ohem_hc = metrics_from_summary(summary(0.604790, 0.700000, 0.970000, 250.0))

    rec = make_record(
        250,
        summary(0.831139, 0.899813, 0.980000, 63.334),
        summary(0.710648, 0.821241, 0.970000, 74.524),
        ohem_full,
        ohem_hc,
    )

    assert rec.hcval_positive is True
    assert rec.full_safe is False
    assert rec.eligible_single is False
    assert "full_miou_regression" in rec.ineligible_reasons
    assert "full_precision_regression" in rec.ineligible_reasons


def test_full_safe_hc_positive_ep300_becomes_eligible():
    ohem_full = metrics_from_summary(summary(0.834393, 0.900000, 0.980000, 63.449))
    ohem_hc = metrics_from_summary(summary(0.604790, 0.700000, 0.970000, 250.0))

    rec = make_record(
        300,
        summary(0.835000, 0.901000, 0.980000, 62.0),
        summary(0.660000, 0.760000, 0.970000, 200.0),
        ohem_full,
        ohem_hc,
    )

    assert rec.full_safe is True
    assert rec.hcval_positive is True
    assert rec.eligible_single is True


def test_choose_best_eligible_ignores_full_unsafe_ep250():
    ohem_full = metrics_from_summary(summary(0.834393, 0.900000, 0.980000, 63.449))
    ohem_hc = metrics_from_summary(summary(0.604790, 0.700000, 0.970000, 250.0))
    ep250 = make_record(
        250,
        summary(0.831139, 0.899813, 0.980000, 63.334),
        summary(0.710648, 0.821241, 0.970000, 74.524),
        ohem_full,
        ohem_hc,
    )
    ep300 = make_record(
        300,
        summary(0.835000, 0.901000, 0.980000, 62.0),
        summary(0.660000, 0.760000, 0.970000, 200.0),
        ohem_full,
        ohem_hc,
    )

    best = choose_best_eligible([ep250, ep300])
    assert best is not None
    assert best.epoch == 300


def test_pd_drop_blocks_full_safety_even_when_miou_passes():
    ohem_full = metrics_from_summary(summary(0.834393, 0.900000, 0.980000, 63.449))
    ohem_hc = metrics_from_summary(summary(0.604790, 0.700000, 0.970000, 250.0))
    rec = make_record(
        300,
        summary(0.836000, 0.901000, 0.979900, 62.0),
        summary(0.660000, 0.760000, 0.970000, 200.0),
        ohem_full,
        ohem_hc,
    )

    assert rec.full_safe is False
    assert rec.eligible_single is False
    assert "full_pd_regression" in rec.ineligible_reasons


def test_fa_increase_blocks_full_safety():
    ohem_full = metrics_from_summary(summary(0.834393, 0.900000, 0.980000, 63.449))
    ohem_hc = metrics_from_summary(summary(0.604790, 0.700000, 0.970000, 250.0))
    rec = make_record(
        350,
        summary(0.836000, 0.901000, 0.980000, 64.0),
        summary(0.660000, 0.760000, 0.970000, 200.0),
        ohem_full,
        ohem_hc,
    )

    assert rec.full_safe is False
    assert rec.eligible_single is False
    assert "full_fa_regression" in rec.ineligible_reasons


def _common_cli_files(tmp_path: Path):
    ohem_full = tmp_path / "ohem_full.json"
    ohem_hc = tmp_path / "ohem_hc.json"
    twa_full = tmp_path / "twa_full.json"
    twa_hc = tmp_path / "twa_hc.json"
    gate_e = tmp_path / "gate_e.json"
    ep250_gate = tmp_path / "ep250_gate.json"
    write_json(ohem_full, summary(0.834393, 0.900000, 0.980000, 63.449))
    write_json(ohem_hc, summary(0.604790, 0.700000, 0.970000, 250.0))
    write_json(twa_full, summary(0.838893, 0.903450, 0.981060, 61.082))
    write_json(twa_hc, summary(0.633891, 0.735450, 0.970000, 191.508))
    write_json(gate_e, {"gate_pass": False, "decision": "STOP_TWA_NO_BN_AT_GATE_E"})
    write_json(ep250_gate, {"gate_pass": False, "decision": "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"})
    return ohem_full, ohem_hc, twa_full, twa_hc, gate_e, ep250_gate


def _run_cli(tmp_path: Path, ep300_hc_miou: float, ep300_hc_precision: float):
    tool = Path("tools/official/check_twa_gate_e2_fullsafe_single_control.py")
    ohem_full, ohem_hc, twa_full, twa_hc, gate_e, ep250_gate = _common_cli_files(tmp_path)
    ep250_full = tmp_path / "ep250_full.json"
    ep250_hc = tmp_path / "ep250_hc.json"
    ep300_full = tmp_path / "ep300_full.json"
    ep300_hc = tmp_path / "ep300_hc.json"
    ep350_full = tmp_path / "ep350_full.json"
    ep350_hc = tmp_path / "ep350_hc.json"
    output = tmp_path / "out.json"

    write_json(ep250_full, summary(0.831139, 0.899813, 0.980000, 63.334))
    write_json(ep250_hc, summary(0.710648, 0.821241, 0.970000, 74.524))
    write_json(ep300_full, summary(0.835000, 0.901000, 0.980000, 62.0))
    write_json(ep300_hc, summary(ep300_hc_miou, ep300_hc_precision, 0.970000, 180.0))
    write_json(ep350_full, summary(0.836000, 0.902000, 0.980000, 62.0))
    write_json(ep350_hc, summary(0.625000, 0.725000, 0.970000, 215.0))

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--gate_e_summary",
            str(gate_e),
            "--ep250_gate_a_summary",
            str(ep250_gate),
            "--ohem_full",
            str(ohem_full),
            "--ohem_hcval",
            str(ohem_hc),
            "--twa_full",
            str(twa_full),
            "--twa_hcval",
            str(twa_hc),
            "--snapshot",
            f"250:{ep250_full}:{ep250_hc}",
            "--snapshot",
            f"300:{ep300_full}:{ep300_hc}",
            "--snapshot",
            f"350:{ep350_full}:{ep350_hc}",
            "--output",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_cli_reopens_twa_when_no_fullsafe_single_beats_it(tmp_path):
    result = _run_cli(tmp_path, ep300_hc_miou=0.620000, ep300_hc_precision=0.720000)

    assert result["gate_pass"] is True
    assert result["selected_candidate"] == "TWA-4-noBN"
    assert result["next_allowed_gate"] == "Gate-TWA-F-seed43-44-Full-HCVal"


def test_cli_switches_to_fullsafe_single_when_it_beats_twa(tmp_path):
    result = _run_cli(tmp_path, ep300_hc_miou=0.670000, ep300_hc_precision=0.780000)

    assert result["gate_pass"] is True
    assert result["selected_candidate"] == "LateSnapshot-ep300"
    assert result["next_allowed_gate"] == "Gate-LS-B-ep300-seed43-44-Full-HCVal"
    assert result["promotion_allowed"] is True
    assert result["selection_status"] == "SELECTED_WITH_PRACTICAL_MARGIN"


def test_cli_stops_posthoc_selection_when_single_advantage_is_tie(tmp_path):
    result = _run_cli(tmp_path, ep300_hc_miou=0.636000, ep300_hc_precision=0.780000)

    assert result["gate_pass"] is True
    assert result["selected_candidate"] is None
    assert result["promotion_allowed"] is False
    assert result["selection_status"] == "TIE_OR_NUMERICAL_NOISE_NO_SWITCH"
    assert result["next_allowed_gate"] == "STOP_POSTHOC_CHECKPOINT_SELECTION"
