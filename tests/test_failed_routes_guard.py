import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_apf_train_script_is_guarded():
    text = (PROJECT_ROOT / "tools" / "official" / "train_apf_seed.sh").read_text()
    assert "ALLOW_FAILED_APF_TRAINING" in text
    assert "[BLOCKED]" in text
    assert "APF-OHEM is stopped" in text


def test_failed_routes_include_apf():
    text = (PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py").read_text()
    assert '"apf"' in text
    assert "APF Gate-A" in text


def test_failed_routes_include_eacf_v1():
    text = (PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py").read_text()
    assert '"eacf"' in text
    assert "EACF-v1 stopped" in text


def test_failed_routes_include_sacf_v1():
    text = (PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py").read_text()
    assert '"sacf"' in text
    assert "SACF-v1 stopped" in text


def test_check_failed_routes_blocks_apf():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py"),
            "--model_name",
            "apf",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "[BLOCKED]" in result.stdout


def test_check_failed_routes_blocks_eacf_v1():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py"),
            "--model_name",
            "EACFMSHNet",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "[BLOCKED]" in result.stdout
    assert "identity collapse" in result.stdout


def test_check_failed_routes_blocks_sacf_v1():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "official" / "check_failed_routes_blocked.py"),
            "--model_name",
            "SACFMSHNet",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "[BLOCKED]" in result.stdout
    assert "identity collapse" in result.stdout
