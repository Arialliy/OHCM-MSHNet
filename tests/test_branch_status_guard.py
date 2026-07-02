import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.branch_status import assert_branch_allowed


def test_stopped_branch_blocked():
    with pytest.raises(RuntimeError):
        assert_branch_allowed("BCVMSHNet", allow_stopped_branch=False)


def test_stopped_branch_allowed_for_repro():
    assert_branch_allowed("BCVMSHNet", allow_stopped_branch=True)


def test_active_anchor_allowed():
    assert_branch_allowed("MSHNetOHEM", allow_stopped_branch=False)
