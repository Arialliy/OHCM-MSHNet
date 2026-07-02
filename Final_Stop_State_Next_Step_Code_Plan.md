# OHCM-MSHNet 最终停止态后的下一步：只做代码收口、防误跑、归档，不再开发新方法

> 当前目标：把已经触发的 `F3_FAIL_NO_REDESIGN` 固化为仓库级停止状态。  
> 这一步不是继续优化模型，不是继续评估，也不是换一个新结构。  
> 代码修改只允许用于：**状态一致性检查、阻止误跑、归档失败证据、生成论文/项目复盘材料**。

---

## 0. 当前输入状态

你已经完成以下收口：

```text
新增：
  docs/internal/tce_final/gate_tce_f3_fail_summary.json

更新：
  docs/internal/tce_final/gate_tce_f3_once_lock.json
    status: STOPPED_BY_F3_PD_REGRESSION

  README.md
    顶部状态：TCE-4 已在 F3 external Pd regression 停止

  STOPPED_BRANCHES_SUMMARY.md
    记录 TCE-4 F3 停止证据和禁止项

验证：
  JSON 可解析
  git diff --check PASS

未做：
  没有继续 evaluation
  没有新训练
  没有改 split
  没有 threshold / seed / checkpoint search
  没有修改 loss.py / net.py / train.py / dataset.py / model/
```

正式状态应固定为：

```text
STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
STOP_TCSR_AT_BANK_AUDIT
STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD
STOP_TWA_NO_BN_AT_GATE_E
STOP_LATE_SNAPSHOT_EP250_AT_GATE_A
```

当前不再存在 active AAAI main-method branch。

---

## 1. 下一步总原则

### 1.1 不再继续做的方法开发

不要再做：

```text
新训练
新模型结构
新 loss
新 verifier
新 suppression head
TCSR Stage 2
TWA / ep300 / checkpoint rescue
seed43/44 rescue
threshold rescue
BN tuning
external split 修改
补空 mask 后继续 F3
删除 SIRST3 后继续 F3
```

原因：F3 是 frozen method 的 blind / external once gate。它一旦因为 external Pd regression 失败，后续所有 rescue 都会改变 once-lock 语义。

### 1.2 现在唯一允许的代码工作

只允许做：

```text
[1] final stop-state checker
[2] final stop-state pytest
[3] F3 run script / preflight 防误跑 guard
[4] README / STOPPED summary 状态一致性修正
[5] final closure table / archive manifest
```

这一步的目标是让仓库进入：

```text
READ_ONLY_FAILURE_ARCHIVE_STATE
```

---

## 2. 新增最终收口 gate

新增一个只读 gate：

```text
Gate-FINAL-STOP-CONSISTENCY
```

它不跑模型，不跑评估，只检查：

```text
1. gate_tce_f3_fail_summary.json 存在且 decision == F3_FAIL_NO_REDESIGN
2. gate_tce_f3_once_lock.json 存在且 status == STOPPED_BY_F3_PD_REGRESSION
3. 至少一个已完成 external split 出现 min_delta_Pd < 0
4. F3 final report 不存在，或者被显式标记为 invalid / unused
5. README 顶部含 STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
6. STOPPED_BRANCHES_SUMMARY.md 含 F3_FAIL_NO_REDESIGN 和 Pd regression
7. 不存在 active next_allowed_gate 指向 TCE / TCSR / TWA / ep300 后续开发
```

通过后输出：

```text
docs/internal/final_stop_state_summary.json
```

该文件是最终仓库收口证据。

---

## 3. 文件修改清单

只新增 / 修改以下文件：

```text
新增：
  tools/official/check_final_stop_state.py
  scripts/official/run_final_stop_state_check.sh
  tests/test_final_stop_state.py
  docs/internal/final_stop_state_plan.json

修改：
  scripts/official/run_tce_f3_blind_external_once.sh
  tools/official/check_tce_f3_preflight.py      # 可选，但推荐
  README.md
  STOPPED_BRANCHES_SUMMARY.md
```

明确不改：

```text
loss.py
net.py
train.py
dataset.py
model/
metrics.py
probability.py
```

---

## 4. 新增 `docs/internal/final_stop_state_plan.json`

新增文件：

```text
docs/internal/final_stop_state_plan.json
```

建议内容：

```json
{
  "plan": "Final stop-state consistency and archive plan",
  "decision": "READ_ONLY_FAILURE_ARCHIVE_STATE",
  "current_final_stop": "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
  "stopped_routes": {
    "tce4": {
      "decision": "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
      "failed_gate": "Gate-TCE-F3-blind-external-once",
      "reason": "external Pd regression",
      "lock_status_required": "STOPPED_BY_F3_PD_REGRESSION"
    },
    "tcsr_v1": {
      "decision": "STOP_TCSR_AT_BANK_AUDIT",
      "failed_gate": "Gate-TCSR-A",
      "reason": "train split had insufficient target-safe sparse hard-clutter negatives"
    },
    "posthoc_checkpoint_selection": {
      "decision": "STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD",
      "reason": "ep300 advantage over TWA-4 was a numerical tie and not a structural method"
    },
    "twa_no_bn": {
      "decision": "STOP_TWA_NO_BN_AT_GATE_E",
      "reason": "failed mechanism comparison against single late checkpoint control"
    },
    "late_snapshot_ep250": {
      "decision": "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A",
      "reason": "HC-Val strong but Full split unsafe"
    }
  },
  "allowed_next_work": [
    "final stop-state checker",
    "final stop-state pytest",
    "README / STOPPED summary consistency",
    "failure-analysis table",
    "archive packaging"
  ],
  "forbidden_next_work": [
    "new training",
    "new evaluation",
    "threshold search",
    "seed search",
    "checkpoint search",
    "split redefinition",
    "SIRST3 mask imputation",
    "BN tuning",
    "new verifier",
    "new suppression head",
    "loss.py modification",
    "net.py modification",
    "train.py modification",
    "dataset.py modification"
  ]
}
```

---

## 5. 新增 `tools/official/check_final_stop_state.py`

新增文件：

```text
tools/official/check_final_stop_state.py
```

完整代码：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class FinalStopStateError(RuntimeError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FinalStopStateError(f"Missing required JSON file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise FinalStopStateError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FinalStopStateError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def read_text(path: Path) -> str:
    if not path.exists():
        raise FinalStopStateError(f"Missing required text file: {path}")
    return path.read_text(encoding="utf-8")


def first_n_lines(text: str, n: int = 160) -> str:
    return "\n".join(text.splitlines()[:n])


def normalize_token(x: str) -> str:
    return " ".join(str(x).lower().replace("_", " ").replace("-", " ").split())


def contains_token(text: str, token: str) -> bool:
    # Accept both exact and normalized matches. This is intentionally tolerant
    # because README/summary wording may use hyphenated or underscored labels.
    return token in text or normalize_token(token) in normalize_token(text)


def collect_numeric_values(obj: Any, key_names: Iterable[str]) -> List[float]:
    keys = set(key_names)
    values: List[float] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) in keys:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
            values.extend(collect_numeric_values(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(collect_numeric_values(item, keys))
    return values


def ensure_no_active_gate_in_top_readme(readme_text: str, errors: List[str]) -> None:
    top = first_n_lines(readme_text, 180)
    suspicious_phrases = [
        "Current active candidate",
        "Next allowed gate: Gate-TWA",
        "Next allowed gate: Gate-TCSR",
        "Next allowed gate: Gate-TCE",
        "PROCEED_TCE4_TO_F3",
        "Gate-TCE-F3-blind-external-once",
        "Gate-LS-B-ep300",
    ]
    allowed_context = [
        "No active AAAI main-method branch remains",
        "no active AAAI main-method branch remains",
        "No active next_allowed_gate",
    ]
    if any(contains_token(top, ctx) for ctx in allowed_context):
        return
    for phrase in suspicious_phrases:
        if contains_token(top, phrase):
            errors.append(
                f"README top block still looks active because it contains {phrase!r}. "
                "Move historical gate text below an archive section and make the top state final-stopped."
            )


def check_required_text_tokens(label: str, text: str, tokens: Iterable[str], errors: List[str]) -> None:
    for token in tokens:
        if not contains_token(text, token):
            errors.append(f"{label} does not contain required token: {token}")


def check_final_stop_state(
    root: Path,
    f3_fail_summary_path: Optional[Path] = None,
    f3_once_lock_path: Optional[Path] = None,
    readme_path: Optional[Path] = None,
    stopped_summary_path: Optional[Path] = None,
    f3_final_report_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = root.resolve()
    tce_dir = root / "docs" / "internal" / "tce_final"

    f3_fail_summary_path = f3_fail_summary_path or (tce_dir / "gate_tce_f3_fail_summary.json")
    f3_once_lock_path = f3_once_lock_path or (tce_dir / "gate_tce_f3_once_lock.json")
    readme_path = readme_path or (root / "README.md")
    stopped_summary_path = stopped_summary_path or (root / "STOPPED_BRANCHES_SUMMARY.md")
    f3_final_report_path = f3_final_report_path or (tce_dir / "gate_tce_f3_blind_external_report.json")
    output_path = output_path or (root / "docs" / "internal" / "final_stop_state_summary.json")

    errors: List[str] = []
    warnings: List[str] = []

    fail_summary = load_json(f3_fail_summary_path)
    once_lock = load_json(f3_once_lock_path)
    readme_text = read_text(readme_path)
    stopped_text = read_text(stopped_summary_path)

    decision = fail_summary.get("decision")
    if decision != "F3_FAIL_NO_REDESIGN":
        errors.append(
            "gate_tce_f3_fail_summary.json must have "
            "decision == 'F3_FAIL_NO_REDESIGN', got "
            f"{decision!r}"
        )

    lock_status = once_lock.get("status")
    if lock_status != "STOPPED_BY_F3_PD_REGRESSION":
        errors.append(
            "gate_tce_f3_once_lock.json must have "
            "status == 'STOPPED_BY_F3_PD_REGRESSION', got "
            f"{lock_status!r}"
        )

    pd_values = collect_numeric_values(fail_summary.get("failed_splits", fail_summary), [
        "min_delta_Pd",
        "delta_Pd",
    ])
    negative_pd_values = [x for x in pd_values if x < 0]
    if not negative_pd_values:
        errors.append(
            "F3 fail summary must contain at least one negative min_delta_Pd / delta_Pd "
            "inside failed_splits."
        )

    # SIRST3 integrity failure is not the main fail reason, but if it is recorded,
    # require it to be marked as not completed rather than silently redefined.
    not_completed = fail_summary.get("not_completed_splits", {})
    if isinstance(not_completed, Mapping) and "external_sirst3" in not_completed:
        sirst3 = not_completed["external_sirst3"]
        reason_text = json.dumps(sirst3, ensure_ascii=False).lower()
        if "manifest" not in reason_text and "integrity" not in reason_text and "missing" not in reason_text:
            warnings.append(
                "external_sirst3 is listed as not completed, but the reason does not clearly "
                "mention manifest/integrity/missing data."
            )

    if f3_final_report_path.exists():
        errors.append(
            "F3 final report exists even though F3 stopped before final report generation: "
            f"{f3_final_report_path}. Do not use it as official final report."
        )

    check_required_text_tokens(
        "README.md",
        first_n_lines(readme_text, 200),
        [
            "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
            "STOP_TCSR_AT_BANK_AUDIT",
            "No active AAAI main-method branch remains",
        ],
        errors,
    )
    ensure_no_active_gate_in_top_readme(readme_text, errors)

    check_required_text_tokens(
        "STOPPED_BRANCHES_SUMMARY.md",
        stopped_text,
        [
            "F3_FAIL_NO_REDESIGN",
            "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
            "Pd regression",
            "STOP_TCSR_AT_BANK_AUDIT",
            "STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD",
        ],
        errors,
    )

    forbidden_actions = fail_summary.get("forbidden_next_actions", [])
    forbidden_text = json.dumps(forbidden_actions, ensure_ascii=False)
    for token in [
        "threshold search",
        "seed search",
        "checkpoint search",
        "split redefinition",
        "new model training",
    ]:
        if not contains_token(forbidden_text, token):
            warnings.append(f"F3 fail summary forbidden_next_actions does not explicitly list: {token}")

    result: Dict[str, Any] = {
        "gate": "Gate-FINAL-STOP-CONSISTENCY",
        "gate_pass": len(errors) == 0,
        "decision": "READ_ONLY_FAILURE_ARCHIVE_STATE" if len(errors) == 0 else "FINAL_STOP_STATE_INCONSISTENT",
        "root": str(root),
        "checked_files": {
            "f3_fail_summary": str(f3_fail_summary_path),
            "f3_once_lock": str(f3_once_lock_path),
            "readme": str(readme_path),
            "stopped_summary": str(stopped_summary_path),
            "f3_final_report": str(f3_final_report_path),
        },
        "required_status": {
            "f3_fail_decision": "F3_FAIL_NO_REDESIGN",
            "f3_once_lock_status": "STOPPED_BY_F3_PD_REGRESSION",
            "top_readme_state": "STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION",
        },
        "observed": {
            "f3_fail_decision": decision,
            "f3_once_lock_status": lock_status,
            "negative_pd_values": negative_pd_values,
            "f3_final_report_exists": f3_final_report_path.exists(),
        },
        "errors": errors,
        "warnings": warnings,
        "forbidden_next_actions": [
            "new evaluation",
            "new training",
            "threshold search",
            "seed search",
            "checkpoint search",
            "split redefinition",
            "SIRST3 mask imputation",
            "TCSR Stage 2",
            "new verifier",
            "new suppression head",
            "loss.py modification",
            "net.py modification",
            "train.py modification",
            "dataset.py modification",
        ],
        "allowed_next_actions": [
            "archive final stopped state",
            "generate failure-analysis table",
            "prepare internal report",
            "freeze repository",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if errors:
        raise FinalStopStateError(
            "Gate-FINAL-STOP-CONSISTENCY failed:\n" + "\n".join(f"- {e}" for e in errors)
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check final stopped state after TCE-4 F3 external Pd regression."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--f3_fail_summary", type=Path, default=None)
    parser.add_argument("--f3_once_lock", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--stopped_summary", type=Path, default=None)
    parser.add_argument("--f3_final_report", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: docs/internal/final_stop_state_summary.json",
    )
    args = parser.parse_args()

    try:
        result = check_final_stop_state(
            root=args.root,
            f3_fail_summary_path=args.f3_fail_summary,
            f3_once_lock_path=args.f3_once_lock,
            readme_path=args.readme,
            stopped_summary_path=args.stopped_summary,
            f3_final_report_path=args.f3_final_report,
            output_path=args.output,
        )
    except FinalStopStateError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

---

## 6. 新增 `scripts/official/run_final_stop_state_check.sh`

新增文件：

```text
scripts/official/run_final_stop_state_check.sh
```

代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}

python "${ROOT}/tools/official/check_final_stop_state.py" \
  --root "${ROOT}" \
  --output "${ROOT}/docs/internal/final_stop_state_summary.json"

echo "Gate-FINAL-STOP-CONSISTENCY PASS: repository is in read-only failure archive state."
```

赋权：

```bash
chmod +x scripts/official/run_final_stop_state_check.sh
```

---

## 7. 修改 `scripts/official/run_tce_f3_blind_external_once.sh`

目的：防止之后误运行 F3 once 脚本。

在脚本最顶部 `set -euo pipefail` 后加入：

```bash
ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}
F3_FAIL_SUMMARY="${ROOT}/docs/internal/tce_final/gate_tce_f3_fail_summary.json"
F3_ONCE_LOCK="${ROOT}/docs/internal/tce_final/gate_tce_f3_once_lock.json"

if [[ -f "${F3_FAIL_SUMMARY}" ]]; then
  python "${ROOT}/tools/official/check_final_stop_state.py" \
    --root "${ROOT}" \
    --output "${ROOT}/docs/internal/final_stop_state_summary.json"
  echo "F3 is already stopped by external Pd regression. Refusing to rerun blind/external once." >&2
  exit 2
fi

if [[ -f "${F3_ONCE_LOCK}" ]]; then
  LOCK_STATUS=$(python - <<PY
import json
from pathlib import Path
p = Path("${F3_ONCE_LOCK}")
try:
    print(json.loads(p.read_text(encoding="utf-8")).get("status", ""))
except Exception:
    print("INVALID_LOCK")
PY
)
  if [[ "${LOCK_STATUS}" == STOPPED* ]]; then
    echo "F3 once-lock is stopped (${LOCK_STATUS}). Refusing to rerun." >&2
    exit 2
  fi
fi
```

注意：这段 guard 只阻止误跑，不改变任何实验结果。

---

## 8. 可选修改 `tools/official/check_tce_f3_preflight.py`

如果你希望在 Python 层也防误跑，而不是只靠 shell script，在 preflight 开头加入：

```python
from pathlib import Path
import json


def assert_f3_not_already_stopped(root: str | Path) -> None:
    root = Path(root)
    fail_summary = root / "docs" / "internal" / "tce_final" / "gate_tce_f3_fail_summary.json"
    once_lock = root / "docs" / "internal" / "tce_final" / "gate_tce_f3_once_lock.json"

    if fail_summary.exists():
        raise RuntimeError(
            "F3 fail summary already exists. This repository is stopped at "
            "F3_FAIL_NO_REDESIGN; blind/external once must not be rerun."
        )

    if once_lock.exists():
        try:
            status = json.loads(once_lock.read_text(encoding="utf-8")).get("status", "")
        except Exception as exc:
            raise RuntimeError(f"Invalid F3 once-lock JSON: {once_lock}: {exc}") from exc
        if str(status).startswith("STOPPED"):
            raise RuntimeError(
                f"F3 once-lock is already stopped ({status}); blind/external once must not be rerun."
            )
```

然后在 `main()` 解析参数后、任何 evaluation 前调用：

```python
assert_f3_not_already_stopped(args.root)
```

如果当前 preflight 没有 `--root`，就用默认：

```python
root = Path(__file__).resolve().parents[2]
assert_f3_not_already_stopped(root)
```

---

## 9. 新增 `tests/test_final_stop_state.py`

新增文件：

```text
tests/test_final_stop_state.py
```

完整测试代码：

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.official.check_final_stop_state import (
    FinalStopStateError,
    check_final_stop_state,
)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_minimal_stopped_repo(root: Path) -> None:
    tce_dir = root / "docs" / "internal" / "tce_final"
    write_json(
        tce_dir / "gate_tce_f3_fail_summary.json",
        {
            "gate": "Gate-TCE-F3-blind-external-once",
            "decision": "F3_FAIL_NO_REDESIGN",
            "failed_splits": {
                "external_nuaa_sirst": {"min_delta_Pd": -0.018348624},
                "external_irstd_1k": {"min_delta_Pd": -0.013468013},
            },
            "not_completed_splits": {
                "external_sirst3": {
                    "reason": "manifest integrity failure: missing masks/images",
                    "total_entries": 1079,
                    "missing_masks": 365,
                    "missing_images": 1,
                }
            },
            "forbidden_next_actions": [
                "threshold search",
                "seed search",
                "checkpoint search",
                "split redefinition",
                "new model training",
            ],
        },
    )
    write_json(
        tce_dir / "gate_tce_f3_once_lock.json",
        {"status": "STOPPED_BY_F3_PD_REGRESSION"},
    )
    write_text(
        root / "README.md",
        """
# OHCM-MSHNet

## Current Official Status

STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
STOP_TCSR_AT_BANK_AUDIT
No active AAAI main-method branch remains.

Forbidden: no new training, no new evaluation, no threshold search.
""".strip(),
    )
    write_text(
        root / "STOPPED_BRANCHES_SUMMARY.md",
        """
# Stopped Branches Summary

STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
F3_FAIL_NO_REDESIGN because of external Pd regression.
STOP_TCSR_AT_BANK_AUDIT.
STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD.
""".strip(),
    )


def test_final_stop_state_passes(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    out = tmp_path / "docs" / "internal" / "final_stop_state_summary.json"

    result = check_final_stop_state(root=tmp_path, output_path=out)

    assert result["gate_pass"] is True
    assert result["decision"] == "READ_ONLY_FAILURE_ARCHIVE_STATE"
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["gate"] == "Gate-FINAL-STOP-CONSISTENCY"


def test_final_stop_state_fails_if_lock_is_started(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_json(
        tmp_path / "docs" / "internal" / "tce_final" / "gate_tce_f3_once_lock.json",
        {"status": "STARTED"},
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)


def test_final_stop_state_fails_if_no_negative_pd(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_json(
        tmp_path / "docs" / "internal" / "tce_final" / "gate_tce_f3_fail_summary.json",
        {
            "gate": "Gate-TCE-F3-blind-external-once",
            "decision": "F3_FAIL_NO_REDESIGN",
            "failed_splits": {
                "external_nuaa_sirst": {"min_delta_Pd": 0.0},
            },
        },
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)


def test_final_stop_state_fails_if_final_report_exists(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_json(
        tmp_path / "docs" / "internal" / "tce_final" / "gate_tce_f3_blind_external_report.json",
        {"should_not_exist": True},
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)


def test_final_stop_state_fails_if_readme_top_still_active(tmp_path: Path) -> None:
    make_minimal_stopped_repo(tmp_path)
    write_text(
        tmp_path / "README.md",
        """
# OHCM-MSHNet

## Current Official Status

Current active candidate:
- TWA-OHEM without BN recalibration
- next allowed gate: Gate-TWA-D HC-Val on seed42 only
""".strip(),
    )

    with pytest.raises(FinalStopStateError):
        check_final_stop_state(root=tmp_path)
```

---

## 10. README 顶部最终状态块

把 README 顶部改成这个状态块。历史路线可以保留在后面 archive section，但顶部必须清楚：没有 active branch。

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Final development decision:
- STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION.
- STOP_TCSR_AT_BANK_AUDIT.
- STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD.
- STOP_TWA_NO_BN_AT_GATE_E.
- STOP_LATE_SNAPSHOT_EP250_AT_GATE_A.

No active AAAI main-method branch remains.

Latest final gate:
- Gate-TCE-F3-blind-external-once was started after F0/F1/F2 passed.
- Preflight passed and once-lock was created.
- F3 stopped because completed external splits showed Pd regression:
  - external_nuaa_sirst: min_delta_Pd = -0.018348624
  - external_irstd_1k: min_delta_Pd = -0.013468013
- external_sirst3 was not completed due to manifest integrity failure:
  - total entries = 1079
  - missing masks = 365
  - missing images = 1

Decision:
- F3_FAIL_NO_REDESIGN.
- Do not generate an F3 final report.
- Do not rerun blind/external.
- Do not modify external split definitions.
- Do not use SIRST3 mask imputation or labeled-subset rescue.

Allowed next work:
- final stop-state consistency checker
- stopped-branch archive
- failure-analysis table
- repository freeze

Forbidden next work:
- new training
- new evaluation
- threshold search
- seed search
- checkpoint search
- BN tuning
- TCSR Stage 2
- new verifier / suppression head
- changes to loss.py / net.py / train.py / dataset.py / model/
```

---

## 11. STOPPED_BRANCHES_SUMMARY 增补块

在 `STOPPED_BRANCHES_SUMMARY.md` 顶部或 TCE section 加：

```markdown
## TCE-4 stopped at F3 external Pd regression

Decision: STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION.

Gate:
- Gate-TCE-F3-blind-external-once.

Once-lock:
- gate_tce_f3_once_lock.json status: STOPPED_BY_F3_PD_REGRESSION.

Fail summary:
- gate_tce_f3_fail_summary.json decision: F3_FAIL_NO_REDESIGN.

Completed external split failures:
- external_nuaa_sirst: min_delta_Pd = -0.018348624.
- external_irstd_1k: min_delta_Pd = -0.013468013.

Not completed:
- external_sirst3 was stopped due to manifest integrity failure:
  - test_SIRST3.txt entries: 1079
  - missing masks: 365
  - missing images: 1

Interpretation:
- TCE-4 is an internal trajectory-consensus oracle that reduces internal hard-clutter false alarms.
- It is not an external Pd-safe final AAAI method.
- F3 is a frozen-method once gate, so no rescue is allowed.

Forbidden:
- threshold search
- seed search
- checkpoint search
- external split redefinition
- SIRST3 mask imputation
- SIRST3 labeled-subset rescue
- new model training
- new verifier / suppression head
- TCSR Stage 2

Allowed:
- final stop-state checker
- archive / failure-analysis report
- repository freeze
```

---

## 12. 执行命令

在仓库根目录执行：

```bash
# 1. 语法检查
python -m py_compile \
  tools/official/check_final_stop_state.py \
  tools/official/check_tce_f3_preflight.py

# 2. pytest
pytest tests/test_final_stop_state.py -q

# 3. diff hygiene
git diff --check

# 4. 生成最终停止态 summary
bash scripts/official/run_final_stop_state_check.sh
```

预期结果：

```text
Gate-FINAL-STOP-CONSISTENCY PASS

docs/internal/final_stop_state_summary.json:
  gate_pass: true
  decision: READ_ONLY_FAILURE_ARCHIVE_STATE
```

---

## 13. 如果这一步失败，怎么改代码

### 13.1 允许修的情况

如果失败原因是：

```text
JSON key 名不一致
README 顶部缺少 STOP token
STOPPED summary 缺少 F3_FAIL_NO_REDESIGN
路径写错
final_stop_state_summary.json 输出目录不存在
pytest fixture 写错
shell ROOT 默认值不对
```

允许修：

```text
tools/official/check_final_stop_state.py
scripts/official/run_final_stop_state_check.sh
tests/test_final_stop_state.py
README.md
STOPPED_BRANCHES_SUMMARY.md
```

### 13.2 不允许修的情况

如果失败原因涉及：

```text
想让 negative Pd 变成非负
想让 F3 final report 存在也通过
想把 STARTED lock 当成合法
想把 README 写成还有 next allowed gate
想删除 failed split
想补 SIRST3 mask 后继续
```

不允许修 checker 去迁就这些情况。应保持 fail。

---

## 14. 最终归档表

建议再人工整理一个文件，不需要脚本强制：

```text
docs/internal/final_failure_closure_table.md
```

表格内容：

```markdown
| Route | Positive evidence | Stop gate | Stop reason | Rescue forbidden? |
|---|---|---|---|---|
| TWA-4 no-BN | seed42 Full and HC-Val positive | Gate-E | not better than single late checkpoint control | yes |
| LateSnapshot-ep250 | HC-Val strong | Gate-LS-A | Full split unsafe | yes |
| LateSnapshot-ep300 | Full-safe, HC-Val positive | E2-FSC / post-hoc analysis | +0.000204 over TWA-4 is numerical tie; not structural method | yes |
| TCSR-v1 | clean target leakage = 0 | Gate-TCSR-A | sparse hard-clutter negative signal almost absent | yes |
| TCE-4 | F0/F1/F2 internal evidence passed | Gate-TCE-F3 | external Pd regression | yes |
| MSHNetOHEM | strong baseline anchor | not stopped | final anchor, not new method | n/a |
```

这个表是后续写复盘、组会、论文重判最有用的材料。

---

## 15. 当前论文策略

按当前 gate 体系，不能再写成：

```text
我们提出的新方法在 internal 和 external 上稳定降低 FA 且不损 Pd。
```

可以写成：

```text
MSHNetOHEM 是强 baseline。
TCE-4 暴露了 internal hard-clutter trajectory-consensus signal。
但是 TWA / ep300 / TCSR 均未能把该 signal 干净压缩为 external Pd-safe 的 final method。
```

现实选择：

```text
A. 停止 AAAI 主方法投稿，归档失败证据。
B. 改成 failure-aware reliability study / negative result，但主会风险很高。
C. 以 MSHNetOHEM 为强锚点，整理复现实验和失败机制，为下一轮方法设计服务。
```

---

## 16. 最终决策

```text
Decision:
  PROCEED_TO_GATE_FINAL_STOP_CONSISTENCY_ONLY

Meaning:
  The project is no longer in method-development mode.
  The only next code modification is final stop-state consistency and anti-rerun protection.

Allowed code changes:
  tools/official/check_final_stop_state.py
  scripts/official/run_final_stop_state_check.sh
  tests/test_final_stop_state.py
  README.md
  STOPPED_BRANCHES_SUMMARY.md
  F3 script/preflight rerun guard

Forbidden code changes:
  loss.py
  net.py
  train.py
  dataset.py
  model/
  metrics or threshold logic for rescue

Success criterion:
  Gate-FINAL-STOP-CONSISTENCY PASS
  final_stop_state_summary.json written
  no active next_allowed_gate remains

After success:
  Freeze repository.
  Do not run more experiments.
```

一句话：

> 下一步不是继续救结果，而是把“所有主线已严格停止”的状态写成机器可检查、pytest 可验证、脚本不可误跑的仓库级最终停止态。
