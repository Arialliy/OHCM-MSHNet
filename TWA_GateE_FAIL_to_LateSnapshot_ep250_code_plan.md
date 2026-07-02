# OHCM-MSHNet / TWA-OHEM：Gate-E FAIL 后的逐步修正方案与代码修改清单

> 当前结论：`STOP_TWA_NO_BN_AT_GATE_E` 不能改写、不能绕过。  
> 当前失败点：`TWA-4 without BN` 在 HC-Val 上显著差于 best single late checkpoint `ep250`。  
> 当前保留证据：Gate-D 仍然有效；TWA-4 vs OHEM、TCE retention、TWA-2/3/4 trend 仍然有效。  
> 当前下一步：不要推翻前面所有代码；只新增一个 **late snapshot ep250 rescue gate**，验证 `ep250` 是否可以作为冻结的单模型候选。  
> 当前仍然禁止：seed43/44、HC-Test、blind、external、BN tuning、新模型结构、新 loss、新 verifier、新 suppression head、穷举 checkpoint combination。

---

## 1. 先给结论

这次 Gate-E 的失败非常明确：

```text
Gate-E result: FAIL
failed condition:
  twa4_not_worse_than_best_single_late_hcval = false

best single late checkpoint:
  ep250

HC-Val:
  ep250 mIoU = 0.710648
  TWA-4 mIoU = 0.633891

TWA-4 - ep250:
  mIoU      = -0.076757
  FA ppm    = +116.984
  Precision = -0.085791
  Pd        =  0.0
```

这说明：

```text
TWA-4 weight averaging 不是当前最优机制。
```

但这不说明：

```text
整个训练轨迹路线无效。
```

更准确的解释是：

```text
ep250 是当前 seed42 late trajectory 中明显更强的 hard-clutter checkpoint。
TWA-4 把 ep250 的 hard-clutter 优势平均掉了。
```

因此当前不应该：

```text
降低 Gate-E 阈值
删掉 best single late 条件
继续试 TWA-5 / TWA-6 / TWA excluding ep250 / TWA weighted combination
回到 BN recalibration
回到 verifier / suppression branch
改 model/loss/train.py
```

应该做的是：

```text
保留 Gate-E FAIL；停止 TWA-4 no-BN；
从 Gate-E 已经发现的唯一 winner ep250 出发，新增一个严格的 late-snapshot gate。
```

这个新 gate 不是新训练、不是新结构、不是重新搜索，而是对 Gate-E 失败暴露出的候选进行最小验证。

---

## 2. 已有通过部分不要推翻

### 2.1 继续保留的文件

这些文件已经通过测试，不要重写：

```text
utils/twa_gate_utils.py
tools/official/check_twa_gate_e_mechanism.py
scripts/official/run_twa_gate_e_seed42.sh
tests/test_twa_gate_e_mechanism.py
docs/internal/twa/seed42_nudt/gate_twa_d_summary.json
docs/internal/twa/seed42_nudt/gate_twa_e_summary.json
```

### 2.2 继续保留的结果

Gate-D 仍然是有效结果：

```text
TWA-4 without BN vs OHEM-400 on seed42 HC-Val:
  mIoU      +0.02910
  FA ppm    -58.4920
  Precision +0.03545
  Pd        +0.00000
```

Gate-E 中这些条件也仍然有效：

```text
Gate-D 允许 Gate-E：PASS
TWA-4 Full vs OHEM 不退化：PASS
TWA-4 HC-Val vs OHEM 提升：PASS
TCE retention：PASS
TWA-2/3/4 trend：PASS
```

只失败这一条：

```text
TWA-4 HC-Val 不差于 best single late checkpoint：FAIL
```

所以代码修改应该是局部追加，而不是全部推翻。

---

## 3. 从 Gate-E 结果反推 ep250 的强度

已知：

```text
TWA-4 vs OHEM on HC-Val:
  ΔmIoU      = +0.02910
  ΔFA ppm    = -58.4920
  ΔPrecision = +0.03545
  ΔPd        = +0.00000

TWA-4 vs ep250 on HC-Val:
  ΔmIoU      = -0.076757
  ΔFA ppm    = +116.984
  ΔPrecision = -0.085791
  ΔPd        =  0.0
```

所以可以推导：

```text
ep250 vs OHEM on HC-Val:
  ΔmIoU      = +0.105857
  ΔFA ppm    = -175.476
  ΔPrecision = +0.121241
  ΔPd        = +0.00000
```

这不是小幅波动，是强信号。

但是它仍然不能直接进 seed43/44，因为：

```text
1. ep250 是在 Gate-E 的 HC-Val single-late comparison 中被识别出来的；
2. 这存在 seed42 HC-Val selection bias；
3. 还没有证明 ep250 在 seed42 Full split 不退化；
4. 还没有证明 ep250 在 seed43/44 稳定；
5. 还没有 threshold-matched / FP component / blind / external。
```

所以新分支必须从一个更小的 Gate 开始。

---

## 4. 新分支命名

建议新增分支名：

```text
LateSnapshot-ep250
```

或短名：

```text
LS-ep250
```

不要叫：

```text
TWA-v2
TWA-rescue
TWA-selected
```

原因：Gate-E 已经证明 TWA-4 的 weight averaging 机制不成立；如果继续叫 TWA，会把论文逻辑弄乱。

建议论文内部表述：

```text
Gate-E rejected weight-space averaging as the compression mechanism because a single late snapshot dominated TWA-4 on hard-clutter validation. We therefore freeze the discovered snapshot ep250 and evaluate it as a late-trajectory single-snapshot candidate.
```

中文解释：

```text
Gate-E 否定的是“平均权重压缩”的机制；
但没有否定“训练轨迹中存在更可靠 hard-clutter snapshot”的现象。
```

---

## 5. 下一步 Gate 设计：Gate-LS-A

### 5.1 Gate-LS-A 的作用

Gate-LS-A 只回答一个问题：

```text
在不重训、不调参、不换阈值、不跑新种子的前提下，
Gate-E 发现的 ep250 是否有资格从 diagnostic winner 变成 active candidate？
```

### 5.2 Gate-LS-A 输入

必须输入：

```text
docs/internal/twa/seed42_nudt/gate_twa_e_summary.json
OHEM-400 seed42 Full summary
ep250 seed42 Full summary
OHEM-400 seed42 HC-Val summary
ep250 seed42 HC-Val summary
TWA-4 seed42 HC-Val summary
```

如果 `ep250 seed42 Full summary` 还没有，就只补这一项 evaluation。不要训练新模型；只评估已有 ep250 checkpoint。

### 5.3 Gate-LS-A 条件

必须全部满足：

| 条件 | 要求 |
|---|---|
| Gate-E 状态 | `gate_pass=false` 且 decision/next gate 为 `STOP_TWA_NO_BN_AT_GATE_E` |
| Gate-E 失败原因 | 只能失败 `twa4_not_worse_than_best_single_late_hcval` |
| best single | 必须是冻结的 `ep250` |
| winner 唯一性 | ep250 HC-Val mIoU 至少比第二名单 checkpoint 高 `0.005` |
| ep250 Full vs OHEM | mIoU、Precision、Pd 不下降，FA 不上升 |
| ep250 HC-Val vs OHEM | mIoU 至少 `+0.005`，FA 至少下降 `10 ppm`，Precision/Pd 不下降 |
| ep250 HC-Val vs TWA-4 | mIoU 至少 `+0.005`，FA 至少下降 `10 ppm`，Precision/Pd 不下降 |

### 5.4 Gate-LS-A PASS 后才允许

```text
Gate-LS-B:
  seed43/44 paired Full + HC-Val
  fixed epoch = 250
  fixed model = MSHNetOHEM late snapshot
  fixed threshold = 0.5
  paired baseline = corresponding OHEM-400
```

仍然不允许：

```text
HC-Test
blind
external
threshold search
BN recalibration
new model training for architecture/loss tuning
new checkpoint combination search
```

### 5.5 Gate-LS-A FAIL 后停止

```text
STOP_LATE_SNAPSHOT_EP250_AT_GATE_A
```

如果 ep250 Full 不过，不能继续用 HC-Val 强结果硬推。

---

## 6. 代码修改总览

这次只做局部追加：

```text
[新增] tools/official/check_late_snapshot_gate_a_seed42.py
[新增] scripts/official/run_late_snapshot_ep250_gate_a_seed42.sh
[新增] tests/test_late_snapshot_gate_a_seed42.py
[修改] README.md 顶部 Current Official Status
[可选] STOPPED_BRANCHES_SUMMARY.md 增加 TWA no-BN stopped at Gate-E
```

不修改：

```text
model/
net.py
loss.py
train.py 主训练逻辑
dataset.py
probability.py
tools/official/check_twa_gate_e_mechanism.py 的 Gate-E 判定逻辑
```

理由：Gate-E checker 没错，失败是真失败。现在要增加的是 **失败后的下一步状态机**，不是把失败改成通过。

---

## 7. 新增文件：`tools/official/check_late_snapshot_gate_a_seed42.py`

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.twa_gate_utils import (  # noqa: E402
    DeltaRecord,
    MetricRecord,
    delta_metrics,
    load_json,
    load_metrics,
    metrics_from_summary,
    pass_hcval_improvement,
    pass_nonregression,
    write_json,
)


BEST_SINGLE_CONDITION = "twa4_not_worse_than_best_single_late_hcval"
STOP_TWA_GATE_E = "STOP_TWA_NO_BN_AT_GATE_E"

FORBIDDEN_IF_FAIL = [
    "seed43",
    "seed44",
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new model training",
    "new verifier",
    "new suppression head",
    "new checkpoint combination search",
    "threshold search",
]

FORBIDDEN_BEFORE_LATER_GATES = [
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new verifier",
    "new suppression head",
    "new checkpoint combination search",
    "threshold search",
]


MetricMap = Mapping[str, Any]


def _metric_dict(record: MetricRecord) -> dict[str, float]:
    return record.to_dict()


def _delta_dict(delta: DeltaRecord) -> dict[str, float]:
    return delta.to_dict()


def _failed_conditions(conditions: Mapping[str, Any]) -> list[str]:
    # JSON booleans are expected. Anything other than True is treated as not passed.
    return sorted(str(key) for key, value in conditions.items() if value is not True)


def _gate_e_failure_report(gate_e_summary: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    conditions = gate_e_summary.get("conditions")
    if not isinstance(conditions, Mapping):
        return False, {
            "reason": "missing_or_invalid_conditions",
            "conditions_type": type(conditions).__name__,
        }

    failed = _failed_conditions(conditions)
    gate_pass = gate_e_summary.get("gate_pass") is True
    decision = gate_e_summary.get("decision")
    next_allowed_gate = gate_e_summary.get("next_allowed_gate")

    stopped_at_gate_e = (
        gate_pass is False
        and (decision == STOP_TWA_GATE_E or next_allowed_gate == STOP_TWA_GATE_E)
    )
    only_best_single_failed = failed == [BEST_SINGLE_CONDITION]

    ok = stopped_at_gate_e and only_best_single_failed
    return ok, {
        "gate_pass": gate_pass,
        "decision": decision,
        "next_allowed_gate": next_allowed_gate,
        "failed_conditions": failed,
        "required_failed_conditions": [BEST_SINGLE_CONDITION],
        "stopped_at_gate_e": stopped_at_gate_e,
        "only_best_single_failed": only_best_single_failed,
        "pass": ok,
    }


def _records_from_gate_e_best_single(gate_e_summary: Mapping[str, Any]) -> dict[str, MetricRecord]:
    best_block = gate_e_summary.get("best_single_late_checkpoint")
    if not isinstance(best_block, Mapping):
        return {}

    records: dict[str, MetricRecord] = {}
    all_single = best_block.get("all_single_late_hcval")
    if isinstance(all_single, Mapping):
        for name, payload in all_single.items():
            if isinstance(payload, Mapping):
                records[str(name)] = metrics_from_summary(payload)

    best_name = best_block.get("name")
    best_metrics = best_block.get("metrics")
    if isinstance(best_name, str) and isinstance(best_metrics, Mapping) and best_name not in records:
        records[best_name] = metrics_from_summary(best_metrics)

    return records


def _metric_priority(item: tuple[str, MetricRecord]) -> tuple[float, float, float, float]:
    name, metrics = item
    del name
    # Higher mIoU, lower FA, higher Precision, higher Pd.
    return (metrics.mIoU, -metrics.FA_ppm, metrics.Precision, metrics.Pd)


def _best_single_report(
    *,
    gate_e_summary: Mapping[str, Any],
    expected_snapshot_name: str,
    min_unique_miou_margin: float,
) -> tuple[bool, dict[str, Any]]:
    best_block = gate_e_summary.get("best_single_late_checkpoint")
    if not isinstance(best_block, Mapping):
        return False, {"reason": "missing_best_single_late_checkpoint_block"}

    reported_name = best_block.get("name")
    records = _records_from_gate_e_best_single(gate_e_summary)
    if not records:
        return False, {
            "reason": "missing_all_single_late_hcval_records",
            "reported_name": reported_name,
        }

    ranked = sorted(records.items(), key=_metric_priority, reverse=True)
    computed_best_name, computed_best_metrics = ranked[0]
    second_name = ranked[1][0] if len(ranked) > 1 else None
    second_metrics = ranked[1][1] if len(ranked) > 1 else None
    unique_miou_margin = None
    unique_pass = False
    if second_metrics is not None:
        unique_miou_margin = computed_best_metrics.mIoU - second_metrics.mIoU
        unique_pass = unique_miou_margin >= min_unique_miou_margin

    name_matches_report = reported_name == computed_best_name
    name_is_expected = reported_name == expected_snapshot_name

    ok = name_matches_report and name_is_expected and unique_pass
    return ok, {
        "reported_best_single": reported_name,
        "computed_best_single": computed_best_name,
        "expected_snapshot_name": expected_snapshot_name,
        "name_matches_report": name_matches_report,
        "name_is_expected": name_is_expected,
        "second_best_single": second_name,
        "unique_miou_margin": unique_miou_margin,
        "min_unique_miou_margin": min_unique_miou_margin,
        "unique_pass": unique_pass,
        "ranked_single_late_hcval": [
            {
                "name": name,
                "metrics": _metric_dict(metrics),
            }
            for name, metrics in ranked
        ],
        "pass": ok,
    }


def _comparison_pass_hcval(
    delta: DeltaRecord,
    *,
    min_delta_miou: float,
    min_fa_reduction: float,
    min_delta_precision: float = 0.0,
    min_delta_pd: float = 0.0,
) -> bool:
    return pass_hcval_improvement(
        delta,
        min_delta_miou=min_delta_miou,
        min_fa_reduction=min_fa_reduction,
        min_delta_precision=min_delta_precision,
        min_delta_pd=min_delta_pd,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate-LS-A checker: promote the Gate-E best single late checkpoint "
            "to a frozen late-snapshot candidate only if seed42 Full and HC-Val guards pass."
        )
    )
    parser.add_argument("--gate_e_summary", required=True)
    parser.add_argument("--ohem_full", required=True)
    parser.add_argument("--ohem_hcval", required=True)
    parser.add_argument("--snapshot_full", required=True)
    parser.add_argument("--snapshot_hcval", required=True)
    parser.add_argument("--twa4_hcval", required=True)
    parser.add_argument("--snapshot_name", default="ep250")
    parser.add_argument("--output", required=True)

    parser.add_argument("--min_unique_miou_margin", type=float, default=0.005)
    parser.add_argument("--min_hcval_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_hcval_fa_reduction", type=float, default=10.0)
    parser.add_argument("--min_snapshot_vs_twa4_miou", type=float, default=0.005)
    parser.add_argument("--min_snapshot_vs_twa4_fa_reduction", type=float, default=10.0)
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gate_e_summary = load_json(args.gate_e_summary)

    gate_e_ok, gate_e_report = _gate_e_failure_report(gate_e_summary)
    best_single_ok, best_single_report = _best_single_report(
        gate_e_summary=gate_e_summary,
        expected_snapshot_name=args.snapshot_name,
        min_unique_miou_margin=args.min_unique_miou_margin,
    )

    ohem_full = load_metrics(args.ohem_full)
    ohem_hcval = load_metrics(args.ohem_hcval)
    snapshot_full = load_metrics(args.snapshot_full)
    snapshot_hcval = load_metrics(args.snapshot_hcval)
    twa4_hcval = load_metrics(args.twa4_hcval)

    snapshot_full_delta_vs_ohem = delta_metrics(snapshot_full, ohem_full)
    snapshot_hcval_delta_vs_ohem = delta_metrics(snapshot_hcval, ohem_hcval)
    snapshot_hcval_delta_vs_twa4 = delta_metrics(snapshot_hcval, twa4_hcval)

    full_nonregression_pass = pass_nonregression(snapshot_full_delta_vs_ohem)
    hcval_vs_ohem_pass = _comparison_pass_hcval(
        snapshot_hcval_delta_vs_ohem,
        min_delta_miou=args.min_hcval_delta_miou,
        min_fa_reduction=args.min_hcval_fa_reduction,
    )
    hcval_vs_twa4_pass = _comparison_pass_hcval(
        snapshot_hcval_delta_vs_twa4,
        min_delta_miou=args.min_snapshot_vs_twa4_miou,
        min_fa_reduction=args.min_snapshot_vs_twa4_fa_reduction,
    )

    conditions = {
        "gate_e_failed_only_because_best_single_won": gate_e_ok,
        "best_single_is_frozen_snapshot": best_single_ok,
        "snapshot_full_nonregression_vs_ohem": full_nonregression_pass,
        "snapshot_hcval_improvement_vs_ohem": hcval_vs_ohem_pass,
        "snapshot_hcval_improvement_vs_twa4": hcval_vs_twa4_pass,
    }
    gate_pass = all(conditions.values())

    result = {
        "gate": "Gate-LS-A",
        "method": "LateSnapshot-ep250",
        "origin": "Gate-TWA-E best single late checkpoint audit",
        "seed": 42,
        "threshold": 0.5,
        "gate_pass": gate_pass,
        "decision": (
            "PROCEED_LATE_SNAPSHOT_EP250_TO_GATE_LS_B_SEED43_44"
            if gate_pass
            else "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"
        ),
        "next_allowed_gate": (
            "Gate-LS-B-seed43-seed44-paired-Full-HCVal"
            if gate_pass
            else "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"
        ),
        "conditions": conditions,
        "gate_e_report": gate_e_report,
        "best_single_report": best_single_report,
        "ohem": {
            "Full": _metric_dict(ohem_full),
            "HC-Val": _metric_dict(ohem_hcval),
        },
        "snapshot": {
            "name": args.snapshot_name,
            "Full": _metric_dict(snapshot_full),
            "HC-Val": _metric_dict(snapshot_hcval),
            "delta_full_vs_ohem": _delta_dict(snapshot_full_delta_vs_ohem),
            "delta_hcval_vs_ohem": _delta_dict(snapshot_hcval_delta_vs_ohem),
            "delta_hcval_vs_twa4": _delta_dict(snapshot_hcval_delta_vs_twa4),
        },
        "twa4_without_bn": {
            "HC-Val": _metric_dict(twa4_hcval),
        },
        "gate_criteria": {
            "min_unique_miou_margin": args.min_unique_miou_margin,
            "full_nonregression": {
                "min_delta_mIoU": 0.0,
                "max_delta_FA_ppm": 0.0,
                "min_delta_Precision": 0.0,
                "min_delta_Pd": 0.0,
            },
            "hcval_vs_ohem": {
                "min_delta_mIoU": args.min_hcval_delta_miou,
                "min_fa_reduction_ppm": args.min_hcval_fa_reduction,
                "min_delta_Precision": 0.0,
                "min_delta_Pd": 0.0,
            },
            "hcval_vs_twa4": {
                "min_delta_mIoU": args.min_snapshot_vs_twa4_miou,
                "min_fa_reduction_ppm": args.min_snapshot_vs_twa4_fa_reduction,
                "min_delta_Precision": 0.0,
                "min_delta_Pd": 0.0,
            },
        },
        "forbidden_if_fail": FORBIDDEN_IF_FAIL,
        "forbidden_before_later_gates": FORBIDDEN_BEFORE_LATER_GATES,
        "notes": [
            "Do not rewrite Gate-E as PASS; Gate-E remains failed for TWA-4 weight averaging.",
            "This gate only validates the ep250 snapshot exposed by Gate-E; it does not authorize checkpoint search.",
            "If this gate passes, seed43/44 are allowed only for the frozen ep250-vs-ep400 paired protocol.",
            "HC-Test, blind, and external remain forbidden before later gates.",
        ],
    }
    return result


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    result = evaluate(args)
    write_json(args.output, result)
    if not result["gate_pass"]:
        raise SystemExit("Gate-LS-A failed. Stop LateSnapshot-ep250 before seed43/44.")


if __name__ == "__main__":
    main()
```

---

## 8. 新增脚本：`scripts/official/run_late_snapshot_ep250_gate_a_seed42.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}
BASE="${ROOT}/docs/internal/twa/seed42_nudt"

python "${ROOT}/tools/official/check_late_snapshot_gate_a_seed42.py" \
  --gate_e_summary "${BASE}/gate_twa_e_summary.json" \
  --ohem_full "${BASE}/eval_full_ohem/summary_metrics.json" \
  --ohem_hcval "${BASE}/eval_hcval_ohem/summary_metrics.json" \
  --snapshot_full "${BASE}/eval_full_single_ep250/summary_metrics.json" \
  --snapshot_hcval "${BASE}/eval_hcval_single_ep250/summary_metrics.json" \
  --twa4_hcval "${BASE}/eval_hcval_twa4_no_bn/summary_metrics.json" \
  --snapshot_name "ep250" \
  --output "${BASE}/gate_late_snapshot_ep250_a_summary.json"
```

赋权：

```bash
chmod +x scripts/official/run_late_snapshot_ep250_gate_a_seed42.sh
```

如果 `eval_full_single_ep250/summary_metrics.json` 不存在，只补这个 evaluation：

```bash
# 示例。具体 evaluator 参数以你当前 evaluate_twa_checkpoint.py 为准。
ROOT=/home/ly/AAAI/OHCM-MSHNet-main
BASE=${ROOT}/docs/internal/twa/seed42_nudt
EP250_CKPT=${EP250_CKPT:-${ROOT}/results/official/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_250.pth.tar}

python ${ROOT}/tools/official/evaluate_twa_checkpoint.py \
  --model_name MSHNetOHEM \
  --checkpoint "${EP250_CKPT}" \
  --dataset_name NUDT-SIRST \
  --split full \
  --output_dir "${BASE}/eval_full_single_ep250" \
  --threshold 0.5
```

注意：这里只是评估已有 `ep250` checkpoint，不是训练，不是调参。

---

## 9. 新增测试：`tests/test_late_snapshot_gate_a_seed42.py`

```python
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/official/check_late_snapshot_gate_a_seed42.py")


def _write_summary(path: Path, *, miou: float, fa: float, precision: float, pd: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "mIoU": miou,
                    "FA_ppm": fa,
                    "Precision": precision,
                    "Pd": pd,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _metric_payload(*, miou: float, fa: float, precision: float, pd: float) -> dict:
    return {
        "mIoU": miou,
        "FA_ppm": fa,
        "Precision": precision,
        "Pd": pd,
    }


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
                    "metrics": _metric_payload(
                        miou=ep250_miou,
                        fa=210.524,
                        precision=0.781241,
                        pd=0.833333,
                    ),
                    "all_single_late_hcval": {
                        "ep250": _metric_payload(
                            miou=ep250_miou,
                            fa=210.524,
                            precision=0.781241,
                            pd=0.833333,
                        ),
                        "ep300": _metric_payload(
                            miou=ep300_miou,
                            fa=290.0,
                            precision=0.720000,
                            pd=0.833333,
                        ),
                        "ep350": _metric_payload(
                            miou=0.640000,
                            fa=310.0,
                            precision=0.700000,
                            pd=0.833333,
                        ),
                        "ep400": _metric_payload(
                            miou=0.604791,
                            fa=386.000,
                            precision=0.660000,
                            pd=0.833333,
                        ),
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
        "ohem_full": _write_summary(
            tmp_path / "ohem_full.json",
            miou=0.833393,
            fa=63.449,
            precision=0.906277,
            pd=0.979894,
        ),
        "ohem_hcval": _write_summary(
            tmp_path / "ohem_hcval.json",
            miou=0.604791,
            fa=386.000,
            precision=0.660000,
            pd=0.833333,
        ),
        "snapshot_full": _write_summary(
            tmp_path / "snapshot_full.json",
            miou=0.835000,
            fa=62.000,
            precision=0.910000,
            pd=0.979894,
        ),
        "snapshot_hcval": _write_summary(
            tmp_path / "snapshot_hcval.json",
            miou=0.710648,
            fa=210.524,
            precision=0.781241,
            pd=0.833333,
        ),
        "twa4_hcval": _write_summary(
            tmp_path / "twa4_hcval.json",
            miou=0.633891,
            fa=327.508,
            precision=0.695450,
            pd=0.833333,
        ),
    }


def _cmd(files: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--gate_e_summary", str(files["gate_e"]),
        "--ohem_full", str(files["ohem_full"]),
        "--ohem_hcval", str(files["ohem_hcval"]),
        "--snapshot_full", str(files["snapshot_full"]),
        "--snapshot_hcval", str(files["snapshot_hcval"]),
        "--twa4_hcval", str(files["twa4_hcval"]),
        "--snapshot_name", "ep250",
        "--output", str(output),
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
    files["gate_e"] = _write_gate_e(
        tmp_path / "gate_e_not_unique.json",
        ep250_miou=0.710648,
        ep300_miou=0.708500,
    )
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["best_single_is_frozen_snapshot"] is False


def test_late_snapshot_gate_a_fails_if_snapshot_full_regresses(tmp_path):
    files = _base_files(tmp_path)
    files["snapshot_full"] = _write_summary(
        tmp_path / "snapshot_full_bad.json",
        miou=0.820000,
        fa=70.000,
        precision=0.890000,
        pd=0.979894,
    )
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["snapshot_full_nonregression_vs_ohem"] is False


def test_late_snapshot_gate_a_fails_if_snapshot_hcval_does_not_beat_ohem(tmp_path):
    files = _base_files(tmp_path)
    files["snapshot_hcval"] = _write_summary(
        tmp_path / "snapshot_hcval_bad_ohem.json",
        miou=0.606000,
        fa=380.000,
        precision=0.661000,
        pd=0.833333,
    )
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["snapshot_hcval_improvement_vs_ohem"] is False


def test_late_snapshot_gate_a_fails_if_snapshot_hcval_does_not_beat_twa4(tmp_path):
    files = _base_files(tmp_path)
    files["snapshot_hcval"] = _write_summary(
        tmp_path / "snapshot_hcval_bad_twa4.json",
        miou=0.635000,
        fa=326.000,
        precision=0.696000,
        pd=0.833333,
    )
    output = tmp_path / "gate_ls_a.json"
    result = subprocess.run(_cmd(files, output), text=True, capture_output=True)
    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["conditions"]["snapshot_hcval_improvement_vs_twa4"] is False
```

运行：

```bash
python -m py_compile tools/official/check_late_snapshot_gate_a_seed42.py
pytest tests/test_late_snapshot_gate_a_seed42.py -q
git diff --check
```

预期：

```text
7 passed
git diff --check: PASS
```

---

## 10. README 顶部状态块修改

把 README 顶部 Current Official Status 改成：

```markdown
## Current Official Status

Decision: `STOP_TWA_NO_BN_AT_GATE_E`.

Strong anchor: `MSHNetOHEM`.

Stopped TWA branches:

- `TWA with BN recalibration`: stopped at Gate-TWA-B.
- `TWA-4 without BN recalibration`: stopped at Gate-TWA-E.

Gate-TWA-E result:

- Gate-E status: FAIL.
- Failed condition: `twa4_not_worse_than_best_single_late_hcval`.
- Best single late checkpoint: `ep250`.
- ep250 HC-Val mIoU: `0.710648`.
- TWA-4 HC-Val mIoU: `0.633891`.
- TWA-4 relative to ep250:
  - mIoU: `-0.076757`
  - FA ppm: `+116.984`
  - Precision: `-0.085791`
  - Pd: `+0.000000`

Active diagnostic candidate:

- `LateSnapshot-ep250`
- Origin: Gate-TWA-E best-single-late audit.
- Status: pending `Gate-LS-A` seed42 Full + HC-Val guard.

Allowed now:

- Evaluate or check `ep250` on seed42 Full only if the Full summary is missing.
- Run `Gate-LS-A` for frozen `ep250`.

Forbidden before Gate-LS-A passes:

- seed43 / seed44
- HC-Test
- blind / external
- threshold search
- BN recalibration tuning
- new model training for architecture/loss tuning
- new checkpoint combination search
- new verifier / suppression head
```

不要删除旧失败路线；只把 TWA 状态推进到 Gate-E FAIL。

---

## 11. STOPPED_BRANCHES_SUMMARY.md 可选追加

```markdown
## TWA-4 without BN recalibration

Status: stopped at Gate-TWA-E.

Reason:

- TWA-4 passed OHEM comparison and TCE-retention checks,
  but failed the best-single-late mechanism check.
- Best single late checkpoint `ep250` dominated TWA-4 on seed42 HC-Val.

Decision:

- Do not run seed43/44 for TWA-4.
- Do not run HC-Test, blind, or external for TWA-4.
- Do not tune BN recalibration.
- Do not search new TWA checkpoint combinations.
- `ep250` may be evaluated only through the separate `LateSnapshot-ep250` Gate-LS-A protocol.
```

---

## 12. 最小执行顺序

```bash
# 1. 新增 checker 和 test
python -m py_compile tools/official/check_late_snapshot_gate_a_seed42.py
pytest tests/test_late_snapshot_gate_a_seed42.py -q
git diff --check

# 2. 如果 ep250 Full summary 不存在，只补 seed42 Full evaluation
# 不训练，不调参，不跑 HC-Test/blind/external。

# 3. 运行 Gate-LS-A
bash scripts/official/run_late_snapshot_ep250_gate_a_seed42.sh
```

Gate-LS-A PASS 后：

```text
允许进入 Gate-LS-B:
  seed43/44 paired Full + HC-Val
  fixed epoch = 250
  fixed threshold = 0.5
  paired OHEM baseline = epoch 400
```

Gate-LS-A FAIL 后：

```text
STOP_LATE_SNAPSHOT_EP250_AT_GATE_A
不要跑 seed43/44。
```

---

## 13. Gate-LS-B 预注册规则，只写状态，不现在实现

只有 Gate-LS-A PASS 后才实现 Gate-LS-B checker。

Gate-LS-B 应该要求：

| 项目 | 要求 |
|---|---|
| Seeds | 42、43、44 |
| Method | fixed `ep250` only |
| Baseline | paired `ep400` OHEM |
| Splits | Full、HC-Val |
| Full | 3/3 不退化，或至少 mIoU/Precision/Pd 不降且 FA 不升 |
| HC-Val | 至少 2/3 seed 提升，mean mIoU / FA / Precision 为正向，Pd 不下降 |
| 禁止 | HC-Test、blind、external、threshold search、epoch search |

如果 Gate-LS-B 过，再做 threshold-matched 和 FP component audit。

---

## 14. AAAI 15 天策略

现在最节省时间的路线是：

```text
Day 1:
  完成 Gate-LS-A checker / tests / README status。
  补 ep250 seed42 Full summary。
  跑 Gate-LS-A。

Day 2-4:
  如果 Gate-LS-A PASS，跑 seed43/44 paired ep250 vs ep400 Full + HC-Val。
  如果 checkpoint 已有，只评估；如果没有，按 frozen protocol 训练/保存对应 checkpoint。

Day 5-7:
  Gate-LS-B checker + 三种子表格。
  如果三种子不过，停止，不进入 blind/external。

Day 8-10:
  threshold-matched / FP component analysis。

Day 11-12:
  blind/external 一次性冻结评估。

Day 13-15:
  写论文、消融、失败路线 appendix。
```

如果 Gate-LS-A 没过，不建议继续试新算法。15 天内最可投稿的结果应该来自已有强信号，而不是重新开训练分支。

---

## 15. 最终建议

当前正式状态写成：

```text
Decision: STOP_TWA_NO_BN_AT_GATE_E

TWA-4 without BN is stopped because Gate-E failed:
  twa4_not_worse_than_best_single_late_hcval = false

Best single late checkpoint:
  ep250

Immediate next allowed action:
  Gate-LS-A for frozen LateSnapshot-ep250 on seed42 Full + HC-Val.

Still forbidden:
  seed43/44 before Gate-LS-A PASS,
  HC-Test,
  blind,
  external,
  BN tuning,
  new model/loss/verifier/suppression,
  new checkpoint combination search.
```

一句话：

> **不要把 Gate-E 改成 PASS，也不要重启 TWA 搜索；把 TWA-4 正式停止，然后用最小代码新增 Gate-LS-A，验证 Gate-E 暴露出的 ep250 是否能成为冻结单模型候选。**
