# OHCM-MSHNet / TWA-OHEM：Gate-D PASS 后的 Gate-E 代码方案与修改清单

> 当前结论：`PROCEED_TWA_NO_BN_TO_GATE_E`  
> 当前候选：`TWA without BN recalibration`  
> 当前允许：`Gate-TWA-E` 机制比较  
> 当前仍然禁止：`seed43/44`、`HC-Test`、`blind`、`external`、`BN recalibration tuning`、`new model training`、新 verifier/suppression 结构

---

## 1. 当前结果判定

你给出的 Gate-D 结果：

```text
mIoU delta:      +0.02910
FA ppm delta:    -58.4920
Precision delta: +0.03545
Pd delta:        +0.00000
next_allowed_gate: Gate-TWA-E
```

按之前预注册的 Gate-D 阈值：

| 指标 | Gate-D 要求 | 当前结果 | 判定 |
|---|---:|---:|---|
| HC-Val ΔmIoU | >= +0.005 | +0.02910 | PASS |
| HC-Val ΔFA ppm | <= -10 | -58.4920 | PASS |
| HC-Val ΔPrecision | >= 0 | +0.03545 | PASS |
| HC-Val ΔPd | >= 0 | +0.00000 | PASS |

结论：Gate-D 不是“擦边过”，而是强 PASS。下一步应进入 Gate-E，不应继续改模型、不应调 BN、不应跑 seed43/44、HC-Test、blind、external。

---

## 2. 代码层分析

### 2.1 不应该改的部分

当前不建议改：

```text
model/
net.py
loss.py
train.py 主训练逻辑
dataset.py
probability.py 的概率定义
```

原因：当前 TWA without BN 的收益来自 checkpoint weight-space compression，不来自新结构或新训练目标。现在改模型或 loss，会把“trajectory compression”证据污染成新的训练方法，AAAI 论文定位会变模糊。

### 2.2 应该改的部分

Gate-D PASS 后，工程应该补的是状态机和比较器，而不是模型代码：

```text
README.md
STOPPED_BRANCHES_SUMMARY.md
docs/internal/twa/seed42_nudt/gate_twa_d_summary.json
utils/twa_gate_utils.py
tools/official/check_twa_gate_e_mechanism.py
scripts/official/run_twa_gate_e_seed42.sh
tests/test_twa_gate_e_mechanism.py
```

### 2.3 代码风险点

当前最大风险不是算法，而是流程污染：

1. README 仍可能显示旧路线状态，导致后续误跑 APF/SPS/ERD 或误以为还停在 Gate-D。
2. Gate-D 的 `next_allowed_gate` 已经是 `Gate-TWA-E`，但代码里如果没有 Gate-E checker，就会靠人工判断，容易失控。
3. Gate-E 如果只看 TWA-4 vs OHEM，会被 reviewer 质疑：这是不是某个 late checkpoint 本来就更好？
4. Gate-E 必须显式比较 `best single late checkpoint`，否则无法支撑“trajectory compression”。
5. Gate-E 必须显式比较 `TCE-4`，否则无法说明 TWA 保留了多少 TCE 的 hard-split 收益。
6. Gate-E 必须比较 `TWA-2 / TWA-3 / TWA-4`，否则会像穷举偶然命中。

---

## 3. 修改 1：更新 Gate-D summary

文件：

```text
docs/internal/twa/seed42_nudt/gate_twa_d_summary.json
```

如果当前文件已经有 absolute metrics，保留原字段，只补下面这些字段：

```json
{
  "gate": "Gate-TWA-D",
  "method": "TWA without BN recalibration",
  "candidate": "twa_without_bn",
  "seed": 42,
  "split": "HC-Val",
  "threshold": 0.5,
  "gate_pass": true,
  "delta": {
    "mIoU": 0.02910,
    "FA_ppm": -58.4920,
    "Precision": 0.03545,
    "Pd": 0.0
  },
  "gate_criteria": {
    "min_delta_mIoU": 0.005,
    "max_delta_FA_ppm": -10.0,
    "min_delta_Precision": 0.0,
    "min_delta_Pd": 0.0
  },
  "decision": "PROCEED_TWA_NO_BN_TO_GATE_E",
  "next_allowed_gate": "Gate-TWA-E",
  "forbidden_next_actions": [
    "seed43",
    "seed44",
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new model training",
    "new verifier",
    "new suppression head"
  ],
  "validation": {
    "pytest_new_tests": "6 passed",
    "git_diff_check": "PASS",
    "seed43_44_run": false,
    "hc_test_run": false,
    "blind_run": false,
    "external_run": false
  }
}
```

---

## 4. 修改 2：新增通用 gate 工具

新增文件：

```text
utils/twa_gate_utils.py
```

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "mIoU": (
        "mIoU",
        "miou",
        "mean_iou",
        "mean_IoU",
        "MeanIoU",
        "iou",
    ),
    "FA_ppm": (
        "FA_ppm",
        "fa_ppm",
        "FAppm",
        "FA ppm",
        "FA",
        "false_alarm_ppm",
        "false_alarm_rate_ppm",
    ),
    "Precision": (
        "Precision",
        "precision",
        "Prec",
        "prec",
    ),
    "Pd": (
        "Pd",
        "pd",
        "PD",
        "target_pd",
        "detection_probability",
    ),
}

NESTED_METRIC_KEYS = (
    "metrics",
    "metric",
    "summary_metrics",
    "aggregate",
    "overall",
    "result",
    "results",
    "fixed_threshold",
    "at_threshold",
    "threshold_0.5",
)


@dataclass(frozen=True)
class MetricRecord:
    mIoU: float
    FA_ppm: float
    Precision: float
    Pd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DeltaRecord:
    mIoU: float
    FA_ppm: float
    Precision: float
    Pd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _norm_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _candidate_metric_maps(summary: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield summary
    for key in NESTED_METRIC_KEYS:
        value = summary.get(key)
        if isinstance(value, Mapping):
            yield value


def get_metric(summary: Mapping[str, Any], canonical_name: str) -> float:
    if canonical_name not in METRIC_ALIASES:
        raise KeyError(f"Unknown metric canonical name: {canonical_name}")

    aliases = METRIC_ALIASES[canonical_name]
    normalized_aliases = {_norm_key(alias) for alias in aliases}

    visible_keys: list[str] = []
    for metric_map in _candidate_metric_maps(summary):
        for key, value in metric_map.items():
            visible_keys.append(str(key))
            if key in aliases or _norm_key(str(key)) in normalized_aliases:
                try:
                    return float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Metric {canonical_name} found as {key!r}, but value is not numeric: {value!r}"
                    ) from exc

    raise KeyError(
        f"Metric {canonical_name!r} not found. Available keys: {sorted(set(visible_keys))}"
    )


def metrics_from_summary(summary: Mapping[str, Any]) -> MetricRecord:
    return MetricRecord(
        mIoU=get_metric(summary, "mIoU"),
        FA_ppm=get_metric(summary, "FA_ppm"),
        Precision=get_metric(summary, "Precision"),
        Pd=get_metric(summary, "Pd"),
    )


def load_metrics(path: str | Path) -> MetricRecord:
    return metrics_from_summary(load_json(path))


def delta_metrics(candidate: MetricRecord, baseline: MetricRecord) -> DeltaRecord:
    return DeltaRecord(
        mIoU=candidate.mIoU - baseline.mIoU,
        FA_ppm=candidate.FA_ppm - baseline.FA_ppm,
        Precision=candidate.Precision - baseline.Precision,
        Pd=candidate.Pd - baseline.Pd,
    )


def pass_hcval_improvement(
    delta: DeltaRecord,
    *,
    min_delta_miou: float = 0.005,
    min_fa_reduction: float = 10.0,
    min_delta_precision: float = 0.0,
    min_delta_pd: float = 0.0,
) -> bool:
    return (
        delta.mIoU >= min_delta_miou
        and delta.FA_ppm <= -min_fa_reduction
        and delta.Precision >= min_delta_precision
        and delta.Pd >= min_delta_pd
    )


def pass_nonregression(delta: DeltaRecord, *, eps: float = 1e-12) -> bool:
    return (
        delta.mIoU >= -eps
        and delta.FA_ppm <= eps
        and delta.Precision >= -eps
        and delta.Pd >= -eps
    )


def safe_positive_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Empty name in named path: {value!r}")
    if not path:
        raise ValueError(f"Empty path in named path: {value!r}")
    return name, Path(path)


def load_named_metrics(values: list[str]) -> dict[str, MetricRecord]:
    records: dict[str, MetricRecord] = {}
    for value in values:
        name, path = parse_named_path(value)
        if name in records:
            raise ValueError(f"Duplicate named summary: {name}")
        records[name] = load_metrics(path)
    return records
```

---

## 5. 修改 3：新增 Gate-E 机制比较器

新增文件：

```text
tools/official/check_twa_gate_e_mechanism.py
```

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.twa_gate_utils import (  # noqa: E402
    DeltaRecord,
    MetricRecord,
    delta_metrics,
    load_json,
    load_metrics,
    load_named_metrics,
    pass_hcval_improvement,
    pass_nonregression,
    safe_positive_ratio,
    write_json,
)


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
]

FORBIDDEN_BEFORE_LATER_GATES = [
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new model training",
    "new verifier",
    "new suppression head",
]


def _metric_dict(record: MetricRecord) -> dict[str, float]:
    return record.to_dict()


def _delta_dict(delta: DeltaRecord) -> dict[str, float]:
    return delta.to_dict()


def _best_single(records: dict[str, MetricRecord]) -> tuple[str, MetricRecord]:
    if not records:
        raise ValueError("Gate-E requires at least one --single_late NAME=summary.json entry.")
    return max(
        records.items(),
        key=lambda item: (
            item[1].mIoU,
            -item[1].FA_ppm,
            item[1].Precision,
            item[1].Pd,
        ),
    )


def _check_gate_d(gate_d_summary: str | None) -> tuple[bool, dict[str, Any]]:
    if gate_d_summary is None:
        return False, {"reason": "missing --gate_d_summary"}

    summary = load_json(gate_d_summary)
    gate_pass = bool(summary.get("gate_pass", False))
    next_allowed_gate = summary.get("next_allowed_gate")
    candidate = summary.get("candidate") or summary.get("current_candidate") or summary.get("method")

    ok = gate_pass and next_allowed_gate == "Gate-TWA-E"
    return ok, {
        "path": gate_d_summary,
        "gate_pass": gate_pass,
        "next_allowed_gate": next_allowed_gate,
        "candidate": candidate,
    }


def _retention_report(
    *,
    ohem_hcval: MetricRecord,
    twa4_hcval: MetricRecord,
    tce4_hcval: MetricRecord,
    min_tce_retention: float,
) -> tuple[bool, dict[str, Any]]:
    twa_delta = delta_metrics(twa4_hcval, ohem_hcval)
    tce_delta = delta_metrics(tce4_hcval, ohem_hcval)

    miou_retention = safe_positive_ratio(twa_delta.mIoU, tce_delta.mIoU)
    fa_retention = safe_positive_ratio(-twa_delta.FA_ppm, -tce_delta.FA_ppm)
    precision_retention = safe_positive_ratio(twa_delta.Precision, tce_delta.Precision)

    checks: dict[str, bool | None] = {
        "mIoU": None if miou_retention is None else miou_retention >= min_tce_retention,
        "FA_ppm": None if fa_retention is None else fa_retention >= min_tce_retention,
    }

    active_checks = [value for value in checks.values() if value is not None]
    ok = bool(active_checks) and all(active_checks)

    return ok, {
        "min_tce_retention": min_tce_retention,
        "twa4_delta_vs_ohem_hcval": _delta_dict(twa_delta),
        "tce4_delta_vs_ohem_hcval": _delta_dict(tce_delta),
        "retention_ratio": {
            "mIoU": miou_retention,
            "FA_ppm": fa_retention,
            "Precision": precision_retention,
        },
        "checked_metrics": checks,
        "pass": ok,
    }


def _variant_trend_report(
    *,
    ohem_hcval: MetricRecord,
    variant_hcval: dict[str, MetricRecord],
    trend_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    if "TWA-4" not in variant_hcval:
        return False, {
            "reason": "TWA-4 missing from --twa_variant_hcval entries",
            "variant_names": sorted(variant_hcval),
        }
    if len(variant_hcval) < 3:
        return False, {
            "reason": "Gate-E requires at least TWA-2/TWA-3/TWA-4 HC-Val summaries",
            "variant_names": sorted(variant_hcval),
        }

    records: dict[str, Any] = {}
    positive_count = 0
    for name, metrics in sorted(variant_hcval.items()):
        delta = delta_metrics(metrics, ohem_hcval)
        positive = (
            delta.mIoU >= -trend_tolerance
            and delta.FA_ppm <= trend_tolerance
            and delta.Precision >= -trend_tolerance
            and delta.Pd >= -trend_tolerance
        )
        if positive:
            positive_count += 1
        records[name] = {
            "metrics": _metric_dict(metrics),
            "delta_vs_ohem_hcval": _delta_dict(delta),
            "nonnegative_direction": positive,
        }

    min_variant_miou = min(metrics.mIoU for metrics in variant_hcval.values())
    twa4_not_worst = variant_hcval["TWA-4"].mIoU >= min_variant_miou - trend_tolerance
    enough_positive = positive_count >= 2
    ok = enough_positive and twa4_not_worst

    return ok, {
        "variant_names": sorted(variant_hcval),
        "records": records,
        "positive_count": positive_count,
        "required_positive_count": 2,
        "twa4_not_worst_by_mIoU": twa4_not_worst,
        "pass": ok,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate-TWA-E mechanism checker for TWA without BN on seed42 only."
    )
    parser.add_argument("--gate_d_summary", required=True)
    parser.add_argument("--ohem_full", required=True)
    parser.add_argument("--ohem_hcval", required=True)
    parser.add_argument("--twa4_full", required=True)
    parser.add_argument("--twa4_hcval", required=True)
    parser.add_argument("--tce4_hcval", required=True)
    parser.add_argument(
        "--single_late",
        action="append",
        default=[],
        help="Late single checkpoint summary as NAME=path. Repeat for ep250/ep300/ep350/ep400.",
    )
    parser.add_argument(
        "--twa_variant_hcval",
        action="append",
        default=[],
        help="TWA variant HC-Val summary as NAME=path, e.g. TWA-2=... Repeat for TWA-2/3/4.",
    )
    parser.add_argument(
        "--twa_variant_full",
        action="append",
        default=[],
        help="Optional TWA variant Full summary as NAME=path for reporting only.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_hcval_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_hcval_fa_reduction", type=float, default=10.0)
    parser.add_argument("--min_tce_retention", type=float, default=0.30)
    parser.add_argument("--trend_tolerance", type=float, default=1e-12)
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gate_d_ok, gate_d_report = _check_gate_d(args.gate_d_summary)

    ohem_full = load_metrics(args.ohem_full)
    ohem_hcval = load_metrics(args.ohem_hcval)
    twa4_full = load_metrics(args.twa4_full)
    twa4_hcval = load_metrics(args.twa4_hcval)
    tce4_hcval = load_metrics(args.tce4_hcval)

    single_late_records = load_named_metrics(args.single_late)
    best_single_name, best_single_metrics = _best_single(single_late_records)

    variant_hcval = load_named_metrics(args.twa_variant_hcval)
    variant_hcval.setdefault("TWA-4", twa4_hcval)
    variant_full = load_named_metrics(args.twa_variant_full)
    variant_full.setdefault("TWA-4", twa4_full)

    twa4_full_delta = delta_metrics(twa4_full, ohem_full)
    twa4_hcval_delta = delta_metrics(twa4_hcval, ohem_hcval)
    twa4_vs_best_single_delta = delta_metrics(twa4_hcval, best_single_metrics)

    full_guard_pass = pass_nonregression(twa4_full_delta)
    hcval_guard_pass = pass_hcval_improvement(
        twa4_hcval_delta,
        min_delta_miou=args.min_hcval_delta_miou,
        min_fa_reduction=args.min_hcval_fa_reduction,
    )
    best_single_guard_pass = pass_nonregression(twa4_vs_best_single_delta)
    retention_pass, retention = _retention_report(
        ohem_hcval=ohem_hcval,
        twa4_hcval=twa4_hcval,
        tce4_hcval=tce4_hcval,
        min_tce_retention=args.min_tce_retention,
    )
    trend_pass, trend = _variant_trend_report(
        ohem_hcval=ohem_hcval,
        variant_hcval=variant_hcval,
        trend_tolerance=args.trend_tolerance,
    )

    conditions = {
        "gate_d_passed_and_allows_gate_e": gate_d_ok,
        "twa4_full_nonregression_vs_ohem": full_guard_pass,
        "twa4_hcval_improvement_vs_ohem": hcval_guard_pass,
        "twa4_not_worse_than_best_single_late_hcval": best_single_guard_pass,
        "twa4_retains_tce_hard_split_gain": retention_pass,
        "twa2_twa3_twa4_trend_reasonable": trend_pass,
    }
    gate_pass = all(conditions.values())

    result = {
        "gate": "Gate-TWA-E",
        "method": "TWA without BN recalibration",
        "seed": 42,
        "split_scope": ["Full", "HC-Val"],
        "threshold": 0.5,
        "gate_pass": gate_pass,
        "decision": (
            "PROCEED_TWA_NO_BN_TO_THREE_SEED_GATE"
            if gate_pass
            else "STOP_TWA_NO_BN_AT_GATE_E"
        ),
        "next_allowed_gate": (
            "Gate-TWA-F-seed43-seed44-paired-Full-HCVal"
            if gate_pass
            else "STOP_TWA_NO_BN_AT_GATE_E"
        ),
        "conditions": conditions,
        "gate_d_report": gate_d_report,
        "ohem": {
            "Full": _metric_dict(ohem_full),
            "HC-Val": _metric_dict(ohem_hcval),
        },
        "twa4_without_bn": {
            "Full": _metric_dict(twa4_full),
            "HC-Val": _metric_dict(twa4_hcval),
            "delta_full_vs_ohem": _delta_dict(twa4_full_delta),
            "delta_hcval_vs_ohem": _delta_dict(twa4_hcval_delta),
        },
        "best_single_late_checkpoint": {
            "name": best_single_name,
            "metrics": _metric_dict(best_single_metrics),
            "twa4_delta_vs_best_single_hcval": _delta_dict(twa4_vs_best_single_delta),
            "all_single_late_hcval": {
                name: _metric_dict(metrics)
                for name, metrics in sorted(single_late_records.items())
            },
        },
        "tce4_hcval": _metric_dict(tce4_hcval),
        "tce_retention": retention,
        "twa_variant_trend_hcval": trend,
        "twa_variant_full_report": {
            name: {
                "metrics": _metric_dict(metrics),
                "delta_vs_ohem_full": _delta_dict(delta_metrics(metrics, ohem_full)),
            }
            for name, metrics in sorted(variant_full.items())
        },
        "forbidden_if_fail": FORBIDDEN_IF_FAIL,
        "forbidden_before_later_gates": FORBIDDEN_BEFORE_LATER_GATES,
        "notes": [
            "Gate-E is a mechanism/compression gate, not a new training gate.",
            "Do not run seed43/44 unless this gate passes.",
            "Do not run HC-Test, blind, or external at Gate-E.",
            "BN recalibration remains stopped and is not part of the active candidate.",
        ],
    }
    return result


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    result = evaluate(args)
    write_json(args.output, result)
    if not result["gate_pass"]:
        raise SystemExit("Gate-TWA-E failed. Stop TWA without BN before seed43/44.")


if __name__ == "__main__":
    main()
```

---

## 6. 修改 4：新增 Gate-E 一键比较脚本

新增文件：

```text
scripts/official/run_twa_gate_e_seed42.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}
BASE="${ROOT}/docs/internal/twa/seed42_nudt"

python "${ROOT}/tools/official/check_twa_gate_e_mechanism.py" \
  --gate_d_summary "${BASE}/gate_twa_d_summary.json" \
  --ohem_full "${BASE}/eval_full_ohem/summary_metrics.json" \
  --ohem_hcval "${BASE}/eval_hcval_ohem/summary_metrics.json" \
  --twa4_full "${BASE}/eval_full_twa4_no_bn/summary_metrics.json" \
  --twa4_hcval "${BASE}/eval_hcval_twa4_no_bn/summary_metrics.json" \
  --tce4_hcval "${BASE}/eval_hcval_tce4/summary_metrics.json" \
  --single_late "ep250=${BASE}/eval_hcval_single_ep250/summary_metrics.json" \
  --single_late "ep300=${BASE}/eval_hcval_single_ep300/summary_metrics.json" \
  --single_late "ep350=${BASE}/eval_hcval_single_ep350/summary_metrics.json" \
  --single_late "ep400=${BASE}/eval_hcval_ohem/summary_metrics.json" \
  --twa_variant_hcval "TWA-2=${BASE}/eval_hcval_twa2_no_bn/summary_metrics.json" \
  --twa_variant_hcval "TWA-3=${BASE}/eval_hcval_twa3_no_bn/summary_metrics.json" \
  --twa_variant_hcval "TWA-4=${BASE}/eval_hcval_twa4_no_bn/summary_metrics.json" \
  --twa_variant_full "TWA-2=${BASE}/eval_full_twa2_no_bn/summary_metrics.json" \
  --twa_variant_full "TWA-3=${BASE}/eval_full_twa3_no_bn/summary_metrics.json" \
  --twa_variant_full "TWA-4=${BASE}/eval_full_twa4_no_bn/summary_metrics.json" \
  --output "${BASE}/gate_twa_e_summary.json"
```

权限：

```bash
chmod +x scripts/official/run_twa_gate_e_seed42.sh
```

运行：

```bash
bash scripts/official/run_twa_gate_e_seed42.sh
```

注意：这个脚本只比较 summary，不训练、不跑 seed43/44、不碰 HC-Test/blind/external。

---

## 7. 修改 5：新增测试

新增文件：

```text
tests/test_twa_gate_e_mechanism.py
```

```python
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("tools/official/check_twa_gate_e_mechanism.py")


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
    files = {
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
    return files


def _cmd(files: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--gate_d_summary", str(files["gate_d"]),
        "--ohem_full", str(files["ohem_full"]),
        "--ohem_hcval", str(files["ohem_hc"]),
        "--twa4_full", str(files["twa4_full"]),
        "--twa4_hcval", str(files["twa4_hc"]),
        "--tce4_hcval", str(files["tce4_hc"]),
        "--single_late", f"ep250={files['single250']}",
        "--single_late", f"ep300={files['single300']}",
        "--single_late", f"ep350={files['single350']}",
        "--single_late", f"ep400={files['single400']}",
        "--twa_variant_hcval", f"TWA-2={files['twa2_hc']}",
        "--twa_variant_hcval", f"TWA-3={files['twa3_hc']}",
        "--twa_variant_hcval", f"TWA-4={files['twa4_hc']}",
        "--twa_variant_full", f"TWA-2={files['twa2_full']}",
        "--twa_variant_full", f"TWA-3={files['twa3_full']}",
        "--twa_variant_full", f"TWA-4={files['twa4_full']}",
        "--output", str(output),
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
    # Remove TWA-2 and TWA-3 variant entries, leaving only TWA-4.
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
```

运行：

```bash
python -m py_compile utils/twa_gate_utils.py tools/official/check_twa_gate_e_mechanism.py
pytest tests/test_twa_gate_e_mechanism.py -q
git diff --check
```

---

## 8. 修改 6：README 顶部状态块

建议在 README 顶部新增并置顶：

```markdown
## Current Official Status

Decision: `PROCEED_TWA_NO_BN_TO_GATE_E`

Strong anchor: `MSHNetOHEM`.

Active candidate:

- `TWA-OHEM without BN recalibration`
- seed42 Full Gate-TWA-C: PASS
- seed42 HC-Val Gate-TWA-D: PASS
- Gate-D deltas:
  - mIoU: `+0.02910`
  - FA ppm: `-58.4920`
  - Precision: `+0.03545`
  - Pd: `+0.00000`
- next allowed gate: `Gate-TWA-E`

Stopped or frozen branches:

- TWA with BN recalibration: stopped
- OHCM / prototype / full branch: stopped or legacy only
- SPS-OHEM reranking: not active for current TWA decision
- APF / component mining: stopped
- PFR / ERD / CGA-style trainable correction heads: stopped
- CDV / ECDV / MSCV / BCV verifier-style routes: stopped

Allowed now:

- Gate-TWA-E mechanism comparison on seed42 only
- Compare OHEM-400, best single late checkpoint, TCE-4, TWA-2, TWA-3, TWA-4 without BN
- Fixed threshold `0.5`
- Full and HC-Val only

Forbidden before Gate-TWA-E passes:

- seed43 / seed44
- HC-Test
- blind / external
- BN recalibration tuning
- new model training
- new verifier / suppression head
```

---

## 9. Gate-E 需要准备的 summary

Gate-E checker 需要这些 `summary_metrics.json`：

```text
OHEM-400 Full
OHEM-400 HC-Val
TWA-4 without BN Full
TWA-4 without BN HC-Val
TCE-4 HC-Val
single ep250 HC-Val
single ep300 HC-Val
single ep350 HC-Val
single ep400 HC-Val / OHEM-400 HC-Val
TWA-2 HC-Val
TWA-3 HC-Val
TWA-4 HC-Val
TWA-2 Full, optional report
TWA-3 Full, optional report
TWA-4 Full
```

TWA combinations固定为：

```text
TWA-2 = ep350 + ep400
TWA-3 = ep300 + ep350 + ep400
TWA-4 = ep250 + ep300 + ep350 + ep400
```

不要穷举其它 checkpoint combination。

---

## 10. Gate-E PASS/FAIL 解释

Gate-E PASS 条件：

```text
1. Gate-D summary 必须 gate_pass=true 且 next_allowed_gate=Gate-TWA-E
2. TWA-4 Full vs OHEM-400 不退化
3. TWA-4 HC-Val vs OHEM-400 达到 Gate-D 级别提升
4. TWA-4 HC-Val 不差于 best single late checkpoint
5. TWA-4 至少保留 TCE-4 hard-split 收益的 30%
6. TWA-2/TWA-3/TWA-4 trend 合理，至少 2/3 方向正向，且 TWA-4 不是 mIoU 最差
```

Gate-E FAIL 后：

```text
STOP_TWA_NO_BN_AT_GATE_E
```

不得跑：

```text
seed43/44
HC-Test
blind
external
BN recalibration tuning
new model training
new verifier / suppression head
```

Gate-E PASS 后才允许：

```text
Gate-TWA-F: seed43/44 paired Full + HC-Val for frozen TWA-4 without BN only
```

即使 Gate-E PASS，也仍然不允许 HC-Test、blind、external；这些必须等三种子、threshold-matched、FP component analysis 之后。

---

## 11. AAAI 角度的论文定位

现在可以写成：

```text
Training-trajectory weight averaging compresses the reliability signal of a multi-checkpoint trajectory into a single MSHNetOHEM-compatible checkpoint, preserving the single-forward inference path while reducing hard-clutter false alarms.
```

不要写成：

```text
We simply average checkpoints.
```

审稿风险：

1. SWA / model soup 相似性。
2. 如果 best single late checkpoint 已经更强，TWA 机制不成立。
3. 如果 TWA-2/3/4 trend 不稳定，会像偶然搜索。
4. 如果只 seed42 有效，不能投稿主结果。
5. 如果 threshold-matched 不成立，会被认为只是阈值/校准收益。

因此 Gate-E 的比较器是必须补的代码，不是可选分析。

---

## 12. 最小执行清单

```text
[1] 更新 gate_twa_d_summary.json：写入 Gate-D PASS 和 next_allowed_gate=Gate-TWA-E
[2] README 置顶 Current Official Status
[3] 新增 utils/twa_gate_utils.py
[4] 新增 tools/official/check_twa_gate_e_mechanism.py
[5] 新增 scripts/official/run_twa_gate_e_seed42.sh
[6] 新增 tests/test_twa_gate_e_mechanism.py
[7] 运行 py_compile
[8] 运行 pytest tests/test_twa_gate_e_mechanism.py -q
[9] 运行 git diff --check
[10] 准备 Gate-E summaries
[11] 运行 bash scripts/official/run_twa_gate_e_seed42.sh
```

当前正式决策：

```text
Decision: PROCEED_TWA_NO_BN_TO_GATE_E

Gate-D seed42 HC-Val PASS:
  mIoU      +0.02910
  FA ppm    -58.4920
  Precision +0.03545
  Pd        +0.00000

Next allowed gate:
  Gate-TWA-E mechanism comparison on seed42 only.

Still forbidden:
  seed43/44, HC-Test, blind/external, BN tuning, new training, new verifier/suppression structure.
```
