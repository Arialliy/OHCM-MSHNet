# OHCM-MSHNet：Gate-LS-A FAIL 后的局部修正方案与代码修改清单

> 场景：AAAI 投稿倒计时约 15 天。  
> 原则：**不推翻已通过部分，只处理刚刚暴露的失败点**。  
> 当前正式状态：`STOP_LATE_SNAPSHOT_EP250_AT_GATE_A`。  
> 本轮目标：不要救 `ep250`，不要改模型；只新增一个最小、冻结、no-training 的 **Full-safe single-late control gate**，判断 `Gate-E` 中击败 TWA-4 的 single checkpoint 是否必须先满足 Full split evidence anchor。

---

## 0. 当前结论

你现在的最新结果是：

```text
Gate-LS-A result: FAIL
formal decision: STOP_LATE_SNAPSHOT_EP250_AT_GATE_A

failed condition:
  snapshot_full_nonregression_vs_ohem = false

seed42 Full:
  ep250 mIoU      = 0.831139
  OHEM-400 mIoU   = 0.834393
  delta mIoU      = -0.003254
  delta Precision = -0.000187
  delta FA ppm    = -0.114901
  delta Pd        = +0.000000

seed42 HC-Val:
  ep250 vs OHEM HC-Val mIoU = +0.105858
  ep250 vs OHEM HC-Val FA   = -175.476 ppm
  ep250 vs TWA-4 HC-Val mIoU = +0.076757
  ep250 vs TWA-4 HC-Val FA   = -116.984 ppm
```

解释：

```text
ep250 是 hard-clutter specialist，
但不是 Full-safe candidate。
```

因此不要把 `Gate-LS-A` 改成通过，也不要用 tolerance 把 `-0.003254` 放过去。这个 mIoU 下降虽然小，但它已经违反了当前 pipeline 的 evidence anchor：

```text
Full split 不能退化。
```

---

## 1. 这不是全盘失败，而是 Gate-E 控制组定义需要补一层 eligibility

此前 `Gate-E` 失败的唯一关键条件是：

```text
TWA-4 HC-Val 不差于 best single late checkpoint: FAIL
best single late checkpoint = ep250
```

但现在 `Gate-LS-A` 证明：

```text
ep250 不能作为 active candidate，
因为它没有通过 Full non-regression。
```

所以合理的局部修正不是：

```text
重新跑 TWA
继续找 TWA-5
给 ep250 调阈值
降低 Full gate
跑 seed43/44 赌一把
```

而是新增一个很小的仲裁 gate：

```text
Gate-TWA-E2-FSC:
  Full-Safe Single-Late Control
```

这个 gate 只回答一个问题：

```text
在预注册 late checkpoints 中，
排除 Full-unsafe single checkpoint 后，
TWA-4 是否仍然输给一个 Full-safe single checkpoint？
```

这不是推翻 Gate-E，而是给 Gate-E 的 best-single control 补上候选资格约束：

```text
best single late checkpoint
必须先是 Full-safe single late checkpoint。
```

---

## 2. 为什么下一步不能继续救 ep250？

### 2.1 ep250 已经正式停止

`Gate-LS-A` 的失败原因非常直接：

```text
snapshot_full_nonregression_vs_ohem = false
```

这说明 ep250 的 hard-clutter 强信号是以 Full split mIoU/Precision 轻微退化为代价获得的。

如果现在继续做以下操作：

```text
threshold search for ep250
accept small mIoU drop
only report HC-Val
run seed43/44 despite Full fail
run HC-Test/blind/external despite Full fail
```

在审稿角度会非常像 post-hoc rescue。当前最稳的做法是：

```text
保留 ep250 为 stopped diagnostic control，
不再把它作为候选推进。
```

### 2.2 ep250 的价值仍然保留

ep250 不是无用结果。它现在说明：

```text
1. OHEM trajectory 中确实存在 hard-clutter-friendly snapshot；
2. TWA-4 的 average 把 ep250 的 HC-Val 优势稀释了；
3. 单个 checkpoint 的 hard-clutter 优势可能与 Full split 保真性冲突。
```

这可以写进论文的 ablation / diagnostic，但不能作为主方法候选。

---

## 3. 当前不应修改的部分

这些文件和结果已经通过，不要重写：

```text
utils/twa_gate_utils.py
tools/official/check_twa_gate_e_mechanism.py
scripts/official/run_twa_gate_e_seed42.sh
tests/test_twa_gate_e_mechanism.py
tools/official/check_late_snapshot_gate_a_seed42.py
scripts/official/run_late_snapshot_ep250_gate_a_seed42.sh
tests/test_late_snapshot_gate_a_seed42.py

docs/internal/twa/seed42_nudt/gate_twa_d_summary.json
docs/internal/twa/seed42_nudt/gate_twa_e_summary.json
docs/internal/twa/seed42_nudt/gate_late_snapshot_ep250_a_summary.json
```

不要改：

```text
model/
net.py
loss.py
train.py 主训练逻辑
dataset.py
probability.py 的前景概率定义
```

原因：现在的问题不是网络结构、loss 或概率定义问题，而是 checkpoint trajectory 里的 candidate eligibility 问题。

---

## 4. 下一步唯一建议：Gate-TWA-E2-FSC

### 4.1 Gate 名称

```text
Gate-TWA-E2-FSC
```

完整名：

```text
Gate-TWA-E2: Full-Safe Single-Late Control
```

### 4.2 Gate 输入

只允许使用已预注册 late trajectory checkpoint：

```text
250, 300, 350, 400
```

其中：

```text
ep250: stopped diagnostic control, 已有 Full + HC-Val
ep300: 如果缺 Full/HC-Val summary，只补 seed42 fixed-threshold evaluation
ep350: 如果缺 Full/HC-Val summary，只补 seed42 fixed-threshold evaluation
ep400: OHEM-400 baseline/control，不作为新候选
TWA-4: 250/300/350/400 no-BN average，已通过 Full + HC-Val vs OHEM
```

### 4.3 Gate 不允许

```text
不允许 ep200 / ep225 / ep275 / ep325 / ep375
不允许 weighted checkpoint average
不允许重新构造 TWA combination
不允许 threshold search
不允许 seed43/44
不允许 HC-Test / blind / external
不允许 BN tuning
不允许新训练
不允许改模型/loss/verifier/suppression head
```

### 4.4 Gate 先筛 single checkpoint eligibility

对每个 single checkpoint `ep in {250,300,350}`，先看 Full：

| Full 指标 | 条件 |
|---|---:|
| `delta_mIoU` | `>= 0.0` |
| `delta_Precision` | `>= 0.0` |
| `delta_Pd` | `>= 0.0` |
| `delta_FA_ppm` | `<= 0.0` |

再看 HC-Val：

| HC-Val 指标 | 条件 |
|---|---:|
| `delta_mIoU` | `>= +0.005` |
| `delta_FA_ppm` | `<= -10.0` |
| `delta_Precision` | `>= 0.0` |
| `delta_Pd` | `>= 0.0` |

只有同时满足 Full-safe 和 HC-positive 的 single checkpoint，才允许作为 `eligible_single_late_checkpoint`。

### 4.5 Gate 决策规则

得到 `eligible_single_late_checkpoint` 集合后：

#### 情况 A：没有任何 Full-safe single checkpoint 同时 HC-Val 正向

如果 TWA-4 已经满足：

```text
TWA-4 Full vs OHEM: non-regression PASS
TWA-4 HC-Val vs OHEM: positive PASS
```

则写入：

```text
Gate-TWA-E2-FSC: PASS
selected_candidate: TWA-4-noBN
reason: ep250 was the best HC-Val single but is Full-unsafe; no Full-safe single beats TWA-4.
next_allowed_gate: Gate-TWA-F seed43/44 paired Full + HC-Val for TWA-4 only
```

#### 情况 B：存在 Full-safe single checkpoint，但其 HC-Val mIoU 不高于 TWA-4

写入：

```text
Gate-TWA-E2-FSC: PASS
selected_candidate: TWA-4-noBN
reason: TWA-4 is not worse than the best Full-safe single-late control.
next_allowed_gate: Gate-TWA-F seed43/44 paired Full + HC-Val for TWA-4 only
```

#### 情况 C：存在 Full-safe single checkpoint，且它 HC-Val 明显高于 TWA-4

写入：

```text
Gate-TWA-E2-FSC: PASS_WITH_SWITCH
selected_candidate: LateSnapshot-epXXX
reason: a Full-safe single-late checkpoint dominates TWA-4 on HC-Val.
next_allowed_gate: Gate-LS-B seed43/44 paired Full + HC-Val for epXXX only
```

这里 `epXXX` 必须是 `300` 或 `350`，因为 `ep250` 已停止，`ep400` 是 baseline。

#### 情况 D：TWA-4 不满足已有 Full/HC-Val 正向，且没有 eligible single

写入：

```text
Gate-TWA-E2-FSC: FAIL
selected_candidate: none
decision: STOP_ALL_SINGLE_FORWARD_TRAJECTORY_COMPRESSION
```

---

## 5. 新增文件 1：`tools/official/check_twa_gate_e2_fullsafe_single_control.py`

新增文件：

```text
tools/official/check_twa_gate_e2_fullsafe_single_control.py
```

### 5.1 完整代码

```python
#!/usr/bin/env python3
"""
Gate-TWA-E2-FSC: Full-Safe Single-Late Control.

This checker does not train models and does not tune thresholds.
It only arbitrates among pre-registered late checkpoints after ep250
has been shown to be Full-unsafe.

Allowed checkpoints: 250, 300, 350, 400.
Candidate checkpoints: 250, 300, 350.
OHEM-400 is treated as baseline/control, not as a new candidate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_ALIASES = {
    "mIoU": ("mIoU", "miou", "mean_iou", "mean_IoU"),
    "Precision": ("Precision", "precision", "prec"),
    "Pd": ("Pd", "pd", "PD", "recall"),
    "FA_ppm": ("FA_ppm", "FA", "fa", "fa_ppm", "FAppm"),
}

ALLOWED_EPOCHS = {250, 300, 350, 400}
CANDIDATE_EPOCHS = {250, 300, 350}
BASELINE_EPOCH = 400


@dataclass(frozen=True)
class Metrics:
    mIoU: float
    Precision: float
    Pd: float
    FA_ppm: float


@dataclass(frozen=True)
class DeltaMetrics:
    mIoU: float
    Precision: float
    Pd: float
    FA_ppm: float


@dataclass(frozen=True)
class SnapshotRecord:
    epoch: int
    full: Metrics
    hcval: Metrics
    full_delta_vs_ohem: DeltaMetrics
    hcval_delta_vs_ohem: DeltaMetrics
    full_safe: bool
    hcval_positive: bool
    eligible_single: bool
    ineligible_reasons: Tuple[str, ...]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_nested(summary: Dict[str, Any], key: str) -> Optional[Any]:
    if key in summary:
        return summary[key]
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    official_metrics = summary.get("official_metrics")
    if isinstance(official_metrics, dict) and key in official_metrics:
        return official_metrics[key]
    return None


def get_metric(summary: Dict[str, Any], canonical_key: str) -> float:
    aliases = METRIC_ALIASES[canonical_key]
    for key in aliases:
        value = _lookup_nested(summary, key)
        if value is not None:
            return float(value)
    available = sorted(list(summary.keys()))
    if isinstance(summary.get("metrics"), dict):
        available.extend(f"metrics.{k}" for k in summary["metrics"].keys())
    raise KeyError(f"Metric {canonical_key} not found. Available keys: {available}")


def metrics_from_summary(summary: Dict[str, Any]) -> Metrics:
    return Metrics(
        mIoU=get_metric(summary, "mIoU"),
        Precision=get_metric(summary, "Precision"),
        Pd=get_metric(summary, "Pd"),
        FA_ppm=get_metric(summary, "FA_ppm"),
    )


def delta(candidate: Metrics, baseline: Metrics) -> DeltaMetrics:
    return DeltaMetrics(
        mIoU=candidate.mIoU - baseline.mIoU,
        Precision=candidate.Precision - baseline.Precision,
        Pd=candidate.Pd - baseline.Pd,
        FA_ppm=candidate.FA_ppm - baseline.FA_ppm,
    )


def check_full_safe(
    d: DeltaMetrics,
    *,
    min_delta_miou: float,
    min_delta_precision: float,
    min_delta_pd: float,
    max_delta_fa_ppm: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if d.mIoU < min_delta_miou:
        reasons.append("full_miou_regression")
    if d.Precision < min_delta_precision:
        reasons.append("full_precision_regression")
    if d.Pd < min_delta_pd:
        reasons.append("full_pd_regression")
    if d.FA_ppm > max_delta_fa_ppm:
        reasons.append("full_fa_regression")
    return len(reasons) == 0, reasons


def check_hcval_positive(
    d: DeltaMetrics,
    *,
    min_delta_miou: float,
    min_fa_reduction_ppm: float,
    min_delta_precision: float,
    min_delta_pd: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if d.mIoU < min_delta_miou:
        reasons.append("hcval_miou_not_enough")
    if d.FA_ppm > -min_fa_reduction_ppm:
        reasons.append("hcval_fa_reduction_not_enough")
    if d.Precision < min_delta_precision:
        reasons.append("hcval_precision_regression")
    if d.Pd < min_delta_pd:
        reasons.append("hcval_pd_regression")
    return len(reasons) == 0, reasons


def build_snapshot_record(
    *,
    epoch: int,
    full_summary: Dict[str, Any],
    hcval_summary: Dict[str, Any],
    ohem_full: Metrics,
    ohem_hcval: Metrics,
    min_full_delta_miou: float,
    min_full_delta_precision: float,
    min_full_delta_pd: float,
    max_full_delta_fa_ppm: float,
    min_hcval_delta_miou: float,
    min_hcval_fa_reduction_ppm: float,
    min_hcval_delta_precision: float,
    min_hcval_delta_pd: float,
) -> SnapshotRecord:
    if epoch not in ALLOWED_EPOCHS:
        raise ValueError(f"Epoch {epoch} is not allowed. Allowed: {sorted(ALLOWED_EPOCHS)}")

    full = metrics_from_summary(full_summary)
    hcval = metrics_from_summary(hcval_summary)
    full_delta = delta(full, ohem_full)
    hcval_delta = delta(hcval, ohem_hcval)

    full_safe, full_reasons = check_full_safe(
        full_delta,
        min_delta_miou=min_full_delta_miou,
        min_delta_precision=min_full_delta_precision,
        min_delta_pd=min_full_delta_pd,
        max_delta_fa_ppm=max_full_delta_fa_ppm,
    )
    hcval_positive, hc_reasons = check_hcval_positive(
        hcval_delta,
        min_delta_miou=min_hcval_delta_miou,
        min_fa_reduction_ppm=min_hcval_fa_reduction_ppm,
        min_delta_precision=min_hcval_delta_precision,
        min_delta_pd=min_hcval_delta_pd,
    )

    reasons: List[str] = []
    if epoch == BASELINE_EPOCH:
        reasons.append("baseline_epoch_400_not_candidate")
    if epoch not in CANDIDATE_EPOCHS:
        reasons.append("not_candidate_epoch")
    reasons.extend(full_reasons)
    reasons.extend(hc_reasons)

    eligible = (
        epoch in CANDIDATE_EPOCHS
        and full_safe
        and hcval_positive
    )

    return SnapshotRecord(
        epoch=epoch,
        full=full,
        hcval=hcval,
        full_delta_vs_ohem=full_delta,
        hcval_delta_vs_ohem=hcval_delta,
        full_safe=full_safe,
        hcval_positive=hcval_positive,
        eligible_single=eligible,
        ineligible_reasons=tuple(reasons),
    )


def parse_snapshot_arg(raw: str) -> Tuple[int, Path, Path]:
    """Parse EPOCH:FULL_SUMMARY:HCVAL_SUMMARY."""
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--snapshot must have format EPOCH:FULL_SUMMARY:HCVAL_SUMMARY"
        )
    epoch = int(parts[0])
    return epoch, Path(parts[1]), Path(parts[2])


def choose_best_eligible(records: Sequence[SnapshotRecord]) -> Optional[SnapshotRecord]:
    eligible = [r for r in records if r.eligible_single]
    if not eligible:
        return None
    # Primary: HC-Val mIoU. Secondary: larger FA reduction. Tertiary: higher Precision.
    return max(
        eligible,
        key=lambda r: (
            r.hcval_delta_vs_ohem.mIoU,
            -r.hcval_delta_vs_ohem.FA_ppm,
            r.hcval_delta_vs_ohem.Precision,
            r.hcval_delta_vs_ohem.Pd,
        ),
    )


def validate_context(
    gate_e_summary: Dict[str, Any],
    ep250_gate_a_summary: Dict[str, Any],
) -> Dict[str, Any]:
    gate_e_pass = bool(gate_e_summary.get("gate_pass", False))
    ep250_gate_a_pass = bool(ep250_gate_a_summary.get("gate_pass", False))

    context = {
        "gate_e_pass": gate_e_pass,
        "gate_e_decision": gate_e_summary.get("decision") or gate_e_summary.get("status"),
        "gate_e_failed_conditions": gate_e_summary.get("failed_conditions")
        or gate_e_summary.get("fail_reasons")
        or gate_e_summary.get("failure_reasons"),
        "ep250_gate_a_pass": ep250_gate_a_pass,
        "ep250_gate_a_decision": ep250_gate_a_summary.get("decision")
        or ep250_gate_a_summary.get("status"),
        "ep250_gate_a_failed_conditions": ep250_gate_a_summary.get("failed_conditions")
        or ep250_gate_a_summary.get("fail_reasons")
        or ep250_gate_a_summary.get("failure_reasons"),
    }

    # This gate is intended only after Gate-E failed and ep250 Gate-A failed.
    context["context_valid"] = (not gate_e_pass) and (not ep250_gate_a_pass)
    return context


def decide(
    *,
    twa_full: Metrics,
    twa_hcval: Metrics,
    ohem_full: Metrics,
    ohem_hcval: Metrics,
    best_eligible: Optional[SnapshotRecord],
    min_full_delta_miou: float,
    min_full_delta_precision: float,
    min_full_delta_pd: float,
    max_full_delta_fa_ppm: float,
    min_hcval_delta_miou: float,
    min_hcval_fa_reduction_ppm: float,
    min_hcval_delta_precision: float,
    min_hcval_delta_pd: float,
    twa_vs_single_eps: float,
) -> Dict[str, Any]:
    twa_full_delta = delta(twa_full, ohem_full)
    twa_hc_delta = delta(twa_hcval, ohem_hcval)

    twa_full_safe, twa_full_reasons = check_full_safe(
        twa_full_delta,
        min_delta_miou=min_full_delta_miou,
        min_delta_precision=min_full_delta_precision,
        min_delta_pd=min_full_delta_pd,
        max_delta_fa_ppm=max_full_delta_fa_ppm,
    )
    twa_hc_positive, twa_hc_reasons = check_hcval_positive(
        twa_hc_delta,
        min_delta_miou=min_hcval_delta_miou,
        min_fa_reduction_ppm=min_hcval_fa_reduction_ppm,
        min_delta_precision=min_hcval_delta_precision,
        min_delta_pd=min_hcval_delta_pd,
    )

    twa_eligible = twa_full_safe and twa_hc_positive

    result: Dict[str, Any] = {
        "twa4": {
            "full": asdict(twa_full),
            "hcval": asdict(twa_hcval),
            "full_delta_vs_ohem": asdict(twa_full_delta),
            "hcval_delta_vs_ohem": asdict(twa_hc_delta),
            "full_safe": twa_full_safe,
            "hcval_positive": twa_hc_positive,
            "eligible": twa_eligible,
            "ineligible_reasons": twa_full_reasons + twa_hc_reasons,
        }
    }

    if not twa_eligible and best_eligible is None:
        result.update(
            {
                "gate_pass": False,
                "decision": "STOP_ALL_SINGLE_FORWARD_TRAJECTORY_COMPRESSION",
                "selected_candidate": None,
                "next_allowed_gate": None,
                "reason": "Neither TWA-4 nor any pre-registered single-late checkpoint is Full-safe and HC-Val positive.",
            }
        )
        return result

    if best_eligible is None:
        result.update(
            {
                "gate_pass": True,
                "decision": "REOPEN_TWA4_TO_GATE_F_SEED43_44",
                "selected_candidate": "TWA-4-noBN",
                "next_allowed_gate": "Gate-TWA-F-seed43-44-Full-HCVal",
                "reason": "No Full-safe HC-positive single-late checkpoint exists; ep250 is HC-strong but Full-unsafe.",
            }
        )
        return result

    best_hc = best_eligible.hcval_delta_vs_ohem.mIoU
    twa_hc = twa_hc_delta.mIoU
    twa_not_worse_than_best_fullsafe_single = twa_eligible and (
        twa_hc + twa_vs_single_eps >= best_hc
    )

    result["best_eligible_single"] = {
        "epoch": best_eligible.epoch,
        "hcval_delta_miou": best_eligible.hcval_delta_vs_ohem.mIoU,
        "hcval_delta_fa_ppm": best_eligible.hcval_delta_vs_ohem.FA_ppm,
        "full_delta_miou": best_eligible.full_delta_vs_ohem.mIoU,
    }
    result["twa4_not_worse_than_best_fullsafe_single_hcval"] = (
        twa_not_worse_than_best_fullsafe_single
    )

    if twa_not_worse_than_best_fullsafe_single:
        result.update(
            {
                "gate_pass": True,
                "decision": "REOPEN_TWA4_TO_GATE_F_SEED43_44",
                "selected_candidate": "TWA-4-noBN",
                "next_allowed_gate": "Gate-TWA-F-seed43-44-Full-HCVal",
                "reason": "TWA-4 is not worse than the best Full-safe single-late control on HC-Val.",
            }
        )
    else:
        result.update(
            {
                "gate_pass": True,
                "decision": f"PROCEED_FULLSAFE_SINGLE_EP{best_eligible.epoch}_TO_GATE_LS_B_SEED43_44",
                "selected_candidate": f"LateSnapshot-ep{best_eligible.epoch}",
                "next_allowed_gate": f"Gate-LS-B-ep{best_eligible.epoch}-seed43-44-Full-HCVal",
                "reason": "A Full-safe single-late checkpoint dominates TWA-4 on HC-Val.",
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate_e_summary", required=True, type=Path)
    parser.add_argument("--ep250_gate_a_summary", required=True, type=Path)
    parser.add_argument("--ohem_full", required=True, type=Path)
    parser.add_argument("--ohem_hcval", required=True, type=Path)
    parser.add_argument("--twa_full", required=True, type=Path)
    parser.add_argument("--twa_hcval", required=True, type=Path)
    parser.add_argument(
        "--snapshot",
        action="append",
        required=True,
        help="Format: EPOCH:FULL_SUMMARY:HCVAL_SUMMARY. Allowed epochs: 250,300,350,400.",
    )
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--min_full_delta_miou", type=float, default=0.0)
    parser.add_argument("--min_full_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_full_delta_pd", type=float, default=0.0)
    parser.add_argument("--max_full_delta_fa_ppm", type=float, default=0.0)

    parser.add_argument("--min_hcval_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_hcval_fa_reduction_ppm", type=float, default=10.0)
    parser.add_argument("--min_hcval_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_hcval_delta_pd", type=float, default=0.0)
    parser.add_argument(
        "--twa_vs_single_eps",
        type=float,
        default=1e-12,
        help="Numerical tolerance only. Do not use this to accept real metric drops.",
    )
    args = parser.parse_args()

    gate_e_summary = load_json(args.gate_e_summary)
    ep250_gate_a_summary = load_json(args.ep250_gate_a_summary)
    context = validate_context(gate_e_summary, ep250_gate_a_summary)

    ohem_full = metrics_from_summary(load_json(args.ohem_full))
    ohem_hcval = metrics_from_summary(load_json(args.ohem_hcval))
    twa_full = metrics_from_summary(load_json(args.twa_full))
    twa_hcval = metrics_from_summary(load_json(args.twa_hcval))

    records: List[SnapshotRecord] = []
    seen_epochs = set()
    for raw in args.snapshot:
        epoch, full_path, hcval_path = parse_snapshot_arg(raw)
        if epoch in seen_epochs:
            raise ValueError(f"Duplicate snapshot epoch: {epoch}")
        seen_epochs.add(epoch)
        record = build_snapshot_record(
            epoch=epoch,
            full_summary=load_json(full_path),
            hcval_summary=load_json(hcval_path),
            ohem_full=ohem_full,
            ohem_hcval=ohem_hcval,
            min_full_delta_miou=args.min_full_delta_miou,
            min_full_delta_precision=args.min_full_delta_precision,
            min_full_delta_pd=args.min_full_delta_pd,
            max_full_delta_fa_ppm=args.max_full_delta_fa_ppm,
            min_hcval_delta_miou=args.min_hcval_delta_miou,
            min_hcval_fa_reduction_ppm=args.min_hcval_fa_reduction_ppm,
            min_hcval_delta_precision=args.min_hcval_delta_precision,
            min_hcval_delta_pd=args.min_hcval_delta_pd,
        )
        records.append(record)

    missing = CANDIDATE_EPOCHS.difference(seen_epochs)
    if missing:
        raise ValueError(
            f"Missing candidate epochs {sorted(missing)}. "
            "Provide all of 250,300,350 to avoid post-hoc omission."
        )

    best_eligible = choose_best_eligible(records)
    decision = decide(
        twa_full=twa_full,
        twa_hcval=twa_hcval,
        ohem_full=ohem_full,
        ohem_hcval=ohem_hcval,
        best_eligible=best_eligible,
        min_full_delta_miou=args.min_full_delta_miou,
        min_full_delta_precision=args.min_full_delta_precision,
        min_full_delta_pd=args.min_full_delta_pd,
        max_full_delta_fa_ppm=args.max_full_delta_fa_ppm,
        min_hcval_delta_miou=args.min_hcval_delta_miou,
        min_hcval_fa_reduction_ppm=args.min_hcval_fa_reduction_ppm,
        min_hcval_delta_precision=args.min_hcval_delta_precision,
        min_hcval_delta_pd=args.min_hcval_delta_pd,
        twa_vs_single_eps=args.twa_vs_single_eps,
    )

    output = {
        "gate": "Gate-TWA-E2-FSC",
        "gate_name": "Full-Safe Single-Late Control",
        "seed": 42,
        "split_scope": ["Full", "HC-Val"],
        "threshold": 0.5,
        "context": context,
        "allowed_epochs": sorted(ALLOWED_EPOCHS),
        "candidate_epochs": sorted(CANDIDATE_EPOCHS),
        "baseline_epoch": BASELINE_EPOCH,
        "snapshot_records": [asdict(r) for r in records],
        "best_eligible_single_epoch": best_eligible.epoch if best_eligible else None,
        **decision,
        "forbidden_next_actions": [
            "seed43_or_seed44_before_this_gate_passes",
            "HC-Test",
            "blind",
            "external",
            "threshold_search",
            "BN_tuning",
            "new_training",
            "new_model_structure",
            "new_loss",
            "new_verifier_or_suppression_head",
            "new_checkpoint_epochs_outside_250_300_350_400",
            "new_TWA_combinations",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if not output["gate_pass"]:
        raise SystemExit(output["decision"])


if __name__ == "__main__":
    main()
```

---

## 6. 新增文件 2：`scripts/official/run_twa_gate_e2_fullsafe_single_control_seed42.sh`

新增：

```text
scripts/official/run_twa_gate_e2_fullsafe_single_control_seed42.sh
```

### 6.1 目的

只补齐 seed42 下 `ep300` / `ep350` 的 Full 与 HC-Val summary。已经存在的 summary 不重复跑。

### 6.2 完整脚本

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SEED=42
DATASET=${DATASET:-NUDT-SIRST}
THRESHOLD=${THRESHOLD:-0.5}

BASE_DIR="${ROOT}/docs/internal/twa/seed42_nudt"
RESULT_DIR="${ROOT}/results/official/MSHNetOHEM/seed42/NUDT-SIRST"

# Existing summaries. Override these env vars if your local paths differ.
OHEM_FULL_SUMMARY=${OHEM_FULL_SUMMARY:-"${BASE_DIR}/eval_full_ohem/summary_metrics.json"}
OHEM_HCVAL_SUMMARY=${OHEM_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_ohem/summary_metrics.json"}
TWA_FULL_SUMMARY=${TWA_FULL_SUMMARY:-"${BASE_DIR}/eval_full_twa_no_bn/summary_metrics.json"}
TWA_HCVAL_SUMMARY=${TWA_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_twa_no_bn/summary_metrics.json"}
GATE_E_SUMMARY=${GATE_E_SUMMARY:-"${BASE_DIR}/gate_twa_e_summary.json"}
EP250_GATE_A_SUMMARY=${EP250_GATE_A_SUMMARY:-"${BASE_DIR}/gate_late_snapshot_ep250_a_summary.json"}

# ep250 Full summary was produced by Gate-LS-A.
EP250_FULL_SUMMARY=${EP250_FULL_SUMMARY:-"${BASE_DIR}/eval_full_single_ep250/summary_metrics.json"}
# ep250 HC-Val summary should already exist from Gate-E single-late comparison.
EP250_HCVAL_SUMMARY=${EP250_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_single_ep250/summary_metrics.json"}

# ep300/ep350 summaries are the only ones this script may create if missing.
EP300_FULL_SUMMARY=${EP300_FULL_SUMMARY:-"${BASE_DIR}/eval_full_single_ep300/summary_metrics.json"}
EP300_HCVAL_SUMMARY=${EP300_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_single_ep300/summary_metrics.json"}
EP350_FULL_SUMMARY=${EP350_FULL_SUMMARY:-"${BASE_DIR}/eval_full_single_ep350/summary_metrics.json"}
EP350_HCVAL_SUMMARY=${EP350_HCVAL_SUMMARY:-"${BASE_DIR}/eval_hcval_single_ep350/summary_metrics.json"}

OUTPUT=${OUTPUT:-"${BASE_DIR}/gate_twa_e2_fullsafe_single_control_summary.json"}

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "Missing required file: ${p}" >&2
    exit 2
  fi
}

eval_single_if_missing() {
  local epoch="$1"
  local split="$2"
  local summary_path="$3"
  local out_dir
  out_dir="$(dirname "${summary_path}")"

  if [[ -f "${summary_path}" ]]; then
    echo "[skip] ${summary_path} already exists"
    return 0
  fi

  local ckpt="${RESULT_DIR}/MSHNetOHEM_${epoch}.pth.tar"
  require_file "${ckpt}"

  echo "[eval] ep${epoch} split=${split} -> ${out_dir}"
  python "${ROOT}/tools/official/evaluate_twa_checkpoint.py" \
    --model_name MSHNetOHEM \
    --checkpoint "${ckpt}" \
    --dataset_name "${DATASET}" \
    --split "${split}" \
    --output_dir "${out_dir}" \
    --threshold "${THRESHOLD}"

  require_file "${summary_path}"
}

# Required context files. Do not silently regenerate these.
require_file "${OHEM_FULL_SUMMARY}"
require_file "${OHEM_HCVAL_SUMMARY}"
require_file "${TWA_FULL_SUMMARY}"
require_file "${TWA_HCVAL_SUMMARY}"
require_file "${GATE_E_SUMMARY}"
require_file "${EP250_GATE_A_SUMMARY}"
require_file "${EP250_FULL_SUMMARY}"
require_file "${EP250_HCVAL_SUMMARY}"

# Only fill missing pre-registered single checkpoints.
eval_single_if_missing 300 full "${EP300_FULL_SUMMARY}"
eval_single_if_missing 300 hcval "${EP300_HCVAL_SUMMARY}"
eval_single_if_missing 350 full "${EP350_FULL_SUMMARY}"
eval_single_if_missing 350 hcval "${EP350_HCVAL_SUMMARY}"

python "${ROOT}/tools/official/check_twa_gate_e2_fullsafe_single_control.py" \
  --gate_e_summary "${GATE_E_SUMMARY}" \
  --ep250_gate_a_summary "${EP250_GATE_A_SUMMARY}" \
  --ohem_full "${OHEM_FULL_SUMMARY}" \
  --ohem_hcval "${OHEM_HCVAL_SUMMARY}" \
  --twa_full "${TWA_FULL_SUMMARY}" \
  --twa_hcval "${TWA_HCVAL_SUMMARY}" \
  --snapshot "250:${EP250_FULL_SUMMARY}:${EP250_HCVAL_SUMMARY}" \
  --snapshot "300:${EP300_FULL_SUMMARY}:${EP300_HCVAL_SUMMARY}" \
  --snapshot "350:${EP350_FULL_SUMMARY}:${EP350_HCVAL_SUMMARY}" \
  --output "${OUTPUT}"

echo "Wrote ${OUTPUT}"
```

### 6.3 注意路径

你本地已有：

```text
docs/internal/twa/seed42_nudt/eval_full_single_ep250/summary_metrics.json
docs/internal/twa/seed42_nudt/gate_late_snapshot_ep250_a_summary.json
```

`ep250 HC-Val` 的路径可能不是：

```text
docs/internal/twa/seed42_nudt/eval_hcval_single_ep250/summary_metrics.json
```

如果实际路径不同，不要改代码逻辑，运行时覆盖环境变量即可：

```bash
EP250_HCVAL_SUMMARY=/actual/path/to/ep250_hcval/summary_metrics.json \
bash scripts/official/run_twa_gate_e2_fullsafe_single_control_seed42.sh
```

---

## 7. 新增文件 3：`tests/test_twa_gate_e2_fullsafe_single_control.py`

新增：

```text
tests/test_twa_gate_e2_fullsafe_single_control.py
```

### 7.1 测试代码

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.official.check_twa_gate_e2_fullsafe_single_control import (
    build_snapshot_record,
    choose_best_eligible,
    delta,
    metrics_from_summary,
)


def summary(miou, precision, pd, fa_ppm):
    return {
        "metrics": {
            "mIoU": miou,
            "Precision": precision,
            "Pd": pd,
            "FA_ppm": fa_ppm,
        }
    }


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


def test_cli_reopens_twa_when_no_fullsafe_single_beats_it(tmp_path):
    tool = Path("tools/official/check_twa_gate_e2_fullsafe_single_control.py")

    ohem_full = tmp_path / "ohem_full.json"
    ohem_hc = tmp_path / "ohem_hc.json"
    twa_full = tmp_path / "twa_full.json"
    twa_hc = tmp_path / "twa_hc.json"
    gate_e = tmp_path / "gate_e.json"
    ep250_gate = tmp_path / "ep250_gate.json"
    output = tmp_path / "out.json"

    write_json(ohem_full, summary(0.834393, 0.900000, 0.980000, 63.449))
    write_json(ohem_hc, summary(0.604790, 0.700000, 0.970000, 250.0))
    write_json(twa_full, summary(0.838893, 0.903450, 0.981060, 61.082))
    write_json(twa_hc, summary(0.633891, 0.735450, 0.970000, 191.508))
    write_json(gate_e, {"gate_pass": False, "decision": "STOP_TWA_NO_BN_AT_GATE_E"})
    write_json(ep250_gate, {"gate_pass": False, "decision": "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"})

    ep250_full = tmp_path / "ep250_full.json"
    ep250_hc = tmp_path / "ep250_hc.json"
    ep300_full = tmp_path / "ep300_full.json"
    ep300_hc = tmp_path / "ep300_hc.json"
    ep350_full = tmp_path / "ep350_full.json"
    ep350_hc = tmp_path / "ep350_hc.json"

    write_json(ep250_full, summary(0.831139, 0.899813, 0.980000, 63.334))
    write_json(ep250_hc, summary(0.710648, 0.821241, 0.970000, 74.524))
    # ep300/ep350 are Full-safe but weaker than TWA-4 on HC-Val.
    write_json(ep300_full, summary(0.835000, 0.901000, 0.980000, 62.0))
    write_json(ep300_hc, summary(0.620000, 0.720000, 0.970000, 220.0))
    write_json(ep350_full, summary(0.836000, 0.902000, 0.980000, 62.0))
    write_json(ep350_hc, summary(0.625000, 0.725000, 0.970000, 215.0))

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--gate_e_summary", str(gate_e),
            "--ep250_gate_a_summary", str(ep250_gate),
            "--ohem_full", str(ohem_full),
            "--ohem_hcval", str(ohem_hc),
            "--twa_full", str(twa_full),
            "--twa_hcval", str(twa_hc),
            "--snapshot", f"250:{ep250_full}:{ep250_hc}",
            "--snapshot", f"300:{ep300_full}:{ep300_hc}",
            "--snapshot", f"350:{ep350_full}:{ep350_hc}",
            "--output", str(output),
        ],
        check=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["gate_pass"] is True
    assert result["selected_candidate"] == "TWA-4-noBN"
    assert result["next_allowed_gate"] == "Gate-TWA-F-seed43-44-Full-HCVal"


def test_cli_switches_to_fullsafe_single_when_it_beats_twa(tmp_path):
    tool = Path("tools/official/check_twa_gate_e2_fullsafe_single_control.py")

    ohem_full = tmp_path / "ohem_full.json"
    ohem_hc = tmp_path / "ohem_hc.json"
    twa_full = tmp_path / "twa_full.json"
    twa_hc = tmp_path / "twa_hc.json"
    gate_e = tmp_path / "gate_e.json"
    ep250_gate = tmp_path / "ep250_gate.json"
    output = tmp_path / "out.json"

    write_json(ohem_full, summary(0.834393, 0.900000, 0.980000, 63.449))
    write_json(ohem_hc, summary(0.604790, 0.700000, 0.970000, 250.0))
    write_json(twa_full, summary(0.838893, 0.903450, 0.981060, 61.082))
    write_json(twa_hc, summary(0.633891, 0.735450, 0.970000, 191.508))
    write_json(gate_e, {"gate_pass": False})
    write_json(ep250_gate, {"gate_pass": False})

    ep250_full = tmp_path / "ep250_full.json"
    ep250_hc = tmp_path / "ep250_hc.json"
    ep300_full = tmp_path / "ep300_full.json"
    ep300_hc = tmp_path / "ep300_hc.json"
    ep350_full = tmp_path / "ep350_full.json"
    ep350_hc = tmp_path / "ep350_hc.json"

    write_json(ep250_full, summary(0.831139, 0.899813, 0.980000, 63.334))
    write_json(ep250_hc, summary(0.710648, 0.821241, 0.970000, 74.524))
    # ep300 is Full-safe and beats TWA-4 on HC-Val.
    write_json(ep300_full, summary(0.835000, 0.901000, 0.980000, 62.0))
    write_json(ep300_hc, summary(0.670000, 0.780000, 0.970000, 180.0))
    write_json(ep350_full, summary(0.836000, 0.902000, 0.980000, 62.0))
    write_json(ep350_hc, summary(0.625000, 0.725000, 0.970000, 215.0))

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--gate_e_summary", str(gate_e),
            "--ep250_gate_a_summary", str(ep250_gate),
            "--ohem_full", str(ohem_full),
            "--ohem_hcval", str(ohem_hc),
            "--twa_full", str(twa_full),
            "--twa_hcval", str(twa_hc),
            "--snapshot", f"250:{ep250_full}:{ep250_hc}",
            "--snapshot", f"300:{ep300_full}:{ep300_hc}",
            "--snapshot", f"350:{ep350_full}:{ep350_hc}",
            "--output", str(output),
        ],
        check=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["gate_pass"] is True
    assert result["selected_candidate"] == "LateSnapshot-ep300"
    assert result["next_allowed_gate"] == "Gate-LS-B-ep300-seed43-44-Full-HCVal"
```

---

## 8. README 局部更新

把 README 顶部 `Current Official Status` 改成下面这种状态。不要删除旧分支，只新增当前最新状态：

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Stopped branches:
- OHCM / prototype / full branch
- SPS-OHEM pixel reranking
- APF / component mining
- PFR / ERD / CGA trainable suppression-style branches
- CDV / ECDV / MSCV / BCV
- TWA with BN recalibration
- TWA-4 no-BN as originally judged by Gate-E against unrestricted best single late checkpoint
- LateSnapshot-ep250, stopped at Gate-LS-A because seed42 Full non-regression failed

Current active audit:
- Gate-TWA-E2-FSC: Full-Safe Single-Late Control on seed42 only
- Purpose: re-check the Gate-E single-checkpoint control after ep250 was proven Full-unsafe
- Allowed single checkpoints: ep250 / ep300 / ep350 as candidates, ep400 as OHEM baseline/control
- Allowed evaluations: seed42 Full + HC-Val only, fixed threshold 0.5

Current forbidden actions:
- seed43 / seed44 before Gate-TWA-E2-FSC writes a selected candidate
- HC-Test
- blind / external
- threshold search
- BN recalibration tuning
- new model training
- new checkpoint epochs outside 250/300/350/400
- new TWA combinations
- new verifier / suppression structure
```

---

## 9. STOPPED_BRANCHES_SUMMARY.md 局部更新

增加：

```markdown
## LateSnapshot-ep250 stopped at Gate-LS-A

Decision: `STOP_LATE_SNAPSHOT_EP250_AT_GATE_A`.

Reason:
- ep250 is very strong on seed42 HC-Val.
- However, ep250 fails seed42 Full non-regression vs OHEM-400.

Metrics:
- Full delta mIoU: -0.003254
- Full delta Precision: -0.000187
- Full delta FA ppm: -0.114901
- Full delta Pd: +0.000000
- HC-Val delta mIoU vs OHEM: +0.105858
- HC-Val delta FA ppm vs OHEM: -175.476

Interpretation:
- ep250 is retained as a stopped hard-clutter diagnostic control.
- ep250 must not be advanced to seed43/44, HC-Test, blind, or external.
- The next allowed audit is Gate-TWA-E2-FSC, which compares TWA-4 with Full-safe single-late controls only.
```

---

## 10. 新增 JSON 状态文件，可选但建议

新增：

```text
docs/internal/twa/seed42_nudt/gate_twa_e2_fullsafe_single_control_plan.json
```

内容：

```json
{
  "current_status": "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A",
  "previous_status": [
    "STOP_TWA_NO_BN_AT_GATE_E",
    "STOP_LATE_SNAPSHOT_EP250_AT_GATE_A"
  ],
  "next_allowed_gate": "Gate-TWA-E2-FSC",
  "gate_name": "Full-Safe Single-Late Control",
  "seed": 42,
  "allowed_epochs": [250, 300, 350, 400],
  "candidate_epochs": [250, 300, 350],
  "baseline_epoch": 400,
  "fixed_threshold": 0.5,
  "allowed_splits": ["Full", "HC-Val"],
  "forbidden_actions": [
    "seed43",
    "seed44",
    "HC-Test",
    "blind",
    "external",
    "threshold_search",
    "BN_tuning",
    "new_training",
    "new_epochs_outside_250_300_350_400",
    "new_TWA_combinations",
    "new_model_structure",
    "new_loss",
    "new_verifier_or_suppression_head"
  ]
}
```

---

## 11. 最小执行顺序

先只做代码和测试：

```bash
python -m py_compile tools/official/check_twa_gate_e2_fullsafe_single_control.py
pytest tests/test_twa_gate_e2_fullsafe_single_control.py -q
git diff --check
```

再运行 gate：

```bash
bash scripts/official/run_twa_gate_e2_fullsafe_single_control_seed42.sh
```

如果 `ep250 HC-Val summary` 路径不同：

```bash
EP250_HCVAL_SUMMARY=/home/ly/AAAI/OHCM-MSHNet-main/docs/internal/twa/seed42_nudt/<actual_ep250_hcval_dir>/summary_metrics.json \
bash scripts/official/run_twa_gate_e2_fullsafe_single_control_seed42.sh
```

---

## 12. Gate-TWA-E2-FSC 之后怎么走

### 12.1 如果输出 TWA-4-noBN

```text
selected_candidate = TWA-4-noBN
next_allowed_gate = Gate-TWA-F-seed43-44-Full-HCVal
```

下一步只跑：

```text
TWA-4 no-BN
seed43 / seed44
Full + HC-Val
fixed threshold 0.5
paired OHEM-400 baseline
```

要求：

| Split | 要求 |
|---|---|
| Full | seed42/43/44 三种子不退化；已知 seed42 已过，只补 43/44 |
| HC-Val | 至少 2/3 种子 mIoU/FA/Precision 正向，Pd 不下降 |
| mean | HC-Val mean mIoU 上升，FA 下降，Precision 上升 |
| Pd | 不下降 |

### 12.2 如果输出 LateSnapshot-ep300 或 LateSnapshot-ep350

```text
selected_candidate = LateSnapshot-epXXX
next_allowed_gate = Gate-LS-B-epXXX-seed43-44-Full-HCVal
```

下一步只跑：

```text
同一个 frozen epoch = XXX
seed43 / seed44
Full + HC-Val
fixed threshold 0.5
paired OHEM-400 baseline
```

不要同时推进 TWA-4 和 LateSnapshot。只能推进 checker 选出的一个候选。

### 12.3 如果输出 STOP_ALL_SINGLE_FORWARD_TRAJECTORY_COMPRESSION

立即停止：

```text
不跑 seed43/44
不跑 HC-Test
不跑 blind/external
不做 threshold search
不回 BN/TWA/checkpoint search
```

这个结果说明：

```text
single-forward trajectory compression/selection 这条线在当前证据下不能作为 AAAI 主线。
```

此时保留 TCE-4 作为 diagnostic oracle，论文路线需要另行决策，不应再花时间局部打补丁。

---

## 13. 为什么这个方案比“继续救 ep250”更适合 AAAI 15 天倒计时

### 13.1 它不改变模型和训练

新增的是 checker/script/test，不引入新结构，不污染现有结果。

### 13.2 它只处理新失败点

当前失败点不是：

```text
TWA-4 Full/HcVal vs OHEM 不行
```

而是：

```text
Gate-E 的 unrestricted best single control 是 ep250；
ep250 后来被证明 Full-unsafe。
```

所以本轮只补：

```text
best single control 必须 Full-safe。
```

### 13.3 它能快速给出三种明确结论

最多补 `ep300/ep350` 的 seed42 Full/HC-Val summary，然后 checker 直接给出：

```text
1. TWA-4 重新进入三种子；或
2. Full-safe single ep300/ep350 进入三种子；或
3. 全部 single-forward trajectory 路线停止。
```

不会继续无止境搜索。

---

## 14. 论文表述建议

如果 Gate-TWA-E2-FSC 选回 TWA-4：

```text
The initially strongest hard-clutter single checkpoint, ep250, dominated TWA-4 on HC-Val but failed the Full-split non-regression guard. We therefore compare TWA-4 against Full-safe single-checkpoint controls only. Under this eligible-control comparison, TWA-4 is retained as the safest single-forward trajectory compression candidate.
```

如果 Gate-TWA-E2-FSC 选 ep300/ep350：

```text
The unrestricted best hard-clutter snapshot ep250 was rejected by the Full-split guard. Among Full-safe late snapshots, epXXX provides the best hard-clutter improvement while preserving the MSHNetOHEM evidence anchor. We freeze epXXX before any multi-seed or external evaluation.
```

如果 Gate-TWA-E2-FSC 全停：

```text
Late-trajectory single-forward compression did not satisfy the required Full-split safety and hard-clutter robustness constraints. We therefore retain the trajectory ensemble only as a diagnostic oracle rather than an AAAI-ready single-forward method.
```

---

## 15. 最终建议

当前下一步不要再推进 ep250，也不要回头改 TWA Gate-E。最小增量方案是：

```text
Decision now:
  KEEP STOP_LATE_SNAPSHOT_EP250_AT_GATE_A

Next allowed gate:
  Gate-TWA-E2-FSC: Full-Safe Single-Late Control on seed42 only

Purpose:
  Re-evaluate the Gate-E best-single blocker after ep250 was shown Full-unsafe.

Allowed work:
  evaluate ep300/ep350 seed42 Full + HC-Val only if summaries are missing,
  then run the E2-FSC checker.

Forbidden:
  seed43/44 before E2-FSC decision,
  HC-Test,
  blind/external,
  threshold search,
  BN tuning,
  new training,
  new epochs,
  new TWA combinations,
  model/loss changes.
```

一句话：

> **ep250 已停止；现在只补一个 Full-safe single-control gate。若 TWA-4 只输给 Full-unsafe ep250，它可以被重新纳入三种子；若 ep300/ep350 有 Full-safe 且 HC-Val 更强，则只推进那个 frozen single snapshot；否则停止整条 single-forward trajectory 路线。**
