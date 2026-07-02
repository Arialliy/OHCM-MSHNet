# OHCM-MSHNet：TCE-4 Final F3 Blind / External Once 执行与代码修改方案

> 当前状态：TCSR-v1 已在 Gate-A bank audit 机制失败后停止；TCE-4-OHEM 已冻结为 final AAAI candidate；F0/F1/F2 已 PASS。本文只处理下一步：**Gate-TCE-F3 blind / external once**。
> 原则：F3 不是开发 gate，不再用于调方法；F3 是冻结方法后的最终一次性外部裁决。

---

## 0. 当前输入状态

你已经完成：

```text
STOP_TCSR_AT_BANK_AUDIT: recorded
TCE-4-OHEM: frozen as final AAAI candidate
Gate-TCE-F0 freeze consistency: PASS
Gate-TCE-F1 internal evidence aggregation: PASS
Gate-TCE-F2 threshold/component report: PASS
```

F1 已聚合的内部证据：

```text
Full:
  3/3 pass
  mean delta mIoU +0.002820

HC-Val:
  3/3 pass
  mean delta mIoU +0.017937
  mean delta FA ppm -38.146973

HC-Test:
  3/3 pass
  mean delta mIoU +0.014697
  mean delta FA ppm -25.893703
```

F2 已确认：

```text
HC threshold-matched: 12/12 PASS
Runtime cost: 4x forward, 已明确记录
FP component: 不全种子下降，但 FA ppm 全 split / seed 一致下降
```

当前唯一被允许的下一步：

```text
Gate-TCE-F3-blind-external-once
```

---

## 1. 当前决策

正式写法：

```text
Decision: PROCEED_TCE4_TO_F3_BLIND_EXTERNAL_ONCE

Candidate:
  TCE-4-OHEM

Baseline:
  MSHNetOHEM epoch400

Frozen inference:
  4 checkpoint forward trajectory consensus
  checkpoint epochs = [250, 300, 350, 400]
  threshold = 0.5
  probability definition unchanged

Allowed:
  one locked blind / external evaluation pass

Forbidden:
  new training
  new evaluation search
  seed selection
  checkpoint selection
  threshold search
  TCSR Stage 2
  loss.py / net.py / train.py / dataset.py / model modification
  rerunning blind / external after seeing results
```

---

## 2. 为什么现在可以进入 F3

现在进入 F3 的理由不是因为 TCE-4 完美，而是因为其他路线已经被干净停止，且 TCE-4 是唯一拥有完整内部稳定证据的候选。

已停止路线：

```text
TWA with BN recalibration: stopped
TWA-4 no-BN: diagnostic only
LateSnapshot-ep250: Full unsafe, stopped
LateSnapshot-ep300: post-hoc checkpoint selection, stopped as main method
TCSR-v1: train sparse negative bank too empty, stopped at Gate-A
seed / checkpoint / epoch selection: stopped as final method
```

TCE-4 的定位必须诚实：

```text
它不是 single-forward 方法。
它不是 train-time distillation 方法。
它是 frozen 4-forward trajectory-consensus inference method。
```

论文中可写的贡献应从“单模型压缩”改成：

```text
训练轨迹一致性揭示 IRSTD hard-clutter false alarm 的稳定性差异；
TCE-4 通过 frozen late-trajectory consensus 提升复杂背景鲁棒性；
虽然带来 4x inference cost，但在内部 Full / HC-Val / HC-Test、threshold-matched 和 FA ppm 上形成稳定证据。
```

---

## 3. F3 不是新的开发 gate

F3 的含义：

```text
一次性 final validation。
```

它不能再触发：

```text
换 seed
换 checkpoint
换 threshold
换 split
换 metric 主规则
删掉失败 external dataset
只报告好的 blind seed
```

F3 之后只有两种结果：

```text
F3 PASS:
  可以将 TCE-4 写成 AAAI main candidate，并报告 4x cost。

F3 FAIL:
  不再救方法；TCE-4 只能作为 internal evidence / oracle analysis，不能继续搜索。
```

---

## 4. F3 scope 冻结

建议写入：

```text
docs/internal/tce_final/tce4_f3_locked_eval_manifest.json
```

示例：

```json
{
  "gate": "Gate-TCE-F3-blind-external-once",
  "candidate": "TCE-4-OHEM",
  "baseline": "MSHNetOHEM-400",
  "status": "LOCKED_BEFORE_BLIND_EXTERNAL",
  "threshold": 0.5,
  "seeds": [42, 43, 44],
  "tce_epochs": [250, 300, 350, 400],
  "splits": ["blind", "external"],
  "probability_source": "foreground_probability.py / existing official probability path",
  "allowed_next_action": "RUN_F3_ONCE",
  "forbidden_actions": [
    "new training",
    "checkpoint search",
    "seed search",
    "threshold search",
    "BN tuning",
    "model/loss/train/dataset modification",
    "rerun after seeing blind/external results"
  ],
  "summary_paths": {
    "blind": {
      "42": {
        "ohem": "docs/internal/tce_final/f3/blind/seed42/ohem/summary_metrics.json",
        "tce4": "docs/internal/tce_final/f3/blind/seed42/tce4/summary_metrics.json"
      },
      "43": {
        "ohem": "docs/internal/tce_final/f3/blind/seed43/ohem/summary_metrics.json",
        "tce4": "docs/internal/tce_final/f3/blind/seed43/tce4/summary_metrics.json"
      },
      "44": {
        "ohem": "docs/internal/tce_final/f3/blind/seed44/ohem/summary_metrics.json",
        "tce4": "docs/internal/tce_final/f3/blind/seed44/tce4/summary_metrics.json"
      }
    },
    "external": {
      "42": {
        "ohem": "docs/internal/tce_final/f3/external/seed42/ohem/summary_metrics.json",
        "tce4": "docs/internal/tce_final/f3/external/seed42/tce4/summary_metrics.json"
      },
      "43": {
        "ohem": "docs/internal/tce_final/f3/external/seed43/ohem/summary_metrics.json",
        "tce4": "docs/internal/tce_final/f3/external/seed43/tce4/summary_metrics.json"
      },
      "44": {
        "ohem": "docs/internal/tce_final/f3/external/seed44/ohem/summary_metrics.json",
        "tce4": "docs/internal/tce_final/f3/external/seed44/tce4/summary_metrics.json"
      }
    }
  }
}
```

如果 external 有多个外部数据集，应在 `splits` 里写成显式名称，例如：

```json
"splits": ["blind", "external_sirst_aug", "external_irstd_1k"]
```

不要运行后再决定 external 到底算哪个。

---

## 5. F3 通过条件

F3 应该分成两个层级，避免因为外部数据波动把结论写得过满。

### 5.1 Strong PASS

每个 F3 split 都要求：

```text
mean_delta_mIoU      >= +0.001
mean_delta_FA_ppm    <= -5.0
mean_delta_Precision >= 0.0
min_delta_Pd         >= 0.0
```

并且没有灾难性单种子退化：

```text
min_delta_mIoU   >= -0.005
max_delta_FA_ppm <= +10.0
```

输出：

```text
F3_PASS_STRONG
```

含义：

```text
可以作为 AAAI main candidate。
```

### 5.2 Mixed but reportable

如果某个 split 满足：

```text
mean_delta_mIoU   >= 0.0
mean_delta_FA_ppm <= 0.0
min_delta_Pd      >= 0.0
```

但没有达到 Strong PASS 的 margin，输出：

```text
F3_PASS_MIXED_REPORTABLE
```

含义：

```text
可以报告，但论文表述要更保守：
TCE-4 consistently reduces false alarm on internal hard splits and remains non-regressive on external validation, with explicit 4x runtime cost.
```

### 5.3 Fail

如果任一 F3 split 出现：

```text
mean_delta_mIoU < 0
或 mean_delta_FA_ppm > 0
或 min_delta_Pd < 0
```

输出：

```text
F3_FAIL_NO_REDESIGN
```

含义：

```text
停止，不再 threshold / seed / checkpoint rescue。
```

---

## 6. 代码修改总览

不改：

```text
loss.py
net.py
train.py
dataset.py
model/
```

只新增 / 修改：

```text
[新增] docs/internal/tce_final/tce4_f3_locked_eval_manifest.json
[新增] tools/official/check_tce_f3_preflight.py
[新增] tools/official/check_tce_f3_blind_external_report.py
[新增] scripts/official/run_tce_f3_blind_external_once.sh
[新增] tests/test_tce_f3_blind_external_checker.py
[修改] README.md
[修改] STOPPED_BRANCHES_SUMMARY.md
```

---

## 7. `tools/official/check_tce_f3_preflight.py`

功能：

```text
1. 检查 F0/F1/F2 全部 PASS。
2. 检查 TCE-4 frozen manifest 存在。
3. 检查 F3 locked manifest 合法。
4. 检查 threshold=0.5、seeds=[42,43,44]、tce_epochs=[250,300,350,400]。
5. 检查 F3 final report 不存在，防止重复外部评估。
6. 创建 once lock。
```

代码：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_gate_pass(summary: Dict[str, Any], name: str) -> None:
    if summary.get("gate_pass") is not True:
        raise SystemExit(f"{name} is not PASS: gate_pass={summary.get('gate_pass')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--f0_summary", required=True)
    p.add_argument("--f1_summary", required=True)
    p.add_argument("--f2_summary", required=True)
    p.add_argument("--frozen_method_plan", required=True)
    p.add_argument("--f3_manifest", required=True)
    p.add_argument("--once_lock", required=True)
    p.add_argument("--final_report", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    final_report = Path(args.final_report)
    if final_report.exists():
        raise SystemExit(f"F3 final report already exists. Do not rerun blind/external: {final_report}")

    f0 = load_json(args.f0_summary)
    f1 = load_json(args.f1_summary)
    f2 = load_json(args.f2_summary)
    plan = load_json(args.frozen_method_plan)
    manifest = load_json(args.f3_manifest)

    require_gate_pass(f0, "Gate-TCE-F0")
    require_gate_pass(f1, "Gate-TCE-F1")
    require_gate_pass(f2, "Gate-TCE-F2")

    if manifest.get("candidate") != "TCE-4-OHEM":
        raise SystemExit(f"Unexpected candidate: {manifest.get('candidate')}")
    if manifest.get("baseline") != "MSHNetOHEM-400":
        raise SystemExit(f"Unexpected baseline: {manifest.get('baseline')}")
    if float(manifest.get("threshold")) != 0.5:
        raise SystemExit(f"F3 threshold must be fixed at 0.5, got {manifest.get('threshold')}")
    if list(manifest.get("seeds", [])) != [42, 43, 44]:
        raise SystemExit(f"F3 seeds must be [42,43,44], got {manifest.get('seeds')}")
    if list(manifest.get("tce_epochs", [])) != [250, 300, 350, 400]:
        raise SystemExit(f"F3 TCE epochs must be [250,300,350,400], got {manifest.get('tce_epochs')}")

    splits = manifest.get("splits", [])
    if not splits or not all(isinstance(x, str) and x for x in splits):
        raise SystemExit("F3 manifest must define non-empty splits list.")
    if "summary_paths" not in manifest:
        raise SystemExit("F3 manifest missing summary_paths.")

    # Validate paths are declared for every split/seed/method. They do not need to exist before running.
    for split in splits:
        if split not in manifest["summary_paths"]:
            raise SystemExit(f"Missing summary_paths for split={split}")
        for seed in ["42", "43", "44"]:
            pair = manifest["summary_paths"][split].get(seed)
            if not pair or "ohem" not in pair or "tce4" not in pair:
                raise SystemExit(f"Missing ohem/tce4 summary paths for split={split}, seed={seed}")

    lock_path = Path(args.once_lock)
    if lock_path.exists():
        lock = load_json(lock_path)
        if lock.get("status") == "COMPLETED":
            raise SystemExit("F3 once lock is already completed. Do not rerun.")
        if lock.get("manifest_sha256") != sha256_file(args.f3_manifest):
            raise SystemExit("Existing F3 lock manifest hash differs. Do not continue after manifest change.")
        status = "PREEXISTING_LOCK_RESUME_ALLOWED_FOR_MISSING_OUTPUTS_ONLY"
    else:
        lock = {
            "gate": "Gate-TCE-F3-blind-external-once",
            "status": "STARTED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": manifest.get("candidate"),
            "baseline": manifest.get("baseline"),
            "threshold": manifest.get("threshold"),
            "seeds": manifest.get("seeds"),
            "tce_epochs": manifest.get("tce_epochs"),
            "splits": manifest.get("splits"),
            "frozen_method_plan_sha256": sha256_file(args.frozen_method_plan),
            "f0_sha256": sha256_file(args.f0_summary),
            "f1_sha256": sha256_file(args.f1_summary),
            "f2_sha256": sha256_file(args.f2_summary),
            "manifest_sha256": sha256_file(args.f3_manifest),
        }
        save_json(lock, lock_path)
        status = "NEW_ONCE_LOCK_CREATED"

    out = {
        "gate": "Gate-TCE-F3-preflight",
        "gate_pass": True,
        "status": status,
        "next_allowed_action": "RUN_BLIND_EXTERNAL_ONCE",
        "once_lock": str(lock_path),
        "final_report": str(final_report),
        "forbidden_after_start": [
            "threshold_search",
            "seed_search",
            "checkpoint_search",
            "method_change",
            "rerun_after_result"
        ],
    }
    save_json(out, args.output)


if __name__ == "__main__":
    main()
```

---

## 8. `tools/official/check_tce_f3_blind_external_report.py`

功能：

```text
读取 F3 manifest 中声明的 blind / external 结果；
计算 OHEM-400 vs TCE-4 的 paired delta；
输出 split-level 和 global verdict；
将 once lock 标记为 COMPLETED。
```

代码：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_metric(summary: Dict[str, Any], key: str) -> float:
    candidates = {
        "mIoU": ["mIoU", "miou", "mean_iou"],
        "Precision": ["Precision", "precision", "Prec"],
        "Pd": ["Pd", "pd", "recall", "Recall"],
        "FA_ppm": ["FA_ppm", "FA", "fa", "FAppm", "fa_ppm"],
        "FP_components": ["FP_components", "fp_components", "FP", "fp"],
    }[key]

    for k in candidates:
        if k in summary:
            return float(summary[k])
    metrics = summary.get("metrics", {})
    for k in candidates:
        if k in metrics:
            return float(metrics[k])
    raise KeyError(f"Metric {key} not found. keys={list(summary.keys())}")


def delta_row(split: str, seed: int, ohem_path: str, tce4_path: str) -> Dict[str, Any]:
    ohem = load_json(ohem_path)
    tce4 = load_json(tce4_path)
    metrics = ["mIoU", "Precision", "Pd", "FA_ppm"]
    row: Dict[str, Any] = {
        "split": split,
        "seed": seed,
        "ohem_path": ohem_path,
        "tce4_path": tce4_path,
        "ohem": {},
        "tce4": {},
        "delta": {},
    }
    for m in metrics:
        ov = get_metric(ohem, m)
        tv = get_metric(tce4, m)
        row["ohem"][m] = ov
        row["tce4"][m] = tv
        row["delta"][m] = tv - ov

    # FP components are diagnostic; do not require all seeds to decrease.
    try:
        ov = get_metric(ohem, "FP_components")
        tv = get_metric(tce4, "FP_components")
        row["ohem"]["FP_components"] = ov
        row["tce4"]["FP_components"] = tv
        row["delta"]["FP_components"] = tv - ov
    except Exception:
        row["fp_components_available"] = False
    return row


def summarize_split(rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    ds = [r["delta"] for r in rows]
    out = {
        "num_seeds": len(rows),
        "mean_delta_mIoU": mean(d["mIoU"] for d in ds),
        "mean_delta_Precision": mean(d["Precision"] for d in ds),
        "mean_delta_Pd": mean(d["Pd"] for d in ds),
        "mean_delta_FA_ppm": mean(d["FA_ppm"] for d in ds),
        "min_delta_mIoU": min(d["mIoU"] for d in ds),
        "min_delta_Precision": min(d["Precision"] for d in ds),
        "min_delta_Pd": min(d["Pd"] for d in ds),
        "max_delta_FA_ppm": max(d["FA_ppm"] for d in ds),
        "per_seed": rows,
    }

    strong = (
        out["mean_delta_mIoU"] >= args.strong_min_mean_delta_miou
        and out["mean_delta_FA_ppm"] <= -args.strong_min_mean_fa_reduction
        and out["mean_delta_Precision"] >= args.min_mean_delta_precision
        and out["min_delta_Pd"] >= args.min_delta_pd
        and out["min_delta_mIoU"] >= args.min_seed_delta_miou
        and out["max_delta_FA_ppm"] <= args.max_seed_fa_increase
    )
    mixed = (
        out["mean_delta_mIoU"] >= args.min_mean_delta_miou
        and out["mean_delta_FA_ppm"] <= args.max_mean_delta_fa
        and out["min_delta_Pd"] >= args.min_delta_pd
    )

    if strong:
        out["split_verdict"] = "PASS_STRONG"
        out["split_pass"] = True
    elif mixed:
        out["split_verdict"] = "PASS_MIXED_REPORTABLE"
        out["split_pass"] = True
    else:
        out["split_verdict"] = "FAIL_NO_REDESIGN"
        out["split_pass"] = False
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--once_lock", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--strong_min_mean_delta_miou", type=float, default=0.001)
    p.add_argument("--strong_min_mean_fa_reduction", type=float, default=5.0)
    p.add_argument("--min_mean_delta_precision", type=float, default=0.0)
    p.add_argument("--min_delta_pd", type=float, default=0.0)
    p.add_argument("--min_seed_delta_miou", type=float, default=-0.005)
    p.add_argument("--max_seed_fa_increase", type=float, default=10.0)

    p.add_argument("--min_mean_delta_miou", type=float, default=0.0)
    p.add_argument("--max_mean_delta_fa", type=float, default=0.0)
    args = p.parse_args()

    manifest = load_json(args.manifest)
    lock_path = Path(args.once_lock)
    if not lock_path.exists():
        raise SystemExit(f"F3 once lock missing: {lock_path}")
    lock = load_json(lock_path)
    if lock.get("status") == "COMPLETED":
        raise SystemExit("F3 once lock already completed. Do not recompute final report.")

    splits = manifest["splits"]
    all_split_summaries: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []

    for split in splits:
        rows: List[Dict[str, Any]] = []
        for seed in [42, 43, 44]:
            pair = manifest["summary_paths"][split][str(seed)]
            ohem_path = Path(pair["ohem"])
            tce4_path = Path(pair["tce4"])
            if not ohem_path.exists():
                raise SystemExit(f"Missing OHEM summary: split={split}, seed={seed}, path={ohem_path}")
            if not tce4_path.exists():
                raise SystemExit(f"Missing TCE4 summary: split={split}, seed={seed}, path={tce4_path}")
            row = delta_row(split, seed, str(ohem_path), str(tce4_path))
            rows.append(row)
            all_rows.append(row)
        all_split_summaries[split] = summarize_split(rows, args)

    all_pass = all(v["split_pass"] for v in all_split_summaries.values())
    all_strong = all(v["split_verdict"] == "PASS_STRONG" for v in all_split_summaries.values())

    if all_strong:
        verdict = "F3_PASS_STRONG"
        gate_pass = True
        next_action = "WRITE_AAAI_MAIN_RESULTS_WITH_4X_COST"
    elif all_pass:
        verdict = "F3_PASS_MIXED_REPORTABLE"
        gate_pass = True
        next_action = "WRITE_AAAI_RESULTS_CONSERVATIVELY_WITH_4X_COST"
    else:
        verdict = "F3_FAIL_NO_REDESIGN"
        gate_pass = False
        next_action = "STOP_TCE4_AS_FINAL_AAAI_MAIN_METHOD"

    report = {
        "gate": "Gate-TCE-F3-blind-external-once",
        "candidate": manifest.get("candidate"),
        "baseline": manifest.get("baseline"),
        "threshold": manifest.get("threshold"),
        "seeds": manifest.get("seeds"),
        "tce_epochs": manifest.get("tce_epochs"),
        "gate_pass": gate_pass,
        "verdict": verdict,
        "split_summaries": all_split_summaries,
        "next_action": next_action,
        "forbidden_after_f3": [
            "rerun_blind_external",
            "threshold_search",
            "seed_search",
            "checkpoint_search",
            "new_training",
            "method_change_after_external"
        ],
    }
    save_json(report, args.output)

    lock["status"] = "COMPLETED"
    lock["completed_utc"] = datetime.now(timezone.utc).isoformat()
    lock["final_report"] = str(args.output)
    lock["verdict"] = verdict
    save_json(lock, lock_path)

    if not gate_pass:
        raise SystemExit("Gate-TCE-F3 failed. Stop; do not redesign or rerun after external results.")


if __name__ == "__main__":
    main()
```

---

## 9. `scripts/official/run_tce_f3_blind_external_once.sh`

这个脚本分三段：

```text
1. preflight + once lock
2. 执行 blind / external evaluation
3. 聚合 final report
```

注意：第二段的具体 evaluation 命令必须复用你本地已有 official TCE-4 eval 工具；不要在这里重写 inference 逻辑。

示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}
cd "${ROOT}"

F0="${ROOT}/docs/internal/tce_final/gate_tce_f0_freeze_summary.json"
F1="${ROOT}/docs/internal/tce_final/gate_tce_f1_internal_report.json"
F2="${ROOT}/docs/internal/tce_final/gate_tce_f2_threshold_component_report.json"
PLAN="${ROOT}/docs/internal/tce_final/tce4_frozen_method_plan.json"
MANIFEST="${ROOT}/docs/internal/tce_final/tce4_f3_locked_eval_manifest.json"
LOCK="${ROOT}/docs/internal/tce_final/gate_tce_f3_once_lock.json"
PREFLIGHT="${ROOT}/docs/internal/tce_final/gate_tce_f3_preflight_summary.json"
FINAL="${ROOT}/docs/internal/tce_final/gate_tce_f3_blind_external_report.json"

python tools/official/check_tce_f3_preflight.py \
  --f0_summary "${F0}" \
  --f1_summary "${F1}" \
  --f2_summary "${F2}" \
  --frozen_method_plan "${PLAN}" \
  --f3_manifest "${MANIFEST}" \
  --once_lock "${LOCK}" \
  --final_report "${FINAL}" \
  --output "${PREFLIGHT}"

# -------------------------------------------------------------------
# F3 ONCE EVALUATION BLOCK
# -------------------------------------------------------------------
# Replace the commands below with the exact existing official evaluation
# command(s) already used for TCE-4 and OHEM summaries.
# Rules:
#   - threshold must remain 0.5
#   - seeds must be 42,43,44
#   - TCE epochs must be 250,300,350,400
#   - output paths must match MANIFEST summary_paths
#   - do not add threshold/Pd/mIoU matching here
# -------------------------------------------------------------------

for SPLIT in blind external; do
  for SEED in 42 43 44; do
    echo "[F3] Evaluating OHEM seed=${SEED} split=${SPLIT}"
    # Example placeholder. Replace with your existing official evaluator.
    # python tools/official/evaluate_ohem_checkpoint.py \
    #   --seed "${SEED}" \
    #   --split "${SPLIT}" \
    #   --threshold 0.5 \
    #   --output_dir "docs/internal/tce_final/f3/${SPLIT}/seed${SEED}/ohem"

    echo "[F3] Evaluating TCE-4 seed=${SEED} split=${SPLIT}"
    # Example placeholder. Replace with your existing official TCE-4 evaluator.
    # python tools/official/evaluate_tce4_checkpoint_ensemble.py \
    #   --seed "${SEED}" \
    #   --split "${SPLIT}" \
    #   --epochs 250 300 350 400 \
    #   --threshold 0.5 \
    #   --output_dir "docs/internal/tce_final/f3/${SPLIT}/seed${SEED}/tce4"
  done
done

python tools/official/check_tce_f3_blind_external_report.py \
  --manifest "${MANIFEST}" \
  --once_lock "${LOCK}" \
  --output "${FINAL}"
```

如果你本地已有 F3 结果 summary，直接跳过 evaluation block，但仍必须让 `check_tce_f3_blind_external_report.py` 从 manifest 中读取所有 summary 并产出 final report。

---

## 10. `tests/test_tce_f3_blind_external_checker.py`

测试建议覆盖：

```text
1. F0/F1/F2 任一未 PASS，preflight fail。
2. threshold 不是 0.5，preflight fail。
3. seeds 不是 [42,43,44]，preflight fail。
4. TCE epochs 不是 [250,300,350,400]，preflight fail。
5. final_report 已存在，preflight fail。
6. blind/external strong pass，report gate_pass=True, verdict=F3_PASS_STRONG。
7. mixed reportable，report gate_pass=True, verdict=F3_PASS_MIXED_REPORTABLE。
8. mean FA 或 mIoU 失败，report gate_pass=False。
9. once lock completed 后再次运行 report，应 fail。
```

最小测试骨架：

```python
import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def metrics(miou, fa, precision=0.9, pd=0.98):
    return {"mIoU": miou, "FA_ppm": fa, "Precision": precision, "Pd": pd}


def test_f3_report_strong_pass(tmp_path):
    manifest = {
        "candidate": "TCE-4-OHEM",
        "baseline": "MSHNetOHEM-400",
        "threshold": 0.5,
        "seeds": [42, 43, 44],
        "tce_epochs": [250, 300, 350, 400],
        "splits": ["blind", "external"],
        "summary_paths": {},
    }
    for split in manifest["splits"]:
        manifest["summary_paths"][split] = {}
        for seed in [42, 43, 44]:
            ohem = tmp_path / split / str(seed) / "ohem.json"
            tce = tmp_path / split / str(seed) / "tce.json"
            write_json(ohem, metrics(0.80, 100.0, 0.90, 0.98))
            write_json(tce, metrics(0.81, 90.0, 0.91, 0.98))
            manifest["summary_paths"][split][str(seed)] = {
                "ohem": str(ohem),
                "tce4": str(tce),
            }

    manifest_path = tmp_path / "manifest.json"
    lock_path = tmp_path / "lock.json"
    out_path = tmp_path / "report.json"
    write_json(manifest_path, manifest)
    write_json(lock_path, {"status": "STARTED"})

    subprocess.run(
        [
            sys.executable,
            "tools/official/check_tce_f3_blind_external_report.py",
            "--manifest", str(manifest_path),
            "--once_lock", str(lock_path),
            "--output", str(out_path),
        ],
        check=True,
    )
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["gate_pass"] is True
    assert out["verdict"] == "F3_PASS_STRONG"
```

---

## 11. README 更新块

建议 README 顶部改为：

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Stopped / diagnostic branches:
- TWA with BN recalibration: stopped.
- TWA-4 no-BN: diagnostic only; stopped after mechanism controls.
- LateSnapshot-ep250: stopped at Gate-LS-A because Full split is unsafe.
- LateSnapshot-ep300: diagnostic only; not promoted due to post-hoc checkpoint selection risk.
- TCSR-v1: stopped at Gate-TCSR-A bank audit because train-only sparse negative signal is insufficient.
- Seed / checkpoint / epoch selection: stopped as AAAI main method.

Final frozen candidate:
- TCE-4-OHEM trajectory-consensus inference.
- Checkpoint epochs: 250, 300, 350, 400.
- Seeds for final paired reporting: 42, 43, 44.
- Threshold: fixed 0.5.
- Inference cost: 4x forward, explicitly reported.

Internal evidence:
- Gate-TCE-F0 freeze consistency: PASS.
- Gate-TCE-F1 internal evidence aggregation: PASS.
- Gate-TCE-F2 threshold/component report: PASS.

Next allowed step:
- Gate-TCE-F3 blind/external once.

Forbidden:
- new training
- new model/loss/dataset changes
- seed/checkpoint/threshold search
- rerunning blind/external after seeing results
```

---

## 12. STOPPED_BRANCHES_SUMMARY 更新块

新增：

```markdown
## TCSR-v1 stopped at Gate-A bank audit

Decision: STOP_TCSR_AT_BANK_AUDIT.

Reason:
- train images: 697
- images with sparse negatives: 1, required >= 50
- negative pixels total: 130, required >= 500
- target leakage pixels: 0
- neg/protect overlap pixels: 0

Interpretation:
- This is not a path, shape, image_id, or JSON schema issue.
- The TCSR-v1 mechanism gate failed because the train-only sparse hard-clutter negative signal is insufficient.
- Do not lower thresholds or tune lambda to rescue TCSR.

Next final candidate:
- TCE-4-OHEM frozen trajectory-consensus inference.
- Proceed only to Gate-TCE-F3 blind/external once after F0/F1/F2 pass.
```

---

## 13. 运行顺序

```bash
python -m py_compile \
  tools/official/check_tce_f3_preflight.py \
  tools/official/check_tce_f3_blind_external_report.py

pytest tests/test_tce_f3_blind_external_checker.py -q
git diff --check

bash scripts/official/run_tce_f3_blind_external_once.sh
```

---

## 14. F3 之后怎么写论文

### 如果 F3_PASS_STRONG

论文主结论：

```text
TCE-4 consistently improves hard-clutter robustness and external validation over the strong MSHNetOHEM anchor, at the cost of 4x inference.
```

可写：

```text
main table: Full / HC-Val / HC-Test / Blind / External
ablation: OHEM, TWA, LateSnapshot diagnostics, TCE-4
runtime table: 1x OHEM vs 4x TCE-4
component analysis: FP components mixed, but FA ppm consistently reduced
threshold-matched table: 12/12 internal hard-split PASS
```

### 如果 F3_PASS_MIXED_REPORTABLE

论文表述要降级：

```text
TCE-4 provides stable internal hard-clutter gains and non-regressive external validation, but the external margin is modest.
```

不写：

```text
state-of-the-art generalization on all external data
```

### 如果 F3_FAIL_NO_REDESIGN

停止：

```text
Do not tune after blind/external.
Do not rerun with different threshold.
Do not remove failed external split.
Do not switch back to ep300 / TWA / TCSR.
```

可保留为论文内部分析或 workshop-level fallback：

```text
Trajectory consensus improves internal hard-clutter robustness but does not generalize sufficiently to frozen blind/external validation.
```

---

## 15. 最终结论

现在可以定：

```text
PROCEED_TCE4_TO_F3_BLIND_EXTERNAL_ONCE
```

但必须保持边界：

```text
F3 是最终裁决，不是继续开发。
```

当前不再需要改模型和训练代码。下一步只需要新增 F3 preflight、once lock、final report checker、manifest 和测试。跑完 F3 后，如果通过，就进入论文表格和写作；如果失败，就停止，不再 rescue。
