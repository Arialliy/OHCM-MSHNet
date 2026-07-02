# OHCM-MSHNet：TCSR Gate-A 失败后的停止决策与 15 天 AAAI 最小可行路线

> 当前状态：`STOP_TCSR_AT_BANK_AUDIT` 已触发。
> 当前关键事实：TCSR-v1 所依赖的 train-only sparse hard-clutter negative bank 基本不存在。
> 结论：不要继续 Stage 2，不要写 TCSR loss / net / train，不要调阈值救 bank。下一步应停止所有 single-forward compression rescue，把已经有稳定正信号的 `TCE-4` 冻结为最后的 AAAI 主候选，并做 finalization / reporting gate。

---

## 0. 本文处理什么问题

这份文档只处理当前失败点：

```text
Gate-TCSR-A train-only sparse bank audit: FAIL
```

已报告结果：

```json
{
  "gate_pass": false,
  "num_images": 697,
  "num_images_with_neg": 1,
  "required_num_images_with_neg": 50,
  "neg_pixels_total": 130,
  "required_neg_pixels_total": 500,
  "target_leakage_pixels_total": 0,
  "neg_protect_overlap_pixels_total": 0
}
```

这不是：

```text
path 错
shape 错
image_id 对齐错
JSON schema 错
loader 没读到文件
bank item 写坏
```

而是机制 gate 失败：

```text
train split 上没有足够的 target-safe sparse hard-clutter negative signal。
```

---

## 1. 当前正式决策

```text
Decision:
  STOP_TCSR_AT_BANK_AUDIT

Reason:
  TCSR-v1 requires a train-only sparse negative reliability bank.
  The bank audit found only 1 / 697 images with negative pixels and only 130 negative pixels total.
  This is far below the pre-fixed requirements: >= 50 images and >= 500 pixels.

Do not run:
  Gate-TCSR-B activation sanity
  TCSR seed42 training
  seed43 / seed44
  HC-Test for TCSR
  blind / external for TCSR
  threshold search
  lambda search
  BN tuning
  new verifier / suppression head

Do not modify now:
  loss.py for TCSR
  net.py for TCSR
  train.py for TCSR
  dataset.py for TCSR image_id plumbing

Allowed now:
  status / checker / README updates
  aggregate already validated TCE evidence
  freeze a final no-search TCE-4 method candidate
```

一句话：**TCSR-v1 停止，不进入训练阶段。**

---

## 2. Gate-A 失败说明什么

TCSR-v1 的结构假设是：

```text
TCE trajectory consensus 能在 train split 上暴露出 OHEM 高置信、TCE 低置信、远离 GT target 的 sparse local peaks。
```

Gate-A 的结果否定了这个假设：

```text
num_images_with_neg = 1 / 697
neg_pixels_total    = 130
```

这说明在当前固定定义下，TCE 与 OHEM 的可学习差异不是一个足够大的 train-time pseudo-label source。

更具体地说，失败不是因为 bank 不安全：

```text
target_leakage_pixels_total = 0
neg_protect_overlap_pixels_total = 0
```

安全性没问题，问题是**信号规模不足**。

因此不能说：

```text
TCSR bank 很干净，所以可以继续训练。
```

必须说：

```text
TCSR bank 很干净，但几乎为空，所以不能作为训练监督。
```

---

## 3. 为什么不能“稍微放宽阈值救一下”

当前阈值是方案冻结的一部分：

```text
far_bg radius = 7
local_peak kernel = 7
anchor_high = p_ohem >= 0.50
tce_low = p_tce_mean <= 0.35
disagreement = p_ohem - p_tce_mean >= 0.15
neg dilation radius = 2
min_images_with_neg = 50
min_neg_pixels_total = 500
```

如果现在根据失败结果修改：

```text
p_ohem >= 0.40
tce_low <= 0.50
disagreement >= 0.05
local_peak kernel = 3
min_images_with_neg = 10
min_neg_pixels_total = 100
```

这就不是代码修复，而是 validation-aware mechanism search。即使后续跑出提升，也很难解释为干净方法。

允许修的是实现错误：

```text
路径错误
summary key 不一致
tensor shape / dtype / device 错误
image_id 对齐错误
文件写入缺失
pytest / py_compile / git diff --check 问题
```

不允许修的是机制失败：

```text
bank 太稀疏后放宽阈值
loss 没效果后调 lambda
seed42 不好后换 seed
checkpoint 不好后换 epoch
HC-Val 不好后换 threshold
```

当前属于后者，所以停止。

---

## 4. 当前真正剩下的问题

原目标没有变：

```text
保持 MSHNetOHEM 的小目标 evidence anchor；
降低复杂背景 hard-clutter false alarms；
提升 Precision；
不牺牲 Pd；
尽量保持单模型、单 forward。
```

但现在必须承认一个事实：

```text
“单模型、单 forward”路线目前没有通过机制 gate。
```

已经停止或降级为 diagnostic 的路线：

```text
TWA with BN recalibration       stopped
TWA-4 no-BN                     diagnostic only
LateSnapshot-ep250              stopped, Full unsafe
LateSnapshot-ep300              diagnostic only, post-hoc epoch selection risk
TCSR-v1 sparse bank             stopped, train signal absent
seed / checkpoint / epoch 选择    stopped as main method
```

所以当前不能再问：

```text
哪个 seed 好？
哪个 epoch 好？
哪个 checkpoint 好？
哪个 TWA combination 好？
```

而应该问：

```text
已有证据里，哪个方法真正解决了 hard clutter，并且还能在 15 天内被干净地冻结、验证和写进论文？
```

答案只能是：

```text
TCE-4 trajectory-consensus inference。
```

---

## 5. 为什么下一步应转向 TCE-4 finalization

### 5.1 TCE-4 是目前唯一稳定正信号

已有内部记录显示，TCE-OHEM 曾经是当前最强 trajectory signal：

```text
Full / HC-Val / HC-Test 三种子 3/3 优于 OHEM-400
hard-split threshold-matched 12/12 通过
```

TCE 的问题不是效果弱，而是：

```text
4x inference
多个 checkpoint
创新性容易被认为像 ensemble
```

但在当前 15 天冲刺条件下，这比继续 single-forward rescue 更现实。

### 5.2 TCSR-A 的失败反而支持 TCE 的定位

TCSR-A 失败说明：

```text
TCE 的收益不容易被压缩成 train-time sparse pseudo-label。
```

这意味着 hard-clutter 抑制可能主要来自 inference-time trajectory consensus：

```text
真实 target 在训练轨迹上更稳定；
hard clutter 在训练轨迹上更不稳定；
多 checkpoint consensus 能在推理阶段直接平均掉不稳定激活。
```

这种机制不要求 train split 中存在大量 OHEM-high / TCE-low hard negative bank。

所以 TCSR 停止后，不应再强行蒸馏 TCE，而应诚实地把 TCE 作为 final candidate：

```text
Trajectory-consensus inference for robust IRSTD。
```

### 5.3 论文叙述必须诚实

不要写：

```text
We propose a single-forward efficient model.
```

应该写：

```text
We propose a training-trajectory consensus inference framework that uses late checkpoints from a single training trajectory to suppress unstable hard-clutter activations while preserving stable small-target evidence.
```

也就是说，贡献是可靠性和 false-alarm robustness，不是速度。

---

## 6. 新的冻结路线：TCE-4 Finalization

### 6.1 方法定义

```text
Method name:
  TCE-4-OHEM

Base model:
  MSHNetOHEM

Checkpoints:
  ep250, ep300, ep350, ep400

Training:
  no new training

Inference:
  4 checkpoint forwards
  aggregate foreground probabilities using the existing official TCE implementation
  fixed threshold = 0.5

Forbidden:
  seed selection
  checkpoint selection
  threshold search
  BN recalibration
  TWA combination search
  HC-Val-guided checkpoint subset selection
```

如果现有 TCE 实现是 probability averaging，就保持 probability averaging。
如果现有 TCE 实现是 logit averaging，就保持 logit averaging。
**不要现在切换 aggregation 规则。**

### 6.2 TCE-2 / TCE-3 的位置

可以保留为 ablation，不作为选择空间。

```text
TCE-2: diagnostic ablation
TCE-3: diagnostic ablation
TCE-4: frozen main method
```

不要写：

```text
我们比较 TCE-2/3/4 后选最好的。
```

应该写：

```text
We use TCE-4 as the pre-fixed full trajectory-consensus setting and report TCE-2/3 only as budget-performance ablations.
```

---

## 7. 新增 Gate：Gate-TCE-FINAL

当前不再新增模型结构，而是新增 finalization gate。

```text
Gate-TCE-F0: stop-state consistency
Gate-TCE-F1: frozen TCE-4 internal evidence aggregation
Gate-TCE-F2: threshold-matched / component report consistency
Gate-TCE-F3: one-time blind / external evaluation permission
```

---

## 8. Gate-TCE-F0：stop-state consistency

目标：确认所有错误路线都已经停止，不会再被脚本误推进。

必须检查：

```text
STOP_TCSR_AT_BANK_AUDIT exists
STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD exists
LateSnapshot-ep300 is diagnostic only
TWA-4 no-BN is diagnostic only
TCE-4 is the only active final candidate
```

通过后允许：

```text
aggregate existing TCE summaries
```

不允许：

```text
new training
new TCSR loss
seed43/44 search
checkpoint subset search
threshold search
```

---

## 9. Gate-TCE-F1：internal evidence aggregation

### 9.1 输入

固定比较：

```text
OHEM-400 vs TCE-4
```

固定种子：

```text
seed42
seed43
seed44
```

固定 split：

```text
Full
HC-Val
HC-Test
```

固定指标：

```text
mIoU
Precision
Pd
FA ppm
FP components, if available
```

### 9.2 如果已有 summaries

直接 aggregate，不重复跑。

### 9.3 如果 summaries 缺失

只补固定方法的缺失 evaluation：

```text
TCE-4 fixed checkpoints: 250,300,350,400
OHEM-400 paired baseline
threshold = 0.5
no checkpoint search
no threshold search
no new training
```

### 9.4 建议通过条件

严格版：

```text
Full:
  3/3 seeds mIoU >= OHEM
  3/3 seeds Precision >= OHEM
  3/3 seeds Pd >= OHEM
  3/3 seeds FA ppm <= OHEM

HC-Val:
  3/3 seeds mIoU >= OHEM + 0.005
  3/3 seeds FA ppm <= OHEM - 10
  3/3 seeds Precision >= OHEM
  3/3 seeds Pd >= OHEM

HC-Test:
  3/3 seeds mIoU >= OHEM + 0.005
  3/3 seeds FA ppm <= OHEM - 10
  3/3 seeds Precision >= OHEM
  3/3 seeds Pd >= OHEM
```

如果历史记录确实是 3/3 全正，这个 gate 应该能通过。
如果某个 split 不是 3/3，则不要降级为“挑 seed”。可以改成论文中诚实报告 mean/std，但 final gate 必须标记为：

```text
TCE_INTERNAL_EVIDENCE_PARTIAL
```

此时是否继续 blind/external 需要人工决定，但不能声称三种子稳定。

---

## 10. Gate-TCE-F2：threshold-matched 与 component report

这一步不是调 threshold，而是验证 TCE 不是只靠固定阈值偶然占优。

必须报告：

```text
fixed threshold = 0.5
Pd-matched FA
mIoU-matched Pd / FA
FP component count
far-background component count, if available
target lost count, if available
runtime / FLOPs / forward count
```

通过条件：

```text
threshold-matched hard-split report remains positive
TCE does not trade away Pd
FP components decrease or FA ppm decreases consistently
runtime overhead is explicitly reported as 4x forward
```

注意：

```text
不要用 threshold-matched 结果反过来选择 threshold。
```

---

## 11. Gate-TCE-F3：blind / external once

只有 F0/F1/F2 都通过，才允许一次 final blind / external。

冻结内容：

```text
method = TCE-4-OHEM
checkpoints = 250,300,350,400
aggregation = existing official TCE aggregation
threshold = 0.5
seeds = 42/43/44 internal report; blind/external no seed search
no BN recalibration
no checkpoint replacement
no threshold adaptation
```

blind / external 失败时：

```text
不要回头调 TCE subset
不要改 threshold
不要改 checkpoints
不要重写 TCSR
```

只能诚实报告 generalization 限制，或者把 AAAI 主线降级。

---

## 12. 代码修改总原则

现在不要动：

```text
loss.py
net.py
train.py
dataset.py
model/
```

只新增或修改：

```text
docs/internal/... 状态 JSON
README.md
STOPPED_BRANCHES_SUMMARY.md
tools/official/check_tce_final_freeze.py
tools/official/aggregate_tce_final_report.py
scripts/official/run_tce_final_freeze_and_report.sh
tests/test_tce_final_freeze_checker.py
```

原因：

```text
现在已经不是模型开发阶段，而是 final candidate 冻结和 evidence aggregation 阶段。
```

---

## 13. 新增状态文件

新增：

```text
docs/internal/tce_final/tce4_frozen_method_plan.json
```

建议内容：

```json
{
  "decision": "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE",
  "previous_stop": {
    "tcsr": "STOP_TCSR_AT_BANK_AUDIT",
    "posthoc_selection": "STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD",
    "late_snapshot_ep300": "DIAGNOSTIC_ONLY",
    "twa4_no_bn": "DIAGNOSTIC_ONLY"
  },
  "method": {
    "name": "TCE-4-OHEM",
    "base": "MSHNetOHEM",
    "checkpoints": [250, 300, 350, 400],
    "aggregation": "existing_official_tce_aggregation",
    "threshold": 0.5,
    "training": "no_new_training",
    "inference_forward_count": 4
  },
  "forbidden": [
    "seed_search",
    "checkpoint_search",
    "threshold_search",
    "BN_recalibration_tuning",
    "TCSR_training",
    "new_loss",
    "new_model_structure"
  ],
  "next_allowed_gate": "Gate-TCE-F1-internal-evidence-aggregation"
}
```

---

## 14. `tools/official/check_tce_final_freeze.py`

新增文件：

```text
tools/official/check_tce_final_freeze.py
```

功能：

```text
1. 读取 TCSR Gate-A summary，确认 gate_pass=false 且 stop decision 正确。
2. 读取 TCE frozen plan，确认 method/checkpoints/threshold/forbidden actions 固定。
3. 阻止 ep300 / TWA-4 被 promotion 成 next gate。
4. 输出 Gate-TCE-F0 summary。
```

代码骨架：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, reason: str, failures: list[str]):
    if not condition:
        failures.append(reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcsr_gate_a_summary", required=True)
    parser.add_argument("--tce_frozen_plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tcsr = load_json(args.tcsr_gate_a_summary)
    plan = load_json(args.tce_frozen_plan)

    failures = []

    require(tcsr.get("gate_pass") is False, "TCSR Gate-A must be failed/stopped", failures)
    require(
        tcsr.get("decision") == "STOP_TCSR_AT_BANK_AUDIT"
        or tcsr.get("next_allowed_gate") == "STOP_TCSR_AT_BANK_AUDIT"
        or tcsr.get("stop_decision") == "STOP_TCSR_AT_BANK_AUDIT",
        "TCSR stop decision missing",
        failures,
    )

    method = plan.get("method", {})
    require(plan.get("decision") == "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE", "wrong freeze decision", failures)
    require(method.get("name") == "TCE-4-OHEM", "method name must be TCE-4-OHEM", failures)
    require(method.get("base") == "MSHNetOHEM", "base must be MSHNetOHEM", failures)
    require(method.get("checkpoints") == [250, 300, 350, 400], "checkpoints must be [250,300,350,400]", failures)
    require(float(method.get("threshold")) == 0.5, "threshold must be fixed at 0.5", failures)
    require(method.get("training") == "no_new_training", "must not require new training", failures)
    require(int(method.get("inference_forward_count")) == 4, "TCE-4 must report 4 forwards", failures)

    forbidden = set(plan.get("forbidden", []))
    for key in [
        "seed_search",
        "checkpoint_search",
        "threshold_search",
        "BN_recalibration_tuning",
        "TCSR_training",
        "new_loss",
        "new_model_structure",
    ]:
        require(key in forbidden, f"forbidden action missing: {key}", failures)

    gate_pass = len(failures) == 0
    result = {
        "gate": "Gate-TCE-F0-freeze-consistency",
        "gate_pass": gate_pass,
        "decision": "PROCEED_TO_TCE_INTERNAL_AGGREGATION" if gate_pass else "STOP_TCE_FINALIZATION_FREEZE_INCONSISTENT",
        "failures": failures,
        "next_allowed_gate": "Gate-TCE-F1-internal-evidence-aggregation" if gate_pass else "STOP",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not gate_pass:
        raise SystemExit("Gate-TCE-F0 failed. Fix status/freeze inconsistency only.")


if __name__ == "__main__":
    main()
```

---

## 15. `tools/official/aggregate_tce_final_report.py`

新增文件：

```text
tools/official/aggregate_tce_final_report.py
```

功能：

```text
读取 OHEM 和 TCE-4 在 seed42/43/44、Full/HC-Val/HC-Test 上的 summary_metrics.json；
输出 paired delta、mean/std、3/3 pass 状态；
不做任何 threshold / seed / checkpoint 选择。
```

核心逻辑骨架：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

METRICS = ["mIoU", "Precision", "Pd", "FA_ppm"]


def load_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(summary, key: str) -> float:
    if key in summary:
        return float(summary[key])
    if "metrics" in summary and key in summary["metrics"]:
        return float(summary["metrics"][key])
    raise KeyError(f"Metric {key} not found. Available keys: {list(summary.keys())}")


def summarize(values):
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON manifest listing paired OHEM/TCE summaries.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_hc_miou_delta", type=float, default=0.005)
    parser.add_argument("--min_hc_fa_reduction", type=float, default=10.0)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    rows = []

    for item in manifest["pairs"]:
        seed = int(item["seed"])
        split = item["split"]
        ohem = load_json(item["ohem_summary"])
        tce = load_json(item["tce_summary"])

        row = {"seed": seed, "split": split, "delta": {}}
        for m in METRICS:
            row["ohem_" + m] = get_metric(ohem, m)
            row["tce_" + m] = get_metric(tce, m)
            row["delta"][m] = row["tce_" + m] - row["ohem_" + m]
        rows.append(row)

    by_split = {}
    for split in sorted({r["split"] for r in rows}):
        split_rows = [r for r in rows if r["split"] == split]
        delta_summary = {
            m: summarize([r["delta"][m] for r in split_rows])
            for m in METRICS
        }

        if split == "full":
            pass_rows = [
                r["delta"]["mIoU"] >= 0.0
                and r["delta"]["Precision"] >= 0.0
                and r["delta"]["Pd"] >= 0.0
                and r["delta"]["FA_ppm"] <= 0.0
                for r in split_rows
            ]
        else:
            pass_rows = [
                r["delta"]["mIoU"] >= args.min_hc_miou_delta
                and r["delta"]["Precision"] >= 0.0
                and r["delta"]["Pd"] >= 0.0
                and r["delta"]["FA_ppm"] <= -args.min_hc_fa_reduction
                for r in split_rows
            ]

        by_split[split] = {
            "num_seeds": len(split_rows),
            "num_pass": int(sum(pass_rows)),
            "all_pass": all(pass_rows),
            "delta_summary": delta_summary,
            "rows": split_rows,
        }

    gate_pass = all(v["all_pass"] for v in by_split.values())
    result = {
        "gate": "Gate-TCE-F1-internal-evidence-aggregation",
        "method": "TCE-4-OHEM",
        "gate_pass": gate_pass,
        "by_split": by_split,
        "rows": rows,
        "decision": "PROCEED_TO_TCE_THRESHOLD_COMPONENT_REPORT" if gate_pass else "TCE_INTERNAL_EVIDENCE_PARTIAL_OR_FAIL",
        "next_allowed_gate": "Gate-TCE-F2-threshold-component-report" if gate_pass else "STOP_OR_REPORT_LIMITATION",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not gate_pass:
        raise SystemExit("Gate-TCE-F1 failed or partial. Do not select seeds/checkpoints to rescue.")


if __name__ == "__main__":
    main()
```

---

## 16. Manifest 文件

新增：

```text
docs/internal/tce_final/tce4_internal_manifest.json
```

示例：

```json
{
  "method": "TCE-4-OHEM",
  "checkpoints": [250, 300, 350, 400],
  "threshold": 0.5,
  "pairs": [
    {
      "seed": 42,
      "split": "full",
      "ohem_summary": "docs/internal/tce/seed42_nudt/eval_full_ohem/summary_metrics.json",
      "tce_summary": "docs/internal/tce/seed42_nudt/eval_full_tce4/summary_metrics.json"
    },
    {
      "seed": 42,
      "split": "hcval",
      "ohem_summary": "docs/internal/tce/seed42_nudt/eval_hcval_ohem/summary_metrics.json",
      "tce_summary": "docs/internal/tce/seed42_nudt/eval_hcval_tce4/summary_metrics.json"
    },
    {
      "seed": 42,
      "split": "hctest",
      "ohem_summary": "docs/internal/tce/seed42_nudt/eval_hctest_ohem/summary_metrics.json",
      "tce_summary": "docs/internal/tce/seed42_nudt/eval_hctest_tce4/summary_metrics.json"
    }
  ]
}
```

实际使用时补全 seed43 / seed44。

---

## 17. 一键脚本

新增：

```text
scripts/official/run_tce_final_freeze_and_report.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}

python tools/official/check_tce_final_freeze.py \
  --tcsr_gate_a_summary "${ROOT}/docs/internal/tcsr/seed42_nudt/gate_tcsr_a_bank_summary.json" \
  --tce_frozen_plan "${ROOT}/docs/internal/tce_final/tce4_frozen_method_plan.json" \
  --output "${ROOT}/docs/internal/tce_final/gate_tce_f0_freeze_summary.json"

python tools/official/aggregate_tce_final_report.py \
  --manifest "${ROOT}/docs/internal/tce_final/tce4_internal_manifest.json" \
  --output "${ROOT}/docs/internal/tce_final/gate_tce_f1_internal_report.json"
```

注意：这个脚本只做 checker 和 aggregation。
如果缺 summary，不要让这个脚本自动跑大量 eval。缺什么，在 manifest 里明确记录后单独补固定 evaluation。

---

## 18. 测试文件

新增：

```text
tests/test_tce_final_freeze_checker.py
```

测试内容：

```python
import json
import subprocess
from pathlib import Path


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_tce_freeze_passes_when_tcsr_stopped_and_plan_fixed(tmp_path):
    tcsr = tmp_path / "tcsr.json"
    plan = tmp_path / "plan.json"
    out = tmp_path / "out.json"

    write_json(tcsr, {
        "gate_pass": False,
        "decision": "STOP_TCSR_AT_BANK_AUDIT",
    })
    write_json(plan, {
        "decision": "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE",
        "method": {
            "name": "TCE-4-OHEM",
            "base": "MSHNetOHEM",
            "checkpoints": [250, 300, 350, 400],
            "aggregation": "existing_official_tce_aggregation",
            "threshold": 0.5,
            "training": "no_new_training",
            "inference_forward_count": 4,
        },
        "forbidden": [
            "seed_search",
            "checkpoint_search",
            "threshold_search",
            "BN_recalibration_tuning",
            "TCSR_training",
            "new_loss",
            "new_model_structure",
        ],
    })

    subprocess.run([
        "python",
        "tools/official/check_tce_final_freeze.py",
        "--tcsr_gate_a_summary", str(tcsr),
        "--tce_frozen_plan", str(plan),
        "--output", str(out),
    ], check=True)

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["gate_pass"] is True
    assert result["next_allowed_gate"] == "Gate-TCE-F1-internal-evidence-aggregation"


def test_tce_freeze_fails_if_checkpoint_list_changes(tmp_path):
    tcsr = tmp_path / "tcsr.json"
    plan = tmp_path / "plan.json"
    out = tmp_path / "out.json"

    write_json(tcsr, {
        "gate_pass": False,
        "decision": "STOP_TCSR_AT_BANK_AUDIT",
    })
    write_json(plan, {
        "decision": "FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE",
        "method": {
            "name": "TCE-4-OHEM",
            "base": "MSHNetOHEM",
            "checkpoints": [300, 350, 400],
            "threshold": 0.5,
            "training": "no_new_training",
            "inference_forward_count": 3,
        },
        "forbidden": [
            "seed_search",
            "checkpoint_search",
            "threshold_search",
            "BN_recalibration_tuning",
            "TCSR_training",
            "new_loss",
            "new_model_structure",
        ],
    })

    proc = subprocess.run([
        "python",
        "tools/official/check_tce_final_freeze.py",
        "--tcsr_gate_a_summary", str(tcsr),
        "--tce_frozen_plan", str(plan),
        "--output", str(out),
    ])

    assert proc.returncode != 0
```

---

## 19. README 更新

把 README 顶部状态改为：

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Stopped / diagnostic branches:
- TWA with BN recalibration: stopped.
- TWA-4 no-BN: diagnostic only; not promoted as final single-forward method.
- LateSnapshot-ep250: stopped at Gate-LS-A because Full split is unsafe.
- LateSnapshot-ep300: diagnostic only; not promoted because its HC-Val advantage over TWA-4 is a numerical tie / post-hoc checkpoint effect.
- TCSR-v1: stopped at Gate-TCSR-A bank audit because the train-only sparse hard-clutter negative bank is too sparse.
- Post-hoc seed / checkpoint / epoch selection: stopped as AAAI main method.

Current final candidate:
- TCE-4-OHEM trajectory-consensus inference.
- Base model: MSHNetOHEM.
- Checkpoints: ep250 / ep300 / ep350 / ep400.
- Inference: 4 forwards, existing official TCE aggregation, fixed threshold 0.5.
- Training: no new training.

Next allowed gate:
- Gate-TCE-F0/F1 final freeze and internal evidence aggregation.

Forbidden:
- seed search
- checkpoint search
- threshold search
- BN recalibration tuning
- TCSR training
- new model / loss / verifier / suppression structure
```

---

## 20. STOPPED_BRANCHES_SUMMARY 更新

新增：

```markdown
## TCSR-v1 stopped at bank audit

Decision: STOP_TCSR_AT_BANK_AUDIT.

Reason:
- Gate-TCSR-A found only 1 / 697 train images with sparse negative reliability pixels.
- Total negative pixels = 130, below the fixed minimum requirement of 500.
- Target leakage and protect overlap were zero, so this is not a safety failure or code-alignment failure.
- The train-only sparse hard-clutter negative signal required by TCSR-v1 is absent under the frozen definition.

Forbidden:
- Do not run TCSR activation sanity.
- Do not add TCSR loss/net/train changes.
- Do not tune bank thresholds or lambdas to rescue the route.
- Do not run seed43/44, HC-Test, blind, or external for TCSR.

Next route:
- Freeze TCE-4-OHEM trajectory-consensus inference as the final AAAI candidate and aggregate already validated internal evidence.
```

---

## 21. 15 天最小执行路线

### Day 1：状态冻结和 checker

```bash
python -m py_compile tools/official/check_tce_final_freeze.py
pytest tests/test_tce_final_freeze_checker.py -q
git diff --check

bash scripts/official/run_tce_final_freeze_and_report.sh
```

### Day 2：补齐 internal report

只补缺失的固定 evaluation：

```text
OHEM-400 vs TCE-4
seeds = 42,43,44
splits = Full, HC-Val, HC-Test
threshold = 0.5
checkpoints = 250,300,350,400
```

不要跑：

```text
TCE-2/3 selection
new checkpoint subset
threshold search
BN tuning
```

### Day 3：threshold-matched / component report

```text
Pd-matched FA
mIoU-matched Pd / FA
FP components
runtime overhead
```

### Day 4：决定是否 blind / external once

只有 F0/F1/F2 全部通过，才运行一次。

### Day 5-10：写论文主线

论文主线：

```text
Trajectory-consensus inference suppresses unstable hard-clutter activations in IRSTD.
```

不是：

```text
single-forward efficient model
best checkpoint selection
early stopping trick
seed selection
```

### Day 11-15：补 ablation / supplement / rebuttal material

Ablation 只作为解释，不作为选择：

```text
OHEM-400
LateSnapshot-ep300 diagnostic
TWA-4 diagnostic
TCE-2 / TCE-3 budget ablation
TCE-4 frozen main
TCSR stopped audit as negative evidence / supplement
```

---

## 22. 论文写法建议

### 22.1 方法标题

可选：

```text
Training-Trajectory Consensus for Hard-Clutter Robust Infrared Small Target Detection
```

或：

```text
Trajectory Consensus Inference for False-Alarm Robust IRSTD
```

### 22.2 核心贡献

```text
1. We identify that hard-clutter false alarms are unstable across late training trajectory checkpoints, while real small targets remain more stable.

2. We propose TCE-4, a training-trajectory consensus inference framework that aggregates late checkpoints from a single MSHNetOHEM training trajectory without extra training data or additional verifier modules.

3. We show that single-forward compression attempts, including weight averaging and sparse reliability distillation, fail under strict gates, indicating that the useful reliability signal is not easily reducible to a single checkpoint.

4. We provide fixed-threshold, threshold-matched, component-level, and multi-seed evidence to demonstrate hard-clutter false-alarm reduction without Pd loss.
```

### 22.3 必须承认的限制

```text
TCE-4 uses 4 inference forwards.
It is more expensive than single-checkpoint MSHNetOHEM.
The paper should present it as a reliability / robustness tradeoff, not as a free efficiency improvement.
```

这不是坏事。只要结果足够稳，可以写成：

```text
For safety-critical IRSTD scenarios where false alarms are costly, trajectory-consensus inference provides a practical robustness tradeoff.
```

---

## 23. 最终结论

当前必须停止：

```text
STOP_TCSR_AT_BANK_AUDIT
```

不要继续：

```text
TCSR Stage 2
TCSR loss / net / train
seed43/44 for TCSR
HC-Test for TCSR
blind/external for TCSR
threshold/lambda/bank threshold rescue
```

下一步唯一有投稿现实性的路线：

```text
FREEZE_TCE4_AS_FINAL_AAAI_CANDIDATE
```

它不是单 forward 方法，但它是目前唯一已经有稳定正信号、能解释 hard-clutter false-alarm reduction、且能在 15 天内完成 final evidence aggregation 和论文写作的主线。

一句话：

> **TCSR-A 的失败意味着“训练时稀疏蒸馏”这条路没有信号；现在不要再开发新结构。把 TCE-4 诚实地冻结为 trajectory-consensus inference 方法，做最终证据聚合、阈值匹配、component 分析和一次 blind/external。**
