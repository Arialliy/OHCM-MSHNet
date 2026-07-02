# OHCM-MSHNet：TWA-OHEM 当前候选的下一步方案与代码修改清单

> 当前仓库：`https://github.com/Arialliy/OHCM-MSHNet`  
> 当前强锚点：MSHNetOHEM  
> 当前最新正向结果：**TWA without BN seed42 Full Gate-C 通过**  
> 当前失败分支：**TWA+BN recalibration 已停止**  
> 当前唯一允许下一步：**Gate-TWA-D：只评估 TWA without BN 的 seed42 HC-Val**  
> 当前禁止：seed43/44、HC-Test、blind、external、新模型训练、BN recalibration 调参、新 verifier / suppression 结构。

---

## 1. 当前状态总结

你已经严格按照 TWA 路线完成 Gate-TWA-A/B/C，并在规定位置停止，没有跑 HC-Val、seed43/44、HC-Test、blind 或 external。

### 1.1 已完成代码

```text
README.md 更新
STOPPED_BRANCHES_SUMMARY.md 新增
utils/branch_status.py 新增
train.py 增加 stopped branch guard 和 --allow_stopped_branch
tools/official/build_twa_checkpoint.py 新增
tools/official/recalibrate_bn.py 新增
tools/official/evaluate_twa_checkpoint.py 新增
tools/official/compare_tce_twa.py 新增

tests/test_branch_status_guard.py 新增
tests/test_twa_checkpoint_average.py 新增
tests/test_bn_recalibration.py 新增
```

### 1.2 验证

```text
py_compile: PASS
pytest: 8 passed
git diff --check: PASS
```

### 1.3 Gate 结果

```text
Gate-TWA-A: PASS
Gate-TWA-B: TWA+BN 不优于 TWA without BN，因此 BN 版本停止
Gate-TWA-C: TWA without BN seed42 Full gate PASS
```

### 1.4 TWA without BN vs OHEM-400 Full

| 指标 | TWA without BN - OHEM-400 |
|---|---:|
| mIoU | +0.00450 |
| Pd | +0.00106 |
| Precision | +0.00345 |
| FA ppm | -2.36695 |
| FP components | -16 |

结论：

```text
TWA without BN 是当前唯一通过 seed42 Full gate 的正向候选。
TWA+BN 已停止，不再调 BN recalibration。
```

---

## 2. 这个结果说明什么？

### 2.1 这是当前项目中少见的正向候选

之前许多路线失败于：

```text
Full mIoU gate failed
Full Pd gate failed
candidate 太少
flat artifact 太多
target top20 太高
selected/OHEM overlap 太高
```

而 TWA without BN 这次在 Full split 上同时做到：

```text
mIoU 上升
Pd 上升
Precision 上升
FA 下降
FP components 下降
```

这说明它没有破坏 MSHNetOHEM evidence anchor。

---

### 2.2 但它还不是 AAAI-ready

当前只证明了：

```text
seed42 Full 保真并正向提升。
```

还没有证明：

```text
HC-Val 有效
threshold-matched 有效
不是某个 single checkpoint 本身更好
三种子稳定
blind/external 有效
```

所以现在不能直接写论文，也不能跑 seed43/44 或 HC-Test。

---

## 3. 当前研究目标有没有变化？

### 3.1 总目标没有变

目标仍然是：

```text
保持 MSHNetOHEM 的真实小目标检测能力
降低复杂背景 false alarms
提升 Precision
不牺牲 Pd
尽量保持单模型、单 forward
```

---

### 3.2 解决路径变了

之前路径：

```text
hard clutter mining
SPS reranking
decoy verifier
background residual verifier
```

均未过完整 gate。

现在路径：

```text
利用 TCE-OHEM 已证明有效的训练轨迹可靠性
尝试将多 checkpoint 的 trajectory signal 压缩到单个 checkpoint
形成 TWA-OHEM 单模型候选
```

---

## 4. 为什么 TWA 值得继续？

### 4.1 TCE 是当前唯一稳定强正信号

TCE-OHEM 之前已经表现出：

```text
Full / HC-Val / HC-Test 三种子 3/3 优于 OHEM-400
hard-split threshold-matched 12/12 通过
```

问题是：

```text
4x inference
多个 checkpoint
创新性像 ensemble
```

TWA 的目标是：

> **把 TCE 的训练轨迹共识压缩成单个模型权重，保留一部分 TCE 收益，同时保持 1x inference。**

---

### 4.2 TWA 避开了之前失败路线的问题

| 失败路线 | 问题 | TWA 如何避开 |
|---|---|---|
| SPS / APF | candidate 稀疏或污染 | 不需要 candidate |
| CDV / ECDV | synthetic decoy artifact | 不合成 decoy |
| BCV / MSCV | verifier 判据不稳定 | 不加 verifier |
| PFR / ERD / CGA | 污染 evidence branch | 不新增 correction head |
| TCE | 4x inference | TWA 为单 checkpoint |

---

## 5. 当前不能做什么？

现在严格禁止：

```text
不要跑 seed43
不要跑 seed44
不要跑 HC-Test
不要跑 blind
不要跑 external
不要调 BN recalibration
不要改模型结构
不要改 loss
不要回到 BCV / SPS / ECDV / ERD
不要尝试新的 verifier / suppression head
```

当前唯一允许：

```text
Gate-TWA-D: seed42 HC-Val evaluation for TWA without BN only
```

---

# 6. 代码修改方案

当前不需要改模型或 loss。  
只需要完善 TWA workflow、gate 工具和状态保护。

---

## 6.1 README 更新当前状态

在 README 顶部更新为：

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Stopped branches:
- OHCM / prototype / full branch
- SPS-OHEM pixel reranking
- APF / component mining
- PFR / ERD / CGA
- CDV / ECDV / MSCV / BCV
- TWA with BN recalibration

Current active candidate:
- TWA-OHEM without BN recalibration
- seed42 Gate-TWA-C Full: PASS
- next allowed gate: Gate-TWA-D HC-Val on seed42 only

Forbidden before Gate-TWA-D:
- seed43 / seed44
- HC-Test
- blind / external
- new model training
- BN recalibration tuning
```

---

## 6.2 更新 `gate_twa_abc_summary.json` 状态字段

文件：

```text
docs/internal/twa/seed42_nudt/gate_twa_abc_summary.json
```

建议增加：

```json
{
  "twa_no_bn_status": "PASS_GATE_TWA_C",
  "twa_bn_status": "STOP_BN_RECALIBRATION",
  "current_candidate": "twa_without_bn",
  "next_allowed_gate": "Gate-TWA-D-HCVal-seed42",
  "forbidden_next_actions": [
    "seed43",
    "seed44",
    "HC-Test",
    "blind",
    "external",
    "BN recalibration tuning",
    "new model training"
  ]
}
```

---

## 6.3 新增 Gate-D 工具

新增文件：

```text
tools/official/check_twa_gate_d_hcval.py
```

### 功能

比较：

```text
TWA without BN seed42 HC-Val
vs
paired OHEM-400 seed42 HC-Val
```

### 通过条件

| 指标 | 要求 |
|---|---:|
| delta_mIoU | ≥ +0.005 |
| delta_FA_ppm | ≤ -10 |
| delta_Precision | ≥ 0 |
| delta_Pd | ≥ 0 |

---

### 代码骨架

```python
import argparse
import json
from pathlib import Path


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(summary, key):
    if key in summary:
        return float(summary[key])
    if "metrics" in summary and key in summary["metrics"]:
        return float(summary["metrics"][key])
    raise KeyError(f"Metric {key} not found in {summary.keys()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohem_summary", required=True)
    parser.add_argument("--twa_summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_delta_miou", type=float, default=0.005)
    parser.add_argument("--min_fa_reduction", type=float, default=10.0)
    parser.add_argument("--min_delta_precision", type=float, default=0.0)
    parser.add_argument("--min_delta_pd", type=float, default=0.0)
    args = parser.parse_args()

    ohem = load_summary(args.ohem_summary)
    twa = load_summary(args.twa_summary)

    ohem_miou = get_metric(ohem, "mIoU")
    twa_miou = get_metric(twa, "mIoU")

    ohem_fa = get_metric(ohem, "FA_ppm")
    twa_fa = get_metric(twa, "FA_ppm")

    ohem_prec = get_metric(ohem, "Precision")
    twa_prec = get_metric(twa, "Precision")

    ohem_pd = get_metric(ohem, "Pd")
    twa_pd = get_metric(twa, "Pd")

    delta_miou = twa_miou - ohem_miou
    delta_fa = twa_fa - ohem_fa
    delta_precision = twa_prec - ohem_prec
    delta_pd = twa_pd - ohem_pd

    gate_pass = (
        delta_miou >= args.min_delta_miou
        and delta_fa <= -args.min_fa_reduction
        and delta_precision >= args.min_delta_precision
        and delta_pd >= args.min_delta_pd
    )

    result = {
        "gate": "Gate-TWA-D",
        "method": "TWA without BN",
        "split": "HC-Val",
        "seed": 42,
        "ohem": {
            "mIoU": ohem_miou,
            "FA_ppm": ohem_fa,
            "Precision": ohem_prec,
            "Pd": ohem_pd,
        },
        "twa": {
            "mIoU": twa_miou,
            "FA_ppm": twa_fa,
            "Precision": twa_prec,
            "Pd": twa_pd,
        },
        "delta": {
            "mIoU": delta_miou,
            "FA_ppm": delta_fa,
            "Precision": delta_precision,
            "Pd": delta_pd,
        },
        "gate_pass": gate_pass,
        "next_allowed_gate": "Gate-TWA-E" if gate_pass else "STOP_TWA_4_NO_BN",
        "forbidden_if_fail": [
            "seed43",
            "seed44",
            "HC-Test",
            "blind",
            "external",
            "new TWA tuning"
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not gate_pass:
        raise SystemExit("Gate-TWA-D failed. Stop TWA without BN.")


if __name__ == "__main__":
    main()
```

---

## 6.4 新增 HC-Val official evaluation script

新增：

```text
scripts/official/eval_twa_seed42_hcval.sh
```

示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}

TWA_CKPT="${ROOT}/docs/internal/twa/seed42_nudt/twa_seed42_250_300_350_400.pth.tar"
OUT_DIR="${ROOT}/docs/internal/twa/seed42_nudt/eval_hcval_twa_no_bn"

python tools/official/evaluate_twa_checkpoint.py \
  --model_name MSHNetOHEM \
  --checkpoint "${TWA_CKPT}" \
  --dataset_name NUDT-SIRST \
  --split hcval \
  --output_dir "${OUT_DIR}" \
  --threshold 0.5
```

注意：

```text
不要默认跑 HC-Test。
不要默认跑 seed43/44。
```

---

## 6.5 新增 Gate-D 一键脚本

新增：

```text
scripts/official/run_twa_gate_d_seed42.sh
```

示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}

bash scripts/official/eval_twa_seed42_hcval.sh

python tools/official/check_twa_gate_d_hcval.py \
  --ohem_summary "${ROOT}/docs/internal/twa/seed42_nudt/eval_hcval_ohem/summary_metrics.json" \
  --twa_summary "${ROOT}/docs/internal/twa/seed42_nudt/eval_hcval_twa_no_bn/summary_metrics.json" \
  --output "${ROOT}/docs/internal/twa/seed42_nudt/gate_twa_d_summary.json"
```

---

## 6.6 给 BN 版本加 stopped status

在 `utils/branch_status.py` 或新增 `utils/twa_status.py`：

```python
STOPPED_TWA_VARIANTS = {
    "twa_bn_recalibrated": (
        "Stopped at Gate-TWA-B: BN recalibration did not improve over TWA without BN."
    ),
}
```

如果工具支持：

```text
--use_bn_recalibration
```

默认必须是 false。

---

## 6.7 新增测试

```text
tests/test_twa_gate_d_checker.py
tests/test_twa_status_schema.py
```

### `test_twa_gate_d_checker.py`

测试内容：

```python
def test_gate_d_pass_when_hcval_improves():
    # delta_mIoU >= 0.005, delta_FA <= -10, precision/pd >= 0
    pass


def test_gate_d_fails_when_pd_drops():
    pass


def test_gate_d_fails_when_fa_not_reduced():
    pass
```

### `test_twa_status_schema.py`

检查：

```python
def test_gate_twa_abc_has_next_allowed_gate():
    assert "next_allowed_gate" in summary
    assert summary["current_candidate"] == "twa_without_bn"
```

---

# 7. Gate-TWA-D：下一步运行

## 7.1 只允许运行

```text
TWA without BN
seed42
HC-Val
fixed threshold 0.5
```

不允许：

```text
TWA+BN
seed43/44
HC-Test
blind/external
new checkpoint combinations
```

---

## 7.2 Gate-D 通过条件

| 指标 | 要求 |
|---|---:|
| HC-Val mIoU | OHEM + ≥ 0.005 |
| HC-Val FA | OHEM - ≥ 10 ppm |
| HC-Val Precision | ≥ OHEM |
| HC-Val Pd | ≥ OHEM |

---

## 7.3 Gate-D 失败条件

任一失败即停止：

```text
HC-Val mIoU 不提升
HC-Val FA 不下降至少 10 ppm
Precision 下降
Pd 下降
```

如果失败：

```text
STOP_TWA_4_NO_BN
```

不要跑 seed43/44，不要跑 HC-Test。

---

# 8. Gate-D 通过后：Gate-TWA-E 机制比较

只有 Gate-D 通过，才进入 Gate-E。

---

## 8.1 必须比较

```text
OHEM-400
Best single late checkpoint
TCE-4
TWA-2
TWA-3
TWA-4 without BN
TWA-4 with BN as stopped control only
```

---

## 8.2 为什么要比较 best single late checkpoint？

如果某个单 checkpoint 本身就优于 TWA，那么 TWA 不是 trajectory compression，而只是平均偶然有效。

所以必须证明：

```text
TWA-4 >= best single late checkpoint
```

至少在 HC-Val 上不差。

---

## 8.3 TWA-2/3/4 允许范围

只允许预注册组合：

```text
TWA-2: 350,400
TWA-3: 300,350,400
TWA-4: 250,300,350,400
```

不要穷举搜索所有 checkpoint combination。

---

## 8.4 Gate-E 通过条件

| 比较 | 要求 |
|---|---|
| TWA-4 vs OHEM-400 | Full 不退化，HC-Val 提升 |
| TWA-4 vs best single late | HC-Val 不低于 best single |
| TWA-4 vs TCE-4 | 获得 TCE hard-split 收益的 30%–50% |
| TWA-2/3/4 | 趋势合理，不是随机挑中 |

---

# 9. Gate-E 通过后：三种子

只有 Gate-E 通过，才跑：

```text
seed43
seed44
```

要求：

| Split | 要求 |
|---|---|
| Full | 3/3 不退化 |
| HC-Val | 至少 2/3 提升，最好 3/3 |
| mean | HC-Val mIoU / FA / Precision 有正收益 |
| Pd | 不下降 |

---

# 10. Threshold-matched 与 FP component

三种子通过后再做：

```text
fixed 0.5
Pd-matched
mIoU-matched
FP component analysis
```

要求：

```text
Pd-matched FA 下降
mIoU-matched 不损 Pd
detached far-FP / FP components 下降
不是只靠阈值校准
```

---

# 11. Blind / external

最后只评估一次。

```text
方法冻结
checkpoint 组合冻结
是否使用 BN 冻结
threshold 规则冻结
所有 gate 通过
```

---

# 12. 如果 Gate-D 失败怎么办？

如果 seed42 HC-Val 失败：

## 12.1 不允许

```text
不跑 seed43/44
不跑 HC-Test
不跑 blind/external
不调 BN
不继续训练
不新建 verifier
```

## 12.2 只允许一个预注册 no-training audit

可以比较：

```text
TWA-2: 350,400
TWA-3: 300,350,400
TWA-4: 250,300,350,400
```

每个必须：

```text
先过 Full
再看 HC-Val
```

如果 TWA-2/3/4 都失败：

```text
STOP_TWA_WEIGHT_SPACE_COMPRESSION
```

此时保留 TCE 作为 diagnostic oracle，不再做单模型 compression。

---

# 13. 当前是否能投 AAAI？

现在还不能。

但 TWA without BN 是目前最值得继续的候选，因为：

```text
它不需要新模型结构
它不需要候选挖掘
它不需要 synthetic decoy
它不需要 verifier
它保持 1x inference
它 seed42 Full 明确优于 OHEM
```

如果后续满足：

```text
Gate-D HC-Val PASS
Gate-E 机制比较 PASS
seed43/44 paired PASS
threshold-matched PASS
blind/external PASS
```

则可以作为 AAAI 候选。

但创新性风险仍在：

```text
TWA 接近 SWA / model soup
```

论文要定位为：

```text
training-trajectory reliability compression for IRSTD
```

而不是简单说：

```text
we average checkpoints
```

---

# 14. 当前最简执行清单

```text
[1] 更新 README 当前 TWA 状态
[2] 给 gate_twa_abc_summary.json 增加 status / next_allowed_gate
[3] 标记 TWA+BN stopped
[4] 新增 check_twa_gate_d_hcval.py
[5] 新增 eval_twa_seed42_hcval.sh
[6] 新增 run_twa_gate_d_seed42.sh
[7] 新增 tests/test_twa_gate_d_checker.py
[8] 只跑 TWA without BN seed42 HC-Val
[9] Gate-D PASS → Gate-E
[10] Gate-D FAIL → 停止 TWA-4 no-BN，只做预注册 TWA-2/3 no-training audit
```

---

# 15. 最终结论

当前正式决策应写成：

```text
Decision: PROCEED_TWA_NO_BN_TO_GATE_D

TWA without BN passed seed42 Full Gate-TWA-C:
mIoU +0.00450
Pd +0.00106
Precision +0.00345
FA -2.36695 ppm
FP components -16

TWA+BN failed Gate-TWA-B and is stopped.

The only allowed next step is:
Gate-TWA-D HC-Val evaluation for TWA without BN on seed42.

Forbidden:
seed43/44, HC-Test, blind/external, BN tuning, new model training.
```

一句话：

> **TWA without BN 是当前唯一通过 Full gate 的正向候选。现在不要改模型，不要调 BN，不要跑三种子；只推进 seed42 HC-Val Gate。**
