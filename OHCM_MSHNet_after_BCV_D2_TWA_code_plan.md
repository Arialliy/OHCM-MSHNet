
# OHCM-MSHNet：Gate-D2 后的代码修改与下一步推进方案

> 当前仓库：`https://github.com/Arialliy/OHCM-MSHNet`  
> 当前强锚点：MSHNetOHEM  
> 最新停止点：BCV Gate-D2 失败  
> 当前结论：**停止单帧 false-alarm suppression / verifier / candidate mining 分支；冻结已验证代码；转向训练轨迹压缩路线（TWA-OHEM）或项目 pivot。**  
> 新推荐方向：**TWA-OHEM：Trajectory Weight Averaged MSHNetOHEM**  
> 中文：**训练轨迹权重平均的 MSHNetOHEM**  
> 核心目标：把当前唯一稳定正信号 TCE-OHEM 尽量压缩成单模型、单次 forward。

---

## 0. 当前事实与最新结论

你已经严格完成 BCV Gate-D2，并按停止条件停止。

Gate-D2 关键结果：

| 指标 | 结果 |
|---|---:|
| far_fp_component_count | 29 |
| component recall 99% 下 component rate | 0.0345 |
| component recall 99% 下 pixel mass rate | 0.0144 |
| component recall 99% 下 confidence mass rate | 0.0146 |
| target pixel recall 99.5% 下 pixel mass rate | 0.0 |
| target pixel recall 99.5% 下 confidence mass rate | 0.0 |
| overall_decision | STOP_BCV |

这说明：

> BCV residual / shape family 即使从 component count 扩展到 pixel/confidence mass，也不能在高 target protection 下安全压制足够 OHEM far-FP。

因此必须停止：

```text
BCV residual / shape calibration
BCV learned verifier
BCV deterministic formula
BCV Gate-E
BCV training
HC-Val / seed43/44 / HC-Test / blind / external for BCV
```

---

# 1. 当前代码中哪些部分已经通过，应冻结不改？

## 1.1 冻结 MSHNetOHEM anchor

不要改：

```text
MSHNetOHEM 主检测分支
MSHNet final fused head official path
foreground_probability()
direct/export parity
target-level Pd / component-level evaluation
threshold curve
OHEM checkpoint / export / eval pipeline
```

理由：

> MSHNetOHEM 是当前唯一稳定强基线，也是所有后续方法必须保护的 inference anchor。

---

## 1.2 冻结通过的诊断工具

以下工具是有价值的 negative evidence，不应删除：

```text
tools/official/check_bcv_gate_a.py
tools/official/check_bcv_gate_b.py
tools/official/check_bcv_gate_c_fp_residual.py
tools/official/check_bcv_gate_d_residual_shape.py
tools/official/check_bcv_gate_d2_mass_shape.py
utils/residual_shape_features.py
```

这些工具证明：

```text
BCV-A beta=0 等价 OHEM：PASS
BCV-B target vs ordinary background residual：PASS
BCV-C residual magnitude against OHEM FP：FAIL
BCV-D residual shape against OHEM FP：component suppressibility FAIL
BCV-D2 residual/shape mass suppressibility：FAIL
```

这不是坏代码，而是系统性证据。

---

## 1.3 冻结失败分支，不再训练

以下模型不再作为 official training candidate：

```text
OHCMMSHNet
OHCMMSHNetFull
MSHNetSPSOHEM 当前 pixel reranking 路线
PFRMSHNet
ERDMSHNet
ERDMSHNetV3
CDVMSHNet
ECDVMSHNet
MSCVMSHNet
BCVMSHNet residual/shape family
APF / PCAR / TSR / TNC 相关 legacy 路线
```

可以保留代码用于复现实验与失败分析，但不能再默认训练。

---

# 2. 当前目标是否变化？

## 2.1 总目标不变

总目标仍然是：

```text
在不损失 MSHNetOHEM 目标检测能力的前提下，
降低复杂背景 false alarms，
保持单模型或可接受推理成本，
并有严格机制解释和三种子 / blind 证据。
```

---

## 2.2 解决路径必须变化

过去路线是：

```text
mining hard clutter
learning clutterness
self-perturbation reranking
synthetic decoy verifier
multi-scale verifier
background residual verifier
```

这些都已经被 gate 证明没有形成可投稿主方法。

因此现在不能再继续：

```text
设计第 N 个 suppression head
设计第 N 个 verifier
设计第 N 个 candidate mining rule
```

而应转向已有强正信号：

```text
TCE-OHEM
```

---

# 3. 为什么下一步应转向 TWA-OHEM？

## 3.1 当前唯一稳定正信号是 TCE-OHEM

此前 TCE-OHEM 已经显示：

```text
Full / HC-Val / HC-Test 三种子 3/3 优于 OHEM-400
hard-split threshold-matched 12/12 通过
```

它的缺点是：

```text
4 个 checkpoint 推理
约 4× FLOPs / latency
创新性容易被认为是 ensemble
```

但它证明了一个关键事实：

> 训练轨迹中存在可利用的可靠性信号。

---

## 3.2 TWA 的目标

TWA-OHEM 的目标是：

> **把 TCE 的训练轨迹共识压缩为单个 MSHNetOHEM 权重模型。**

也就是做：

```text
checkpoint-space compression
```

而不是继续做：

```text
new verifier branch
new candidate mining
new false alarm suppression head
```

---

## 3.3 为什么它比继续设计 verifier 更合理？

| 路线 | 当前证据 |
|---|---|
| verifier / suppression | 多次 Full gate 或机制 gate 失败 |
| candidate mining | 稀疏或污染 |
| synthetic decoy | artifact |
| residual / shape | 不能安全压制 FP mass |
| TCE | 当前唯一稳定正结果 |
| TWA | 尝试保留 TCE 好处，但保持 1× 推理 |

TWA 不保证一定成功，但它是当前最合理的下一步，因为它直接利用唯一正证据。

---

# 4. 新路线：TWA-OHEM

## 4.1 方法名

```text
TWA-OHEM
Trajectory Weight Averaged MSHNetOHEM
```

也可以写为：

```text
Trajectory-Compressed OHEM
```

---

## 4.2 核心公式

给定同一训练轨迹上的 late checkpoints：

\[
\Theta =
\{
\theta_{250}, \theta_{300}, \theta_{350}, \theta_{400}
\}
\]

TCE 做的是概率平均：

\[
P_{\mathrm{TCE}}
=
\frac{1}{K}\sum_k P_{\theta_k}
\]

TWA 做的是权重平均：

\[
\theta_{\mathrm{TWA}}
=
\frac{1}{K}\sum_k \theta_k
\]

然后单模型推理：

\[
P_{\mathrm{TWA}}
=
\sigma(f_{\theta_{\mathrm{TWA}}}(I))
\]

---

## 4.3 与 TCE 的关系

| 项目 | TCE-OHEM | TWA-OHEM |
|---|---|---|
| 聚合空间 | probability space | weight space |
| 推理成本 | 4× | 1× |
| 模型数 | 多 checkpoint | 单 checkpoint |
| 当前证据 | 已有强结果 | 待验证 |
| 目标 | oracle / upper bound | 可部署候选 |

---

## 4.4 与 SWA / Model Soup 的关系

TWA 与 SWA / model soup 接近，所以创新性不能夸大。

正确定位：

> TWA 不是全新算法，而是将 TCE 诊断出的 trajectory reliability 压缩为单模型的一种工程化验证路线。

如果 TWA 成功，论文可以写成：

```text
training-trajectory reliability analysis
+ checkpoint consensus oracle
+ single-model trajectory compression
+ hard-split reliability evaluation
```

如果 TWA 失败，则说明：

```text
TCE 收益主要来自 prediction-space ensemble，而不能简单转移到单模型。
```

---

# 5. 代码修改总览

下一步代码不要再改 `loss.py` 或加新 head。  
只新增 official trajectory tools。

新增：

```text
tools/official/build_twa_checkpoint.py
tools/official/recalibrate_bn.py
tools/official/evaluate_twa_checkpoint.py
tools/official/compare_tce_twa.py
tests/test_twa_checkpoint_average.py
tests/test_bn_recalibration.py
docs/internal/twa/
```

修改：

```text
README.md
train.py branch guard
model registry stopped branch guard
```

不修改：

```text
MSHNetOHEM
foreground_probability.py
loss.py
BCV/CDV/ECDV/MSCV diagnostic tools
direct/export parity
```

---

# 6. 代码修改 1：新增 stopped branch guard

## 6.1 新增文件

```text
utils/branch_status.py
```

内容：

```python
STOPPED_BRANCHES = {
    "PFRMSHNet": "Stopped after Full gate failure.",
    "ERDMSHNet": "Stopped after HC-Val / Full reliability failure.",
    "ERDMSHNetV3": "Stopped after Full gate failure.",
    "CDVMSHNet": "Stopped after Gate-B flat-artifact failure.",
    "ECDVMSHNet": "Stopped after Gate-B flat-artifact failure.",
    "MSCVMSHNet": "Stopped after Gate-B candidate/target-top20 failure.",
    "BCVMSHNet": "Stopped after Gate-D2: residual/shape suppressibility insufficient.",
    "OHCMMSHNetFull": "Stopped after full/prototype branch failure.",
}


def assert_branch_allowed(model_name: str, allow_stopped_branch: bool = False):
    if model_name in STOPPED_BRANCHES and not allow_stopped_branch:
        reason = STOPPED_BRANCHES[model_name]
        raise RuntimeError(
            f"{model_name} is a stopped diagnostic branch. "
            f"Reason: {reason} "
            "Use --allow_stopped_branch only for diagnostic reproduction."
        )
```

---

## 6.2 修改 `train.py`

新增参数：

```python
parser.add_argument(
    "--allow_stopped_branch",
    action="store_true",
    default=False,
)
```

训练前：

```python
from utils.branch_status import assert_branch_allowed

assert_branch_allowed(
    opt.model_names[0],
    allow_stopped_branch=opt.allow_stopped_branch,
)
```

目的：

> 防止误启动已经 gate-stopped 的分支。

---

## 6.3 测试

新增：

```text
tests/test_branch_status_guard.py
```

测试：

```python
def test_stopped_branch_blocked():
    with pytest.raises(RuntimeError):
        assert_branch_allowed("BCVMSHNet", allow_stopped_branch=False)


def test_stopped_branch_allowed_for_repro():
    assert_branch_allowed("BCVMSHNet", allow_stopped_branch=True)
```

---

# 7. 代码修改 2：新增 TWA checkpoint averaging

## 7.1 新增文件

```text
tools/official/build_twa_checkpoint.py
```

---

## 7.2 功能

输入多个 checkpoints：

```bash
python tools/official/build_twa_checkpoint.py \
  --checkpoints ckpt250.pth.tar ckpt300.pth.tar ckpt350.pth.tar ckpt400.pth.tar \
  --output checkpoints/twa_seed42_250_300_350_400.pth.tar \
  --model_key state_dict
```

输出一个单 checkpoint：

```text
twa_seed42_250_300_350_400.pth.tar
```

---

## 7.3 平均规则

只平均 floating tensors：

```python
if tensor.is_floating_point():
    avg = sum(tensors) / len(tensors)
else:
    use final checkpoint tensor
```

需要检查：

```text
所有 checkpoint key 完全一致
所有 tensor shape 完全一致
```

---

## 7.4 代码骨架

```python
import argparse
import torch
from collections import OrderedDict


def load_state(path, key="state_dict"):
    ckpt = torch.load(path, map_location="cpu")
    if key in ckpt:
        return ckpt[key], ckpt
    return ckpt, ckpt


def average_states(states):
    keys = list(states[0].keys())
    for s in states[1:]:
        if list(s.keys()) != keys:
            raise ValueError("Checkpoint keys do not match.")

    avg = OrderedDict()
    for k in keys:
        vals = [s[k] for s in states]
        shapes = [tuple(v.shape) for v in vals if torch.is_tensor(v)]
        if len(set(shapes)) > 1:
            raise ValueError(f"Shape mismatch at {k}: {shapes}")

        if torch.is_tensor(vals[0]) and vals[0].is_floating_point():
            avg[k] = sum(v.float() for v in vals) / len(vals)
            avg[k] = avg[k].to(dtype=vals[0].dtype)
        else:
            avg[k] = vals[-1]
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_key", default="state_dict")
    args = parser.parse_args()

    states = []
    meta = []
    for p in args.checkpoints:
        state, ckpt = load_state(p, args.model_key)
        states.append(state)
        meta.append({"path": p, "epoch": ckpt.get("epoch", None)})

    avg_state = average_states(states)

    out = {
        "state_dict": avg_state,
        "twa_meta": {
            "checkpoints": meta,
            "num_checkpoints": len(args.checkpoints),
            "method": "uniform_weight_average",
        },
    }

    torch.save(out, args.output)


if __name__ == "__main__":
    main()
```

---

# 8. 代码修改 3：BN recalibration

MSHNet 使用 BatchNorm。权重平均后，BN running stats 可能不匹配。

必须新增 BN recalibration。

---

## 8.1 新增文件

```text
tools/official/recalibrate_bn.py
```

---

## 8.2 功能

加载 TWA checkpoint，用 train split 做若干 forward，更新 BN stats：

```text
no gradients
model.train()
only BN running_mean / running_var updates
不保存 optimizer
不看 label 指标
不看 test
```

---

## 8.3 命令

```bash
python tools/official/recalibrate_bn.py \
  --model_name MSHNetOHEM \
  --dataset_name NUDT-SIRST \
  --checkpoint checkpoints/twa_seed42_250_300_350_400.pth.tar \
  --output checkpoints/twa_seed42_250_300_350_400_bn.pth.tar \
  --num_batches 200
```

---

## 8.4 关键点

BN recalibration 只能使用：

```text
train split
```

不能用：

```text
test
HC-Test
blind
external
```

---

# 9. 代码修改 4：TWA evaluation wrapper

新增：

```text
tools/official/evaluate_twa_checkpoint.py
```

其实可以复用现有 official evaluate/export，只是提供标准入口。

命令：

```bash
python tools/official/evaluate_twa_checkpoint.py \
  --model_name MSHNetOHEM \
  --checkpoint checkpoints/twa_seed42_250_300_350_400_bn.pth.tar \
  --dataset_name NUDT-SIRST \
  --split full \
  --output_dir results/official/TWAOHEM/seed42/full
```

要求输出：

```text
summary_metrics.json
threshold_curve.csv
component_fp_analysis.csv
export parity summary
```

---

# 10. 代码修改 5：TCE vs TWA 对比工具

新增：

```text
tools/official/compare_tce_twa.py
```

统计：

```text
OHEM-400
TCE-4
TWA
TWA+BN
```

输出：

```text
Full metrics
HC-Val metrics
threshold-matched
component FP
per-image delta
```

---

# 11. TWA Gate 流程

## Gate-TWA-A：checkpoint compatibility

通过条件：

```text
all checkpoint keys match
all shapes match
floating tensors averaged
non-floating tensors handled
TWA checkpoint can load into MSHNetOHEM
```

失败则停止。

---

## Gate-TWA-B：BN recalibration sanity

评估：

```text
TWA without BN recalibration
TWA with BN recalibration
```

通过条件：

```text
TWA+BN >= TWA without BN
Full metrics not catastrophic
```

如果 BN recalibration 导致严重退化，停止 TWA。

---

## Gate-TWA-C：Full gate seed42

TWA seed42 必须满足：

| 指标 | 要求 |
|---|---:|
| Full mIoU | ≥ OHEM - 0.001 |
| Full Pd | ≥ OHEM |
| Full Precision | ≥ OHEM |
| Full FA | ≤ OHEM + 2 ppm |

失败则停止，不跑 HC-Val。

---

## Gate-TWA-D：HC-Val seed42

只有 Full 过后才跑。

最低线：

| 指标 | 要求 |
|---|---:|
| HC-Val mIoU | OHEM + ≥ 0.005 |
| HC-Val FA | OHEM - ≥ 10 ppm |
| HC-Val Precision | 不低于 OHEM |
| HC-Val Pd | 不下降 |

这只是可行性线。

---

## Gate-TWA-E：机制比较

比较：

```text
OHEM-400
Best single late checkpoint
TCE-4
TWA-2
TWA-3
TWA-4
TWA-4 + BN
```

TWA 必须证明：

```text
优于 OHEM-400
接近 TCE 至少 30%～50% hard-split 收益
不是只等于某个 best checkpoint
```

---

## Gate-TWA-F：三种子 paired

若 seed42 通过，再跑：

```text
seed43
seed44
```

要求：

```text
Full 3/3 不退化
HC-Val 至少 2/3 提升
mean HC-Val 有正收益
```

---

## Gate-TWA-G：threshold-matched

必须做：

```text
fixed 0.5
Pd-matched
mIoU-matched
```

通过条件：

```text
HC-Val Pd-matched FA 下降
Full matched 不伤 Pd
```

---

## Gate-TWA-H：blind / external

最后才做。

---

# 12. TWA 如果通过，能不能投 AAAI？

TWA 的创新性中等，不如真正新结构强，但它有一个优势：

```text
它利用了当前唯一稳定正信号 TCE。
```

如果 TWA 达到：

```text
单模型 1× inference
Full 不退化
HC-Val / blind 有稳定收益
TCE 收益转化 50% 以上
机制分析清楚
```

可以写成：

> **Training-Trajectory Reliability Compression for Infrared Small Target Detection**

但必须承认它与 SWA / model soup 接近。  
论文要强调：

```text
不是简单平均权重，
而是先通过 TCE 诊断证明 trajectory reliability，
再验证 weight-space compression 是否能保留 hard-split reliability。
```

---

# 13. 如果 TWA 失败

如果 TWA 也失败，则说明：

```text
TCE 的收益依赖 prediction-space ensemble，
不能压缩到 single model。
```

这时不建议继续单帧 false-alarm suppression。

应该正式转向：

```text
cross-dataset robustness
multi-frame temporal consistency
failure analysis benchmark
```

---

# 14. 当前代码修改顺序

请严格按顺序：

```text
[1] 更新 README 当前状态：STOP single-frame suppression branches
[2] 新增 utils/branch_status.py
[3] train.py 加 stopped branch guard
[4] 新增 STOPPED_BRANCHES_SUMMARY.md
[5] 新增 build_twa_checkpoint.py
[6] 新增 recalibrate_bn.py
[7] 新增 evaluate_twa_checkpoint.py
[8] 新增 compare_tce_twa.py
[9] 新增 TWA tests
[10] 只跑 Gate-TWA-A/B/C
```

不要跑：

```text
HC-Val
seed43/44
HC-Test
blind/external
```

直到 TWA seed42 Full gate 通过。

---

# 15. 最终总结

当前不是“代码怎么继续改 verifier”的问题。  
当前是：

```text
所有单帧 trainable false-alarm suppression 信息源已经被系统性 gate 证伪。
```

因此：

> **冻结所有已通过诊断代码，阻断失败分支训练，把唯一稳定正信号 TCE 尝试压缩为单模型 TWA。**

如果 TWA 成功，它是当前最实际的 AAAI 候选。  
如果 TWA 失败，项目应转向跨域 / 时序 / failure analysis，而不是继续设计新的 suppression head。
