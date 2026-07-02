# SPS-OHEM / OHCM-MSHNet AAAI 合并版路线图

> 代码目录：`/home/ly/AAAI/OHCM-MSHNet-main`  
> 基础模型：`MSHNet`  
> 强基线：`MSHNetOHEM`  
> 当前候选方法：`SPS-OHEM`  
> 当前结论：**HOLD；idea 可行，但当前还不是 AAAI-ready。**  
> 执行原则：**不新建分支，不切换代码目录；所有修复、实验、脚本和记录都围绕 `/home/ly/AAAI/OHCM-MSHNet-main` 原地整理。**

---

## 0. 一句话结论

当前代码已经是一个比较成熟的 **SPS-OHEM 研究原型**，但还不是可以直接投稿 AAAI 的最终仓库。

现在最重要的不是继续包装已有结果，也不是继续扫 `alpha / lambda / epoch`，而是严格按下面路线推进：

```text
代码语义修复
→ Gate0 机制诊断
→ seed42 机制消融
→ 三种子 paired gate
→ threshold-matched 与 FP component 分析
→ blind / external final test
→ AAAI 写作与 artifact 整理
```

只有当 SPS 明确优于：

```text
two-view OHEM
global consistency
confidence-only far OHEM
SPS without far mask
```

并且在 **单模型 1× inference** 下稳定获得 hard split 收益，才具备 AAAI 投稿价值。

---

## 1. 当前定位

### 1.1 项目主线

最终目标不是“在 MSHNet 上再加一个 loss”，而是：

> 在 MSHNetOHEM 强基线之上，提出一种稳定性指导的难负样本挖掘机制，在不牺牲真实小目标检测能力的前提下，稳定降低复杂背景中的 detached far-FP，并保持单模型、单次 forward 推理成本。

也就是：

```text
Full split 保真
+ hard split 降 FA
+ Pd 不掉
+ Precision 提升
+ 单模型 1× inference
+ 机制消融能证明 SPS 不是普通 two-view / consistency
```

### 1.2 最终方法名

推荐最终方法名：

```text
SPS-OHEM
Self-Perturbation Stability-guided Online Hard Example Mining
```

### 1.3 投稿状态

```text
当前状态：HOLD
当前 idea：可继续
当前代码：可作为研究原型继续推进
当前 AAAI：不建议直接投
```

AAAI-ready 的核心标准不是 fixed threshold 0.5 下主表好看，而是：

```text
机制解释成立
paired seed 稳定
threshold-matched 成立
far-FP component 真的下降
blind / external 不崩
artifact 可复现
```

---

## 2. 为什么这个 idea 值得继续

### 2.1 MSHNet 给出的启发

MSHNet 的核心启发不是“换更复杂 backbone”，而是：

> IRSTD 中目标极小，常规 IoU / Dice 对尺度和位置误差不够敏感，因此需要更精确的 scale-location-sensitive supervision。

这说明 IRSTD 的性能提升不一定来自堆网络，而可能来自更精确地定义优化目标。

对当前项目的启发是：

```text
强模型已经能产生目标响应；
下一步更重要的问题是：
哪些高响应是可信目标？
哪些高响应是偶然虚警？
```

### 2.2 OHEM 给出的启发

当前三种子 Full split 历史结果：

| 方法 | Full mIoU | Full Pd | Full FA ppm | Full Precision | Full F1 |
|---|---:|---:|---:|---:|---:|
| MSHNet | 0.8066 | 0.9767 | 70.91 | 0.8912 | 0.8927 |
| MSHNetOHEM | 0.8320 | 0.9778 | 62.31 | 0.9050 | 0.9082 |

MSHNet → MSHNetOHEM：

```text
mIoU 约 +0.0254
FA 约 -8.60 ppm
Precision 约 +0.0138
```

结论：

> hard-example selection 是当前最有效的变量之一。

因此新方法必须以 `MSHNetOHEM` 为强基线，而不是只超过原始 MSHNet。

### 2.3 早期 OHCM / hard-clutter bank 路线的问题

早期假设是：

> 主要剩余错误来自远背景 hard clutter，例如云边缘、海杂波、热噪声、地物亮斑。

但后续实验显示：

```text
OHCM-light 单 seed 有亮点，但三种子不稳定；
prototype / full 分支导致训练轨迹不稳定；
TSR-OHEM 虽然提升 Full mIoU，但 hard split FA 恶化；
PCAR / persistent miner 受 bank precision 和 flat candidate 污染；
Oracle bank 审核后真正 target-like clutter 数量太少；
推理期 inhibition 容易改变概率校准，导致 fixed threshold 与 threshold-matched 不一致。
```

因此当前结论是：

> 直接定义 hard clutter 并显式监督它，不稳定。

### 2.4 TCE-OHEM 给出的正证据

TCE-OHEM 使用同一次训练后期多个 checkpoint：

```text
epoch 250 / 300 / 350 / 400
```

对前景概率取平均：

```text
P_TCE(x) = 1/K * sum_k P_k(x)
```

当前 TCE 相对 OHEM 的收益：

| Split | TCE ΔmIoU | TCE ΔFA |
|---|---:|---:|
| Full | +0.0028 | -3.13 ppm |
| HC-Val | +0.0179 | -38.15 ppm |
| HC-Test | +0.0147 | -25.89 ppm |

并且：

```text
Full / HC-Val / HC-Test 三种子 3/3 优于 OHEM-400
hard split threshold-matched 12/12 通过
```

TCE 证明了一件事：

> 稳定性信号是有价值的。真实目标在多个训练状态下更稳定，部分 detached far-FP 更不稳定。

但 TCE 不能作为最终方法，因为：

```text
4× inference cost
checkpoint ensemble 创新性弱
容易被认为是 snapshot ensemble trick
```

所以 SPS-OHEM 的目标是：

> 用训练期自扰动稳定性近似 TCE 的稳定性信号，并把它转化为单模型 OHEM 难负样本排序机制。

---

## 3. 正式方法定义

### 3.1 双视图输入

同一张图生成两个标签保持视图：

```text
I^w = T_w(I)
I^p = T_p(I)
```

同一个 MSHNetOHEM 共享参数前向：

```text
P^w = sigmoid(f_theta(I^w))
P^p = sigmoid(f_theta(I^p))
```

对扰动视图逆变换回弱视图坐标：

```text
P_tilde^p = T_p^{-1}(P^p)
```

定义扰动不稳定性：

```text
U_i = |P_i^w - P_tilde_i^p|
```

定义远背景安全掩膜：

```text
M_far = 1 - Dilate(Y, r)
```

背景负样本难度：

```text
loss_i^- = BCEWithLogits(z_i, 0)
```

最终 SPS 排序分数：

```text
S_i = M_far,i * loss_i^- * (1 + alpha * U_hat_i)
```

选择与 OHEM 相同数量的负样本：

```text
Omega_SPS^- = TopK(S_i, K_OHEM)
```

核心要求：

> SPS 不增加 OHEM 负样本预算，只改变 hard-negative ranking。

### 3.2 论文中应该强调的贡献

建议写成三点：

```text
1. 识别强 IRSTD 模型中的 residual failure mode：detached far-FP hard clutter；
2. 证明 far-FP 与 true target 在 self-perturbation stability 上具有可分性；
3. 提出 fixed-budget SPS-OHEM，把稳定性用于 hard-negative reranking，在不增加推理成本的情况下减少 far-FP。
```

不要写成：

```text
我们在 MSHNet 上加了一个新的 loss。
```

这个表述太弱，且容易被认为只是工程 trick。

---

## 4. 当前代码状态

### 4.1 已经做得好的地方

| 项目 | 状态 |
|---|---|
| README 已说明 SPS 是当前主线 | PASS |
| `foreground_probability()` 统一概率语义 | PASS |
| direct/export parity gate | PASS |
| export probs / logits / masks / threshold curve | PASS |
| MSHNet direct/export 默认 final head | 基本 PASS |
| SPS batch-concat 双视图 forward | PASS |
| PixelRecall 不再命名为 Pd | PASS |
| checkpoint metadata 增强 | PASS |
| unknown dataset normalization 改为 train-only | PASS |
| 基础测试 `test_sps_core_semantics.py` 通过 | PASS |

### 4.2 当前主要 blocker

| 等级 | 问题 | 影响 |
|---|---|---|
| P0 | 当前 SPS 三种子 AAAI gate 未通过 | 不能投稿 |
| P0 | 机制消融未证明 SPS 优于 two-view / global consistency / no-far-mask | 创新性不足 |
| P0 | target-contrast 诊断工具存在 selected score 语义 bug | 后续 Gate0 不可信 |
| P1 | legacy 工具仍有 output0 / sigmoid 旧逻辑 | 未来误用会污染结果 |
| P1 | official script 默认设置与当前 e40 candidate 不一致 | artifact reviewer 复现困难 |
| P1 | `train.py` 默认模型仍是 HCNet | 默认入口不符合论文主线 |
| P1 | 缺少 `requirements.txt` | artifact 不完整 |
| P1 | 根目录内部文档和 legacy 工具过多 | 匿名投稿仓库不干净 |

---

## 5. 总体执行路线

严格按下面流程执行：

```text
Step 0  当前目录原地清理与流程冻结
Step 1  代码语义 P0 修复
Step 2  OHEM 强基线复现
Step 3  SPS 机制 Gate0：只诊断，不训练
Step 4  候选机制设计：target-contrast / region-level SPS
Step 5  单 seed 小规模训练
Step 6  机制消融对照
Step 7  三种子 paired gate
Step 8  threshold-matched 与 FP component 分析
Step 9  blind / external final test
Step 10 AAAI 写作与 artifact 整理
```

优先级必须是：

```text
先修代码语义
再验证机制信号
再跑 seed42
再跑机制对照
最后才跑三种子和 blind/external
```

不要一开始就三种子大跑。

---

## 6. Step 0：当前目录原地清理与流程冻结

### 6.1 工作目录

所有操作默认在：

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
```

明确要求：

```text
不创建新分支
不切换到其他代码目录
不把本轮路线拆到另一个仓库
不让 OHCM / PCAR / TSR / prototype 继续作为默认入口
```

如果需要保留当前状态，只做轻量备份或记录：

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
pwd
git status --short > docs/internal/git_status_before_sps_cleanup.txt 2>/dev/null || true
```

如果当前仓库不是 git 仓库，就只保留文件列表：

```bash
find . -maxdepth 2 -type f | sort > docs/internal/filelist_before_sps_cleanup.txt
```

### 6.2 建议目录结构

统一成一条主线：

```text
configs/
model/
loss.py
train.py
tools/official/
tools/legacy/
tests/
docs/internal/
legacy/
requirements.txt
README.md
```

说明：

```text
tools/official/ 只放 OHEM / SPS / TCE / eval 主线工具；
tools/legacy/ 放 PCAR / OHCM / TSR / prototype / old step scripts；
docs/internal/ 放 audit、gate log、内部实验记录；
README.md 只讲 MSHNetOHEM 与 SPS-OHEM 主线。
```

如果当前已有 `scripts/official/`，二选一：

```text
要么全部迁到 tools/official/
要么保留 scripts/official/ 并删除 tools/official/ 中重复入口
```

不要同时维护两套 official scripts。

### 6.3 继续条件

| 检查项 | 必须达到 |
|---|---|
| README 默认主线 | SPS-OHEM |
| official scripts | 只包含 OHEM / SPS / TCE / eval |
| legacy 代码 | 不在默认入口调用 |
| requirements.txt | 存在 |
| 默认训练模型 | MSHNetOHEM，或 README 所有命令显式指定 model |
| `python -m py_compile` | 全部通过 |

### 6.4 停止条件

如果 README 和官方脚本仍然混合 OHCM / PCAR / TNC / TSR 作为主线，不进入后续实验。

---

## 7. Step 1：代码语义 P0 修复

### 7.1 MSHNet final head 一致性

所有 official 路径必须使用：

```text
final fused head
```

不得默认使用：

```text
output0
```

检查：

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
rg "output0|mshnet_export_head|resolve_mshnet_head"
```

继续条件：

| 项目 | 要求 |
|---|---|
| direct eval | default final |
| export eval | default final |
| Net.forward eval | default final |
| legacy output0 | 必须显式 `--allow_legacy_output0` |

### 7.2 foreground probability 统一

所有 official 评估与导出必须调用：

```python
foreground_probability()
```

检查：

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
rg "torch.sigmoid|softmax|1 -|foreground_probability"
```

继续条件：

| 项目 | 要求 |
|---|---|
| official eval | 调用统一函数 |
| export | 调用统一函数 |
| TCE | 调用统一函数 |
| SPS instability | 调用统一函数 |
| FP census | 调用统一函数 |
| legacy 工具 | 移入 legacy 或修复 |

### 7.3 SPS 双视图一次 forward

必须使用：

```python
torch.cat([img, sps_img], dim=0)
```

然后：

```python
weak, pert = output.chunk(2, dim=0)
```

继续条件：

| 项目 | 要求 |
|---|---|
| SPS active | 一次 batch-concat forward |
| 无 sequential BN forward | 是 |
| transform inverse alignment | 有测试 |

### 7.4 SPS disabled 等价 OHEM

必须新增或保留测试：

```text
alpha_sps = 0 equals OHEM
```

继续条件：

| 指标 | 要求 |
|---|---:|
| loss diff | < 1e-7 |
| grad max diff | < 1e-7 |
| one-step param diff | < 1e-7 |
| selected negative budget | 等于 OHEM |

### 7.5 Pd 语义统一

论文所有 Pd 必须是：

```text
target-level component Pd
```

不是 pixel recall。

继续条件：

| 项目 | 要求 |
|---|---|
| `BinaryMetricsGPU` | 不输出 `Pd`，只输出 `PixelRecall` |
| official eval | component-level Pd |
| README | 明确 Pd 定义 |

### 7.6 target-contrast 语义 bug 修复

必须先修：

```text
tools/sps_perturbation_census.py
```

当前 target-contrast selected score 不应使用：

```python
target_margin_signal()
```

而应使用：

```python
target_contrast_signal()
```

伪代码：

```python
if metric.startswith("target_margin_"):
    signal = target_margin_signal()
elif metric.startswith("target_contrast_"):
    signal = target_contrast_signal()
```

同时修 `loss.py` additive 分支：

```text
target_contrast 不允许调用 target_margin_signal
```

### 7.7 Step 1 通过条件

全部通过才进入 Step 2：

```text
python -m py_compile: PASS
test_probability_semantics: PASS
test_direct_export_parity: PASS
test_mshnet_head_consistency: PASS
test_sps_disabled_equals_ohem: PASS
test_sps_batch_concat_forward: PASS
test_pd_semantics: PASS
test_target_contrast_metric_is_continuous_relative_to_target: PASS
test_sps_census_target_contrast_uses_contrast_signal: PASS
```

建议检查命令：

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
python -m py_compile $(find . -name "*.py" -not -path "./legacy/*" -not -path "./tools/legacy/*")
pytest -q tests/test_sps_core_semantics.py
```

---

## 8. Step 2：OHEM 强基线复现

### 8.1 为什么要做

所有后续 SPS 结果必须和稳定 OHEM paired 比较。

如果 OHEM 本身复现不稳，SPS 结果没有意义。

### 8.2 要跑什么

三种子：

```text
seed42
seed43
seed44
```

Split：

```text
Full
Dev-HC / HC-Val
```

指标：

```text
mIoU
target-level Pd
FA ppm
Precision
F1
FP component count
far-FP count
```

### 8.3 继续条件

Full split OHEM 三种子均值应接近历史结果：

| 指标 | 目标 |
|---|---:|
| Full mIoU | 约 0.8320，允许小幅波动 |
| Full FA | 约 62 ppm |
| Full Precision | 约 0.905 |
| Full Pd | 约 0.978–0.980 |

若偏差较大：

```text
mIoU 差 > 0.01
FA 差 > 8 ppm
Pd 掉目标明显
```

先查代码、数据 split、probability、head、threshold，不进入 SPS。

---

## 9. Step 3：SPS 机制 Gate0，只诊断不训练

### 9.1 目的

验证核心假设：

> detached far-FP 的 self-perturbation instability 高于 target。

### 9.2 要做什么

使用冻结 OHEM checkpoint，对同一图像生成：

```text
weak view
perturbed view
```

计算：

```text
U = |P^w - P_tilde^p|
```

统计区域：

```text
target
boundary excess
detached near-FP
detached far-FP
easy background
```

### 9.3 Gate0 指标

| 指标 | 最低要求 | 强要求 |
|---|---:|---:|
| far-FP instability > target | 3/3 seeds | 3/3 seeds |
| far-FP vs target AUROC | ≥ 0.65 | ≥ 0.70 |
| SPS/TCE instability Spearman | ≥ 0.25 | ≥ 0.30 |
| target top-20% instability rate | ≤ 0.15 | ≤ 0.10 |
| selected/OHEM Jaccard | ≤ 0.70 | ≤ 0.60 |
| fallback images / batch | 0 | 0 |
| candidate ratio | 大于 OHEM budget | 稳定大于 OHEM budget |
| flat candidate ratio | 低 | 很低 |

### 9.4 当前问题

已有 Gate0 表明：

```text
far-FP instability 信号存在
但 target top-20% 过高
selected/OHEM overlap 过高
```

所以当前策略需要重新设计 candidate rule。

Gate0 不过，不训练。

---

## 10. Step 4：候选机制设计

### 10.1 不要继续 plain alpha sweep

已经确认：

```text
plain alpha sweep 不能解决机制问题
```

不要继续只调：

```text
alpha
lambda
tau
epoch
```

### 10.2 第一候选：target-contrast SPS

#### 为什么设计它

`target-margin` hard cutoff 会造成：

```text
candidate pool 太稀疏
fallback-dominated
```

`target-contrast` 用连续 sigmoid：

```text
U_contrast = sigmoid((U - U_target_ref - margin) / T)
```

目标是：

```text
降低目标不安全像素权重
避免 hard cutoff 导致 candidate 为空
```

#### Gate0 sweep 参数

候选参数：

```text
target_margin_quantile: 0.50 / 0.65 / 0.75 / 0.85
target_margin_temp: 0.005 / 0.01 / 0.02 / 0.05
candidate_topk_ratio: 0.02 / 0.05
candidate_topk_metric:
  target_contrast_instability
  target_contrast_sps_score
```

继续条件：

| 指标 | 要求 |
|---|---:|
| target top20 rate | ≤ 0.15 |
| selected/OHEM Jaccard | ≤ 0.65～0.70 |
| fallback images / batch | 0 |
| candidate ratio | > OHEM budget |
| far-FP removed recall | ≥ 0.50 |
| flat candidate ratio | 低 |

Gate0 不过，不训练。

### 10.3 第二候选：region-level SPS

如果 target-contrast 仍不能赢机制消融，转 region-level SPS。

原因：

```text
detached far-FP 通常不是单个像素，而是：
small blob
component
target-like region
```

像素级 SPS 容易和 OHEM 高度重合。

region-level SPS 使用区域分数：

```text
S(R) = mean(loss^-_R) * mean(U_R) * A(R) * M_far(R)
```

其中：

```text
mean(loss^-_R): 区域平均 hard-negative loss
mean(U_R): 区域平均 instability
A(R): 目标尺度面积先验
M_far(R): 远背景安全约束
```

Gate0 要求：

| 指标 | 要求 |
|---|---:|
| component candidate precision | ≥ 60% |
| target leakage | 0 |
| flat candidate ratio | ≤ 20% |
| selected/OHEM pixel overlap | ≤ 0.65 |
| far-FP component coverage | ≥ 50% |
| candidate count | 稳定非空 |

---

## 11. Step 5：单 seed 训练

### 11.1 只跑 seed42

先不要直接三种子。seed42 必须同时过：

```text
Full gate
HC-Val gate
mechanism control gate
```

### 11.2 Full gate

| 指标 | 要求 |
|---|---:|
| Full mIoU | ≥ OHEM - 0.001 |
| Full Pd | ≥ OHEM |
| Full target count | 不少于 OHEM |
| Full FA | 不高于 OHEM + 2 ppm |
| Full Precision | ≥ OHEM |

### 11.3 HC-Val gate

| 指标 | 要求 |
|---|---:|
| HC-Val mIoU | OHEM + ≥ 0.010 |
| HC-Val FA | OHEM - ≥ 20 ppm |
| HC-Val Precision | OHEM + ≥ 0.012 |
| HC-Val Pd | 不下降 |

### 11.4 停止条件

如果 seed42 出现：

```text
Full Pd 掉
HC-Val FA 上升
HC-Val 只 mIoU 涨但 Precision 不涨
candidate fallback 仍严重
selected/OHEM overlap 仍 > 0.8
```

停止当前版本，不跑 seed43/44。

---

## 12. Step 6：机制消融对照

SPS 必须赢过下列对照，否则创新性不成立。

| 对照 | 目的 |
|---|---|
| Two-view OHEM | 排除双视图训练收益 |
| Global consistency | 排除普通 consistency learning |
| Confidence-only far OHEM | 排除只靠高置信远背景 |
| Instability-only | 证明 hardness 必要 |
| SPS without far mask | 证明 far-background mask 必要 |
| OHEM | 强基线 |
| TCE | diagnostic oracle |

seed42 上必须满足：

```text
SPS > two-view OHEM
SPS > global consistency
SPS > confidence-only
SPS > no-far-mask
```

至少在以下指标上成立：

```text
HC-Val mIoU
HC-Val FA
HC-Val Precision
Pd 不下降
far-FP component count
Pd-matched FA
```

停止条件：

```text
two-view OHEM >= SPS
global consistency ≈ SPS
no-far-mask > SPS
```

如果出现以上情况，当前 SPS 机制不能作为 AAAI 主方法。

---

## 13. Step 7：三种子 paired gate

只有 seed42 和机制消融过后，才跑：

```text
seed42
seed43
seed44
```

### 13.1 Full split gate

| 指标 | 要求 |
|---|---:|
| 3/3 seeds mIoU | 不低于 paired OHEM |
| 3/3 seeds Pd | 不低于 paired OHEM |
| mean Precision | ≥ OHEM |
| mean FA | ≤ OHEM |
| target count | 不能少于 OHEM |

### 13.2 HC-Val gate

| 指标 | 要求 |
|---|---:|
| 3/3 seeds mIoU | 提升 |
| 3/3 seeds FA | 下降 |
| 3/3 seeds Precision | 提升或持平 |
| mean mIoU | +0.010 以上 |
| mean FA | -20 ppm 以上 |
| mean Precision | +0.012 以上 |
| Pd | 不下降 |

### 13.3 强 gate

| 指标 | 要求 |
|---|---:|
| HC-Val mIoU | +0.0125 以上 |
| HC-Val FA | -26 ppm 以上 |
| TCE hard-split 收益转化 | ≥ 70% |
| seed-wise | 3/3 全部通过 |

---

## 14. Step 8：threshold-matched 与 FP component 分析

### 14.1 为什么必须做

fixed threshold 0.5 好，不代表方法真的降低虚警。

必须排除：

```text
只是概率校准改变
只是阈值变保守
只是 Pd 损失换 FA
```

### 14.2 必须做两类 matched

```text
Pd-matched
mIoU-matched
```

### 14.3 继续条件

#### Full

| 指标 | 要求 |
|---|---:|
| Pd-matched FA | 3/3 seeds 不高于 OHEM |
| mIoU-matched Pd | 至少 2/3 不低于 OHEM |
| Precision | 不下降 |

#### HC-Val

| 指标 | 要求 |
|---|---:|
| Pd-matched FA | 3/3 下降 |
| mIoU-matched FA | 至少 2/3 下降 |
| far-FP count | 下降 |
| boundary excess | 不能是唯一改善来源 |

### 14.4 FP component 指标

必须报告：

```text
boundary excess pixels
detached near-FP components
detached far-FP components
far-FP confidence mass
far-FP pixel mass
removed FP
retained FP
new FP
```

继续条件：

| 指标 | 要求 |
|---|---:|
| detached far-FP count | 明显下降 |
| far-FP confidence mass | 明显下降 |
| new FP | 不增加 |
| Pd | 不下降 |
| boundary excess | 不能解释全部收益 |

---

## 15. Step 9：blind / external final test

### 15.1 为什么必须有

当前 HC-Val / HC-Test 已被用于多轮内部路线决策。最终投稿需要更干净证据：

```text
external dataset
cross-dataset evaluation
blind holdout
组内保存标签的 blind evaluation
```

### 15.2 只允许最终运行一次

```text
方法冻结
超参数冻结
checkpoint 选择规则冻结
threshold 规则冻结
然后只评估一次
```

### 15.3 最低投稿线

| 指标 | 要求 |
|---|---:|
| mIoU | OHEM + 0.008～0.010 |
| FA | OHEM - ≥ 15 ppm |
| Precision | OHEM + ≥ 0.010 |
| Pd | 不下降 |
| seeds | 至少 2/3 正收益 |

### 15.4 强结果线

| 指标 | 要求 |
|---|---:|
| mIoU | OHEM + ≥ 0.010 |
| FA | OHEM - ≥ 18 ppm |
| Precision | OHEM + ≥ 0.012 |
| seeds | 3/3 最好 |

---

## 16. Step 10：AAAI 投稿判断

### 16.1 可以投的条件

```text
[代码]
foreground probability 统一
direct/export parity 通过
MSHNet final head 统一
SPS disabled 等价 OHEM
无 train-time test leakage
Pd 语义统一
official scripts 可复现
requirements.txt 完整
README 主线清楚
legacy 不污染默认入口

[方法]
SPS 不是额外 loss，而是 same-budget OHEM reranking
SPS 赢 two-view OHEM
SPS 赢 global consistency
SPS 赢 confidence-only
SPS 赢 no-far-mask
far-FP component 下降

[性能]
Full 3/3 不退化
HC-Val 3/3 提升
threshold-matched 支持
blind/external 至少 2/3 提升
推理仍为单模型 1×
```

### 16.2 不能投的条件

满足任一条就不建议投稿：

```text
Full Pd 掉
Full 只有 mIoU 涨但 hard split 恶化
HC-Val 只有一个 seed 提升
SPS 和 two-view 差不多
SPS 和 global consistency 差不多
SPS 和 no-far-mask 差不多甚至更差
去掉 instability 不影响结果
去掉 far mask 不影响结果
FA 降低来自 Pd 下降
没有 blind/external evidence
代码仍有 probability/head/Pd 语义不一致
```

---

## 17. 当前代码下一步建议

### 17.1 立即修复

在当前目录执行：

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
```

然后按顺序修：

```text
1. 修 tools/sps_perturbation_census.py 中 target-contrast selected score：
   target_contrast_* 必须使用 target_contrast_signal()。

2. 修 loss.py additive target-contrast 分支：
   不要让 target_contrast 调用 target_margin_signal。

3. 将 tools/pcar_validate_exports.py 移到 legacy，或修成 unified probability + final head。

4. 增加 requirements.txt。

5. 增加 official e40 candidate 脚本，避免 README 结果与 official script 默认参数不一致。

6. 修改 train.py 默认模型，或强制 README 所有命令显式指定 model。

7. 根目录内部文档移到 docs/internal/。

8. 新增 target-contrast census 语义测试：
   test_sps_census_target_contrast_uses_contrast_signal。
```

### 17.2 立即停止

```text
plain alpha sweep
positive-margin filter branch
current target-margin branch
HC-Test 调参
OHCM / prototype / PCAR / TNC 回退
```

### 17.3 下一轮最合理实验

#### 方向 A：target-contrast SPS

前提：

```text
先修 target-contrast 诊断代码
只跑 Gate0
Gate0 不过不训练
```

#### 方向 B：region-level SPS

如果 target-contrast 仍不赢 two-view / no-far-mask，转 region-level SPS。

核心原因：

```text
detached far-FP 本质上是 component / region，不是孤立 pixel。
```

---

## 18. 最小 official 命令清单

以下是建议最终 README 保留的最小命令集合。具体参数按你当前工程脚本实际参数名修改，但入口应该收敛成这些类别。

### 18.1 代码检查

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
python -m py_compile $(find . -name "*.py" -not -path "./legacy/*" -not -path "./tools/legacy/*")
pytest -q tests/test_sps_core_semantics.py
```

### 18.2 OHEM baseline

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
bash tools/official/train_ohem_seed.sh 42
bash tools/official/train_ohem_seed.sh 43
bash tools/official/train_ohem_seed.sh 44
```

### 18.3 SPS candidate

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
bash tools/official/train_sps_seed.sh 42
```

seed42 过 gate 后才跑：

```bash
bash tools/official/train_sps_seed.sh 43
bash tools/official/train_sps_seed.sh 44
```

### 18.4 fixed threshold eval

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
bash tools/official/eval_fixed.sh
```

### 18.5 threshold-matched eval

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
bash tools/official/eval_threshold_matched.sh
```

### 18.6 FP component eval

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
bash tools/official/eval_component_fp.sh
```

---

## 19. 最终 Go / No-Go 表

| 阶段 | Go 条件 | No-Go 条件 |
|---|---|---|
| Step 1 代码语义 | probability/head/Pd/SPS disabled 全部通过 | 任一语义测试不通过 |
| Step 2 OHEM 复现 | OHEM 接近历史三种子结果 | OHEM 自身不稳 |
| Step 3 Gate0 | far-FP instability 明显高于 target，Jaccard 不高 | target top20 高，fallback 严重 |
| Step 5 seed42 | Full 不退，HC-Val 降 FA，Precision 涨 | Pd 掉，FA 升，只 mIoU 涨 |
| Step 6 机制消融 | SPS 赢 two-view / consistency / no-far-mask | 对照方法 ≥ SPS |
| Step 7 三种子 | Full 3/3 不退，HC-Val 3/3 提升 | 只有单 seed 好 |
| Step 8 matched | Pd-matched FA 仍下降 | 降 FA 来自阈值或 Pd 损失 |
| Step 9 blind/external | 至少 2/3 正收益 | 外部或 blind 不成立 |
| Step 10 AAAI | 方法、性能、代码三者同时过线 | 任一主线证据缺失 |

---

## 20. 最终一句话

**这个 idea 可以继续，但当前不要直接按 AAAI 投。**

最合理的路线是：在 `/home/ly/AAAI/OHCM-MSHNet-main` 原地完成代码语义修复和实验 gate，先证明 SPS-OHEM 不是普通 two-view / consistency / no-far-mask 的副产物，再用三种子、threshold-matched 和 blind/external 结果决定是否投稿 AAAI。
