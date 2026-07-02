# OHCM-MSHNet：Gate-D 失败后的代码分析、冻结清单与最小推进方案

> 当前仓库：`https://github.com/Arialliy/OHCM-MSHNet`  
> 当前强锚点：MSHNetOHEM  
> 当前阶段：BCV Gate-D 已失败  
> 当前结论：**不要整体推翻；冻结已通过代码，只针对失败的 validity 判据继续做最小诊断。**  
> 下一步唯一允许：**Gate-D2：mass-weighted residual/shape audit**  
> 当前禁止：训练、HC-Val、seed43/44、HC-Test、blind、external、大改模型结构。

---

## 0. 当前事实

你已经完成：

```text
新增 utils/residual_shape_features.py
新增 tools/official/check_bcv_gate_d_residual_shape.py
新增 tools/official/apply_bcv_shape_formula.py
新增 tests/test_residual_shape_features.py
新增 tests/test_bcv_gate_d_residual_shape.py
少量修改 BCV_MSHNet.py / net.py，仅增加 shape_formula、bcv_shape_theta、bcv_shape_temp 支持
```

验证通过：

```text
py_compile: PASS
pytest: PASS
git diff --check: PASS
```

Gate-D 结果：

| 指标 | 结果 |
|---|---:|
| shape_auc_target_vs_far_fp | 0.7785 |
| shape_auc_target_vs_near_fp | 0.7944 |
| far_fp_component_count | 29 |
| suppressible_far_fp_rate_at_target_recall_99 | 0.0345 |
| suppressible_far_fp_rate_at_target_recall_995 | 0.0 |
| target_shape_q10 | 1.7597 |
| far_fp_shape_median | 2.1676 |
| overall_decision | STOP_BCV |

文档停止原因：

```text
99% target recall 下只能压制 3.45% far-FP，低于 <10% 停止线；
target_shape_q10 <= far_fp_shape_median；
target 与 far-FP shape 分布重叠；
未进入 Gate-E、未训练、未跑 HC-Val / seed43/44 / HC-Test / blind / external。
```

---

# 1. 先明确：哪些代码已经通过，不能再随便改

当前不能再无限推翻。已经通过的代码必须冻结。

---

## 1.1 冻结 MSHNetOHEM evidence anchor

不改：

```text
MSHNetOHEM 主检测分支
MSHNet final fused head official path
foreground_probability()
direct/export parity
target-level Pd evaluation
OHEM checkpoint / export / threshold curve 逻辑
```

理由：

```text
MSHNetOHEM 是当前唯一稳定 anchor。
所有新模型必须以它为 evidence anchor。
```

---

## 1.2 冻结 BCV Gate-A 路径

Gate-A 已通过：

```text
beta=0 final == OHEM
max_prob_diff = 0.0
mask_diff_pixels = 0
mIoU/Pd/FA_ppm diff = 0.0
```

不改：

```text
BCV beta=0 逻辑
z_final = z_e when beta=0
Gate-A 工具
direct/export parity 参数
```

这部分已经证明新结构关闭时不会污染 OHEM。

---

## 1.3 冻结 BCV background residual branch 基础实现

Gate-B split 已证明：

```text
background_residual_gate_pass = true
target_residual_bg_ratio_mean = 5.7653
residual_auroc_target_vs_far_mean = 0.9355
background_reconstruction_error_mean = 0.2767
target_leakage_pixels_total = 0
```

不改：

```text
background branch 主体结构
background reconstruction loss
residual = |I - B|
residual_norm
target-near exclusion mask
Gate-B split decision
```

解释：

> 背景残差能很好地区分 target 和普通 far background。这个信号是有效的。

---

## 1.4 冻结 Gate-C / Gate-D 工具作为诊断工具

Gate-C 和 Gate-D 不是失败代码，而是成功发现了失败假设。

Gate-C 发现：

```text
residual magnitude 无法在保护 99% target 的前提下压制足够 far-FP。
```

Gate-D 发现：

```text
residual shape 有全局 AUC，但在 high target recall 下仍不能安全压制足够 far-FP。
```

所以不删除这些工具。它们应保留在：

```text
tools/official/
docs/internal/
```

用于 negative evidence。

---

# 2. Gate-D 失败到底说明什么？

## 2.1 AUC 通过，不代表可校准

Gate-D 有：

```text
shape_auc_target_vs_far_fp = 0.7785
```

这说明 shape score 对 target 和 far-FP 有一定全局排序能力。

但关键指标是：

```text
suppressible_far_fp_rate_at_target_recall_99 = 0.0345
```

这说明：

> 当你要求保护 99% 的真实 target 时，只能安全压掉约 1 / 29 个 far-FP component。

因此：

```text
AUC 有用，但 tail separation 不够。
```

对 IRSTD 来说，tail separation 比平均 AUC 更重要，因为漏一个目标就会导致 Pd 下降。

---

## 2.2 target_shape_q10 <= far_fp_shape_median 是核心失败信号

结果：

```text
target_shape_q10 = 1.7597
far_fp_shape_median = 2.1676
```

含义：

> 至少有一部分真实 target 的 shape score 低于大量 far-FP。

如果你用 shape threshold 去压 far-FP，就会先伤到 weak / irregular target。

所以：

```text
shape score 不能作为 component-level hard suppression 判据。
```

---

## 2.3 far_fp_component_count = 29，样本很少

只有 29 个 far-FP component。

这说明两件事：

1. component-level suppression rate 很不稳定；
2. 用 component count 作为唯一 gate 可能过于保守。

例如：

```text
如果被压掉的 1 个 component 很大、概率很高，
它可能贡献大量 FA pixel mass。
```

所以 Gate-D 失败后，不应立刻改模型，而应先问：

> component-level 只压 3.45%，但它压掉的 pixel mass / confidence mass 是否有意义？

这是下一步最小诊断。

---

# 3. 当前真正失败的部分

当前失败的不是：

```text
BCV 模型
background residual
shape feature
Gate 工具
```

失败的是：

```text
把 residual magnitude / shape score 当作 component-level suppressor。
```

也就是：

```text
component-level binary decision 不可用。
```

但仍可能存在：

```text
pixel-level partial suppression
confidence-mass reduction
large-FP mass reduction
```

这还没验证。

---

# 4. 下一步不应做什么

现在不要做：

```text
不要训练 BCV
不要跑 Gate-E
不要跑 HC-Val
不要跑 seed43/44
不要跑 HC-Test
不要跑 blind/external
不要调 beta
不要调 lambda
不要重新设计大模型
不要回到 ECDV / MSCV / SPS
```

---

# 5. 下一步唯一允许：Gate-D2 mass-weighted residual/shape audit

## 5.1 为什么要做 Gate-D2

Gate-D 是 component-level：

```text
压掉多少 far-FP components？
```

但 FA 和 Precision 不只取决于 component 数量，也取决于：

```text
FP pixel mass
FP confidence mass
FP area
FP peak probability
```

因此需要检查：

> residual / shape 虽然压不掉很多 FP components，但能否压掉足够的 FP pixels 或 confidence mass？

---

## 5.2 Gate-D2 的目标

回答三个问题：

1. 在保护 99% target components 时，能压掉多少 far-FP pixel mass？
2. 在保护 99.5% target pixels 时，能压掉多少 FP confidence mass？
3. 被压掉的 far-FP 是否是高置信、大面积、真正影响 FA 的 component？

---

# 6. 新增工具：`check_bcv_gate_d2_mass_shape.py`

## 6.1 文件路径

新增：

```text
tools/official/check_bcv_gate_d2_mass_shape.py
```

不修改 Gate-D 原工具。

---

## 6.2 输入

```bash
python tools/official/check_bcv_gate_d2_mass_shape.py \
  --dataset_name NUDT-SIRST \
  --split train \
  --ohem_checkpoint /path/to/MSHNetOHEM_400.pth.tar \
  --bcv_checkpoint_or_init /path/to/BCV_beta0.pth.tar \
  --threshold 0.5 \
  --gate_d_summary docs/internal/bcv_gate_d/seed42_nudt_train/summary.json \
  --output_dir docs/internal/bcv_gate_d2/seed42_nudt_train
```

---

## 6.3 统计对象

沿用 Gate-D 的 components：

```text
GT target components
OHEM matched target prediction components
OHEM detached near-FP components
OHEM detached far-FP components
```

新增 pixel / mass 统计：

```text
target pixels
target confidence mass
far-FP pixels
far-FP confidence mass
far-FP peak prob
far-FP area
```

---

## 6.4 输出 `summary.json`

```json
{
  "target_component_count": 968,
  "far_fp_component_count": 29,

  "target_component_recall_threshold": 0.0,
  "target_pixel_recall_threshold": 0.0,

  "suppressible_far_fp_component_rate_at_target_component_recall_99": 0.0,
  "suppressible_far_fp_pixel_mass_rate_at_target_component_recall_99": 0.0,
  "suppressible_far_fp_confidence_mass_rate_at_target_component_recall_99": 0.0,

  "suppressible_far_fp_pixel_mass_rate_at_target_pixel_recall_995": 0.0,
  "suppressible_far_fp_confidence_mass_rate_at_target_pixel_recall_995": 0.0,

  "top_suppressible_fp_area_mean": 0.0,
  "top_suppressible_fp_peak_prob_mean": 0.0,

  "mass_gate_pass": false,
  "overall_decision": "STOP_OR_PROCEED"
}
```

---

# 7. Gate-D2 继续条件

Gate-D2 通过才允许做 deterministic shape / residual formula calibration。

### 7.1 Component-level 不再是唯一标准

因为 far_fp_component_count 只有 29。

继续条件改为 mass-weighted：

| 指标 | 要求 |
|---|---:|
| far_fp_component_count | ≥ 20，当前满足 |
| suppressible_far_fp_component_rate_at_target_component_recall_99 | 可低，但记录 |
| suppressible_far_fp_pixel_mass_rate_at_target_component_recall_99 | ≥ 0.15 |
| suppressible_far_fp_confidence_mass_rate_at_target_component_recall_99 | ≥ 0.15 |
| suppressible_far_fp_pixel_mass_rate_at_target_pixel_recall_995 | ≥ 0.10 |
| suppressible_far_fp_confidence_mass_rate_at_target_pixel_recall_995 | ≥ 0.10 |
| target leakage | 0 |

---

## 7.2 停止条件

如果出现：

```text
component rate 低
pixel mass rate < 0.10
confidence mass rate < 0.10
suppressed FP 都是低置信小噪声
```

则说明：

```text
residual / shape family 不能有效降低 FA。
```

BCV 应正式停止，不再训练。

---

# 8. 如果 Gate-D2 通过：Gate-E deterministic formula calibration

Gate-D2 通过后，仍不训练。  
先做 deterministic calibration。

---

## 8.1 validity formula

组合 residual + shape：

\[
S_{\mathrm{valid}}
=
w_r S_{\mathrm{residual}}
+
w_s S_{\mathrm{shape}}
\]

\[
p_{\mathrm{valid}}
=
\sigma\left(
\frac{S_{\mathrm{valid}}-\theta}{T}
\right)
\]

其中：

```text
theta 来自 Gate-D2 的 target-protection threshold
T 小范围固定
```

---

## 8.2 calibration

\[
z_{final}
=
z_e
-
\beta
\cdot
(1-p_{\mathrm{valid}})
\cdot
ReLU(z_e-\tau)
\]

---

## 8.3 只跑 Full

允许 beta：

```text
0.02
0.05
0.10
```

Full 继续条件：

| 指标 | 要求 |
|---|---:|
| Full mIoU | ≥ OHEM - 0.001 |
| Full Pd | ≥ OHEM |
| Full target count | 不少于 OHEM |
| Full Precision | ≥ OHEM |
| Full FA | ≤ OHEM + 2 ppm |

全部失败则停止。

---

## 8.4 Full 过后才跑 HC-Val

HC-Val 初始 sanity 线：

| 指标 | 要求 |
|---|---:|
| HC-Val mIoU | OHEM + ≥ 0.005 |
| HC-Val FA | OHEM - ≥ 10 ppm |
| HC-Val Precision | 不低于 OHEM |
| HC-Val Pd | 不下降 |

这是 sanity，不是投稿线。

---

# 9. 代码修改边界

## 9.1 不修改

```text
model/BCV_MSHNet.py 的 background branch
BCV beta=0 path
foreground_probability.py
Gate-A
Gate-B split
Gate-C
Gate-D
residual_shape_features.py
target-level Pd evaluation
direct/export parity
```

---

## 9.2 只新增

```text
tools/official/check_bcv_gate_d2_mass_shape.py
tests/test_bcv_gate_d2_mass_shape.py
```

---

## 9.3 可选小修改

仅当 Gate-D2 通过后，才修改：

```text
BCV_MSHNet.py: validity_mode="shape_residual_formula"
net.py: bcv_validity_mode / bcv_shape_residual_theta / temp
tools/official/apply_bcv_shape_formula.py: 支持 mass-threshold
```

---

# 10. 如果 Gate-D2 失败，怎么办？

如果 Gate-D2 失败，就正式停止 BCV residual/shape family。

此时结论是：

```text
background residual 能区分 target 与普通 background，
但不能安全区分 OHEM false positives 与 targets。
```

这意味着：

> 单帧 false-alarm suppression 在当前 MSHNetOHEM + NUDT-SIRST 训练协议下缺少可学习信息。

下一步不应再继续：

```text
SPS
ECDV
MSCV
BCV
PFR/ERD/CGA
hard-clutter bank
component mining
```

应该转向：

```text
cross-dataset generalization
multi-frame temporal consistency
weak supervision
failure analysis benchmark
```

---

# 11. 为什么这才是“仔细分析代码”后的正确推进？

因为你现在已经有很多通过的代码：

```text
parity
probability
background branch
residual computation
shape feature extraction
Gate tools
```

不能每次失败就整体推翻。

当前失败点非常明确：

```text
component-level suppressibility 不足
```

所以最小下一步是：

```text
检查 mass-level suppressibility
```

如果 mass-level 也不足，才停止 BCV。  
如果 mass-level 足够，再做 deterministic formula calibration。

---

# 12. 当前项目状态建议写入 README

建议新增：

```markdown
## Current BCV Status

BCV-A beta=0 parity: PASS.
BCV-B background residual separation: PASS.
BCV-C residual magnitude OHEM-FP audit: FAIL for component suppressibility.
BCV-D residual shape audit: FAIL for component suppressibility, despite AUC > 0.77.
Next allowed work:
  BCV-D2 mass-weighted residual/shape audit only.

Do not train BCV.
Do not run HC-Val, HC-Test, blind, or external.
Do not modify OHEM evidence branch or BCV background branch.
```

---

# 13. 最终结论

当前不是 BCV 全失败，而是：

```text
BCV residual magnitude: cannot safely suppress enough FP components.
BCV residual shape: has global AUC but cannot safely suppress enough FP components.
```

下一步只允许：

```text
BCV-D2 mass-weighted audit
```

一句话：

> **已通过的 evidence、probability、background residual、shape feature 代码都不要再改；只针对失败的 component-level suppressibility 增加 mass-weighted audit。Gate-D2 通过才做 deterministic calibration；Gate-D2 不通过，正式停止 BCV，并转向跨域/时序/弱监督等新方向。**
