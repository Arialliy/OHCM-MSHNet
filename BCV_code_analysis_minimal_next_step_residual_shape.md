# OHCM-MSHNet 代码严格分析：已通过部分冻结，失败部分最小修改

> 当前仓库：`https://github.com/Arialliy/OHCM-MSHNet`  
> 当前强基线：MSHNetOHEM  
> 当前阶段：BCV 已完成 Gate-A / Gate-B split / Gate-C  
> 当前结论：**不要整体推翻，不要继续大改结构；已通过部分冻结，只修改 Gate-C 失败的 validity 判据。**  
> 下一步唯一允许方向：**Gate-D：Residual Shape / Morphology Audit**  
> 当前禁止：训练、HC-Val、seed43/44、HC-Test、blind、external、继续调 beta/lambda。

---

## 1. 先明确：当前哪些代码已经通过，不应再改？

当前项目方向是红外小目标检测，核心问题是复杂背景中高响应伪目标导致 false alarm，代码基础是 BasicIRSTD + MSHNet，后续代码和结果均在 OHCM-MSHNet 下。  
因此所有代码推进都必须围绕这个目标，而不是反复推翻已验证模块。

### 1.1 冻结 MSHNetOHEM evidence anchor

不改：

```text
MSHNetOHEM 主干
MSHNet final fused head official 路径
foreground_probability()
direct/export parity
target-level Pd evaluation
```

理由：

```text
MSHNetOHEM 是当前唯一稳定 anchor。
所有新方法必须保护它，而不是污染它。
```

必须保持：

```text
new_method_disabled == MSHNetOHEM
```

---

### 1.2 冻结 BCV Gate-A 代码

Gate-BCV-A 已通过：

```text
max_prob_diff = 0.0
mask_diff_pixels = 0
mIoU/Pd/FA_ppm diff = 0.0
```

这说明：

```text
beta=0 时 BCV final output 与 OHEM 完全一致。
```

不应再改：

```text
BCV beta=0 path
BCV evidence_logit path
BCV final_logit = evidence_logit when beta=0
check_bcv_gate_a.py
direct/export parity 参数
```

---

### 1.3 冻结 BCV background residual branch 的基本实现

Gate-B split 已通过：

```text
background_residual_gate_pass = true
target_residual_bg_ratio_mean = 5.7653
residual_auroc_target_vs_far_mean = 0.9355
background_reconstruction_error_mean = 0.2767
target_leakage_pixels_total = 0
```

说明：

```text
background residual 能很好地区分 target 与普通 far background。
```

不应再改：

```text
background branch 主结构
background reconstruction loss
residual = |I - B|
residual_norm
target-near exclusion mask
check_bcv_gate_b.py 的 split decision
```

这部分是目前所有尝试里最强的有效信号之一。

---

### 1.4 冻结 Gate-C 工具本身

Gate-C 已经回答了一个非常关键的问题：

```text
residual magnitude 能不能安全区分 OHEM far-FP 与真实 target？
```

答案：

```text
不能。
```

Gate-C 输出：

| 指标 | 结果 |
|---|---:|
| target_component_count | 968 |
| far_fp_component_count | 29 |
| target_vs_far_fp_residual_auroc | 0.7618 |
| suppressible_far_fp_rate_at_target_recall_99 | 0.0345 |
| suppressible_far_fp_rate_at_target_recall_995 | 0.0 |
| target_residual_q10 | 2.439 |
| far_fp_residual_median | 3.281 |
| decision | STOP_BCV under residual magnitude rule |

这说明：

```text
residual magnitude 有一定排序能力，
但为了保护 99% target，只能压掉 3.45% far-FP。
```

所以 Gate-C 工具本身是有价值的诊断工具，不应删除。

---

## 2. 当前真正失败的是什么？

失败的不是整个 BCV，而是这个假设：

```text
低 residual / 背景一致 = false positive
```

Gate-C 证明：

```text
OHEM far-FP residual 并不一定低。
far-FP residual median 甚至高于 target residual q10。
```

这意味着 OHEM far-FP 很可能也是：

```text
高 residual 的背景异常
局部边缘
热斑
纹理峰值
结构化 clutter
```

因此不能再用：

```text
residual magnitude
```

作为唯一 validity 判据。

---

## 3. 代码层面还有哪些需要注意？

下面是对当前公开仓库 / 上传代码中已观察到的代码问题的严格分析。

---

### 3.1 `loss.py` 中 SPS rerank 仍是 legacy 风险，不应作为新路线基础

在 `SelfPerturbationStabilityLoss.rerank_ohem_loss()` 中，有 fallback：

```python
if not disabled_rerank and int(candidate.sum().item()) < 1:
    disabled_rerank = True
    candidate = neg.clone()
    fallback_used = 1
```

这对 SPS 路线来说是风险：

```text
candidate 为空时会退回全部负样本；
这会污染“far-background candidate”语义。
```

结论：

```text
SPS rerank 保留为 legacy diagnostic；
不要在新 BCV / shape 路线中复用这套 candidate fallback。
```

---

### 3.2 `sps_score_mean` 当前记录的是 Jaccard，不是真正 score mean

`loss.py` 的 stats 中：

```python
'sps_score_mean': torch.tensor(float(sum(jaccard_values) / max(1, len(jaccard_values))), ...)
```

这其实是：

```text
selected/OHEM Jaccard
```

不是 score mean。

建议：

```text
若继续保留 SPS legacy，应改名为 sps_ohem_jaccard_mean；
不要把它当 score mean 分析。
```

但这不是当前 BCV 的 blocker。

---

### 3.3 `train.py` 仍有 batch size 1 静默跳过

当前训练循环中有：

```python
if img.shape[0] == 1:
    continue
```

这会导致：

```text
不同数据量 / batch size 下每个 epoch 训练样本不一致；
复现时不透明。
```

建议改成：

```python
DataLoader(..., drop_last=True)
```

或者让 loss 支持 batch size 1。

这是工程 P1 问题。  
不阻断 Gate-D，但投稿前要修。

---

### 3.4 legacy 工具仍可能绕过 official probability/head 语义

公开仓库中仍有历史工具，例如：

```text
tools/pcar_validate_exports.py
```

这类工具可能仍存在：

```text
MSHNet output0 auto head
torch.sigmoid(logit) 绕过 foreground_probability()
```

建议：

```text
移入 tools/legacy/
或统一调用 foreground_probability()
```

投稿 official path 不应使用这些工具。

---

### 3.5 `Net` 和 `loss.py` 过于臃肿

`net.py` 和 `loss.py` 仍包含大量历史路线：

```text
OHCM
PFR
ERD
TSR
SPS
TopK
PCAR 相关残留
```

这对研究阶段可以接受，但投稿代码需要：

```text
official path 最小化
legacy path 隔离
```

建议新增：

```text
net_official.py
loss_official.py
```

或者至少在 README 中明确：

```text
AAAI official run only uses MSHNetOHEM + BCV/Shape path.
```

---

## 4. 下一步不能做什么？

现在不要做：

```text
不要训练 BCV
不要跑 HC-Val
不要跑 seed43/44
不要跑 HC-Test
不要跑 blind/external
不要调 beta
不要调 lambda
不要把 BCV 整体推翻
不要再新建一个大模型
不要回到 SPS / ECDV / MSCV
```

当前唯一允许：

```text
Gate-D residual shape / morphology audit
```

---

# 5. 下一步要改哪里？

只改失败的部分：

```text
validity definition
```

不改通过的部分：

```text
evidence branch
background branch
residual computation
parity path
Gate-A/B/C tools
```

---

# 6. 新增 Gate-D：Residual Shape / Morphology Audit

## 6.1 为什么要做

Gate-C 说明 residual magnitude 不够。  
但 OHEM far-FP 和真实 target 的差异可能不在 residual 大小，而在 residual 形态。

真实 target residual 应该更像：

```text
紧凑
中心峰值
近似圆形 / blob
中心强、周围弱
各向同性
小面积
```

false positive residual 可能更像：

```text
边缘
条带
纹理
扩散斑
不规则结构
局部高亮背景的一部分
```

所以 Gate-D 要验证：

> residual shape 能否区分 OHEM far-FP 与 target？

---

## 6.2 新增文件

```text
utils/residual_shape_features.py
tools/official/check_bcv_gate_d_residual_shape.py
tests/test_residual_shape_features.py
tests/test_bcv_gate_d_residual_shape.py
```

---

## 6.3 `utils/residual_shape_features.py` 设计

需要计算以下 shape features。

### Feature 1：compactness

\[
compactness=\frac{4\pi A}{P^2}
\]

其中：

```text
A = component area
P = component perimeter
```

目标：

```text
真实小目标 compactness 高
边缘/条带类 false alarm compactness 低
```

---

### Feature 2：bbox fill ratio

\[
fill=\frac{A}{w \cdot h}
\]

目标：

```text
真实小目标填充相对集中
细长边缘 / 纹理填充率异常
```

---

### Feature 3：anisotropy

根据 component 坐标协方差矩阵特征值：

\[
anisotropy=\frac{\lambda_{max}}{\lambda_{min}+\epsilon}
\]

目标：

```text
真实小目标更接近各向同性
边缘/条带 false alarm 各向异性高
```

---

### Feature 4：center-surround residual contrast

\[
CSR=\overline{R}_{center}-\overline{R}_{ring}
\]

目标：

```text
真实小目标中心强、周围弱
背景边缘可能环形差异不明显
```

---

### Feature 5：radial symmetry

将 component 周围分为四象限：

\[
symmetry=1-\frac{std(q_1,q_2,q_3,q_4)}{mean(q_1,q_2,q_3,q_4)+\epsilon}
\]

目标：

```text
真实小目标 residual 分布更对称
```

---

### Feature 6：DoG / LoG peakness

用 Difference-of-Gaussian：

\[
DoG = G_{\sigma_1}(R)-G_{\sigma_2}(R)
\]

目标：

```text
真实小目标是 blob-like peak
```

---

## 6.4 shape score

先用非学习公式：

\[
S_{shape}
=
w_1 compactness
+
w_2 fill
-
w_3 anisotropy
+
w_4 CSR
+
w_5 symmetry
+
w_6 peakness
\]

第一版：

```text
不训练
不加网络
只做诊断
```

---

# 7. Gate-D 工具设计

## 7.1 文件

```text
tools/official/check_bcv_gate_d_residual_shape.py
```

---

## 7.2 输入

```bash
python tools/official/check_bcv_gate_d_residual_shape.py \
  --dataset_name NUDT-SIRST \
  --split train \
  --ohem_checkpoint /path/to/MSHNetOHEM_400.pth.tar \
  --bcv_checkpoint_or_init /path/to/BCV_beta0.pth.tar \
  --threshold 0.5 \
  --output_dir docs/internal/bcv_gate_d/seed42_nudt_train
```

---

## 7.3 统计对象

沿用 Gate-C：

```text
GT target components
OHEM matched target components
OHEM detached near-FP components
OHEM detached far-FP components
```

---

## 7.4 输出 `summary.json`

```json
{
  "target_component_count": 0,
  "far_fp_component_count": 0,

  "shape_auc_target_vs_far_fp": 0.0,
  "shape_auc_target_vs_near_fp": 0.0,

  "target_shape_q10": 0.0,
  "far_fp_shape_median": 0.0,
  "near_fp_shape_median": 0.0,

  "suppressible_far_fp_rate_at_target_recall_99": 0.0,
  "suppressible_far_fp_rate_at_target_recall_995": 0.0,

  "compactness_auc": 0.0,
  "anisotropy_auc": 0.0,
  "center_surround_auc": 0.0,
  "dog_peakness_auc": 0.0,

  "gate_pass": false,
  "overall_decision": "STOP_OR_PROCEED"
}
```

---

# 8. Gate-D 继续条件

Gate-D 通过才允许 deterministic shape calibration。

| 指标 | 要求 |
|---|---:|
| far_fp_component_count | ≥ 20 或仅诊断 |
| shape_auc_target_vs_far_fp | ≥ 0.70 |
| suppressible_far_fp_rate_at_target_recall_99 | ≥ 0.20 |
| suppressible_far_fp_rate_at_target_recall_995 | ≥ 0.10 |
| target_shape_q10 > far_fp_shape_median | 最好成立 |
| 单个 feature 至少一个 AUC | ≥ 0.70 |

---

# 9. Gate-D 停止条件

如果出现任一情况，停止 BCV shape 路线：

```text
shape_auc_target_vs_far_fp < 0.65
target 与 far-FP shape 分布高度重叠
target recall 99% 下 suppressible far-FP < 10%
far-FP 数量太少且无法可靠分析
```

如果 Gate-D 失败，说明：

```text
residual magnitude 和 residual shape 都不能安全区分 OHEM far-FP 与 target。
```

那 BCV 应正式停止。

---

# 10. Gate-D 通过后才做 Gate-E

Gate-D 通过后，仍然不要训练 verifier。  
先做 deterministic shape calibration。

---

## 10.1 validity formula

\[
p_{valid}
=
\sigma\left(
\frac{S_{shape}-\theta}{T}
\right)
\]

其中：

```text
theta = Gate-D 中保护 99% target 的 shape threshold
T = temperature
```

---

## 10.2 calibration formula

\[
z_{final}
=
z_e
-
\beta
\cdot
(1-p_{valid})
\cdot
ReLU(z_e-\tau)
\]

---

## 10.3 修改 `BCV_MSHNet.py`

只增加：

```python
validity_mode: str = "learned"
# learned / residual_formula / shape_formula

shape_theta: float = 0.0
shape_temp: float = 0.2
```

forward 中：

```python
if self.validity_mode == "shape_formula":
    p_valid = torch.sigmoid(
        (shape_score - self.shape_theta) / self.shape_temp
    )
elif self.validity_mode == "residual_formula":
    p_valid = torch.sigmoid(
        (residual_norm - self.residual_theta) / self.residual_temp
    )
else:
    p_valid = torch.sigmoid(self.verifier(verifier_input))
```

注意：

```text
shape_formula 第一版可以只在 export/eval 实现，
不必立刻做全可微训练版本。
```

因为 Gate-E 是 deterministic sanity check。

---

## 10.4 Gate-E Full gate

只跑 Full，不看 HC-Val。

允许 beta：

```text
0.02
0.05
0.10
```

继续条件：

| 指标 | 要求 |
|---|---:|
| Full mIoU | ≥ OHEM - 0.001 |
| Full Pd | ≥ OHEM |
| Full target count | 不少于 OHEM |
| Full Precision | ≥ OHEM |
| Full FA | ≤ OHEM + 2 ppm |

全部失败则停止。

---

## 10.5 Gate-F HC-Val gate

只有 Full 过后才跑 HC-Val。

初始 sanity 要求：

| 指标 | 要求 |
|---|---:|
| HC-Val mIoU | OHEM + ≥ 0.005 |
| HC-Val FA | OHEM - ≥ 10 ppm |
| HC-Val Precision | 不低于 OHEM |
| HC-Val Pd | 不下降 |

这只是 deterministic sanity，不是最终投稿线。

---

# 11. Gate-F 通过后，才考虑 learned verifier

如果 deterministic shape calibration 有效，再训练 learned verifier。  
否则不要训练。

---

## 11.1 Learned verifier 输入

```text
z_e
p_e
residual_norm
shape_score_map
local_contrast
background_gradient
```

---

## 11.2 Learned verifier 监督

target：

```text
p_valid = 1
```

low-shape far high-evidence candidate：

```text
p_valid = 0
```

---

## 11.3 Stage 1

```text
beta = 0
freeze evidence
train verifier only
```

---

## 11.4 Stage 2

```text
beta ramp
Full gate every N epochs
```

---

# 12. 代码修改边界

## 12.1 不修改

```text
MSHNetOHEM evidence branch
foreground_probability.py
BCV beta=0 equivalence path
background branch and residual computation
Gate-A / Gate-B / Gate-C tools
target-level Pd evaluation
direct/export parity
```

---

## 12.2 只新增

```text
utils/residual_shape_features.py
tools/official/check_bcv_gate_d_residual_shape.py
tools/official/apply_bcv_shape_formula.py
tests/test_residual_shape_features.py
tests/test_bcv_gate_d_residual_shape.py
```

---

## 12.3 少量修改

`BCV_MSHNet.py`：

```text
新增 validity_mode = shape_formula
新增 shape_theta
新增 shape_temp
```

`net.py`：

```text
新增 bcv_validity_mode
新增 bcv_shape_theta
新增 bcv_shape_temp
```

---

# 13. 为什么这是“仔细分析代码”后的最小推进？

因为当前代码中已经有几部分是稳定的：

```text
OHEM anchor 稳定
probability 语义稳定
BCV beta=0 等价稳定
background residual 有效
Gate-C 工具有效
```

真正不通过的是：

```text
validity 判据只用了 residual magnitude
```

所以最小修改就是：

```text
保留 residual
改变 validity 判据
从 magnitude 改为 shape / morphology
```

而不是：

```text
推翻 BCV
再新建 ECDV/MSCV/CDV
再重写大结构
```

---

# 14. 如果 Gate-D 失败

如果 residual shape 也失败：

```text
shape_auc_target_vs_far_fp < 0.65
99% target recall 下 suppressible far-FP < 10%
```

那说明：

```text
BCV residual family 无法区分 OHEM FP 与 target。
```

此时应停止 BCV，不再训练。

下一步不应继续单帧 false-alarm suppression，而应转向：

```text
cross-dataset robustness
domain generalization
multi-frame temporal consistency
weak supervision
failure analysis benchmark
```

---

# 15. 最终判断

当前结论：

```text
BCV-0 PASS
BCV-1 PASS
BCV-2 FAIL
```

下一步：

```text
BCV-3 / Gate-D residual shape audit
```

一句话：

> **不要修改已经通过的 evidence / probability / residual / parity 代码。只修改失败的 validity 判据：从 residual magnitude 改成 residual shape。Gate-D 通过后，再做 deterministic calibration；Gate-D 不通过，就停止 BCV。**
