# 为什么越修改越差：OHCM-MSHNet / TCE-TWA-TCSR 路线根因分析

## 0. 结论先行

当前现象不应理解为“代码越改越差”，而应理解为：

```text
随着 gate 从 seed42 internal validation 推进到 mechanism control、train-only signal audit、multi-seed internal、threshold-matched，再推进到 blind/external，之前内部 split 上可见的小正收益被逐层剥离；最后暴露出该路线的核心收益不是稳定泛化机制，而是内部数据分布上的保守化 false-alarm reduction。
```

更直接地说：

```text
MSHNetOHEM baseline 已经很强。
后续多数修改都试图降低 false alarm。
但 IR small target detection 里，降低 false alarm 很容易同时压低弱小真实目标。
内部 HC split 上这看起来像 Precision / FA 改善；外部 split 上就变成 Pd 下降。
```

因此，真正恶化的不是 py_compile / pytest / schema 层面的代码质量，而是候选方法的科学假设在更严格的 gate 上逐步失败。

---

## 1. 当前事实链

### 1.1 TWA-4 no-BN 曾经看起来很强

Gate-D seed42 HC-Val：

```text
mIoU delta      +0.02910
FA ppm delta    -58.4920
Precision delta +0.03545
Pd delta        +0.0
```

这个结果说明：在 seed42 HC-Val 上，TWA-4 no-BN 确实降低了 hard-clutter false alarms，而且没有损 Pd。

但是 Gate-E 发现：

```text
TWA-4 HC-Val mIoU = 0.633891
best single late checkpoint ep250 HC-Val mIoU = 0.710648
TWA-4 相对 ep250 mIoU = -0.076757
```

也就是说，TWA 的 HC-Val 正收益不能干净解释为 trajectory averaging 机制，因为一个单 checkpoint 更强。

### 1.2 ep250 是 hard-clutter specialist，不是 Full-safe candidate

ep250 HC-Val 很强，但 Gate-LS-A 失败：

```text
ep250 Full mIoU = 0.831139
OHEM-400 Full mIoU = 0.834393
Full delta mIoU = -0.003254
Full delta Precision = -0.000187
```

这说明 ep250 的本质是：

```text
在 hard-clutter split 上更保守、更少 false alarm；
但在 Full split 上已经损伤了整体 segmentation / precision anchor。
```

所以 ep250 不能继续作为 AAAI 主候选。

### 1.3 ep300 的选择是 numerical tie，不是方法发现

Gate-TWA-E2-FSC 选中 ep300，因为：

```text
ep300 HC-Val delta mIoU = +0.029305
TWA-4 HC-Val delta mIoU = +0.029101
差值 = +0.000204
```

这个差值过小，不能支撑“方法从 TWA-4 切换到 ep300”。

它只能说明：

```text
late training trajectory 上不同 checkpoint 的 hard-clutter 表现有微小波动。
```

不能说明：

```text
ep300 是一个新的结构性方法。
```

### 1.4 TCSR-A 证明 train-only sparse hard-clutter signal 基本不存在

Gate-TCSR-A 结果：

```text
num_images = 697
num_images_with_neg = 1       要求 >= 50
neg_pixels_total = 130        要求 >= 500
target_leakage_pixels_total = 0
neg_protect_overlap_pixels_total = 0
```

这不是代码问题，而是机制失败：

```text
train split 上没有足够多 target-safe sparse hard-clutter negative signal。
```

所以不能继续写 loss.py / net.py / train.py。否则就是在没有训练信号的情况下强行加监督，最终大概率污染 target evidence。

### 1.5 TCE-4 internal 稳，但 external Pd 失败

TCE-4 final aggregation：

```text
Full:    3/3 pass, mean delta mIoU +0.002820
HC-Val:  3/3 pass, mean delta mIoU +0.017937, FA ppm -38.146973
HC-Test: 3/3 pass, mean delta mIoU +0.014697, FA ppm -25.893703
HC threshold-matched: 12/12 PASS
```

但是 F3 external once：

```text
external_nuaa_sirst: min_delta_Pd = -0.018348624
external_irstd_1k:  min_delta_Pd = -0.013468013
```

按你们的文档规则，任一 external split 出现 `min_delta_Pd < 0` 就是：

```text
F3_FAIL_NO_REDESIGN
```

这说明 TCE-4 的内部收益没有跨域保持 Pd。

---

## 2. 为什么越往后越差

## 2.1 因为 gate 越来越接近真实泛化，不是因为代码越来越差

前面的 PASS 大多发生在：

```text
seed42
internal Full
internal HC-Val
known seed set
known hard-clutter split
```

后面的 FAIL 发生在：

```text
mechanism comparison
Full-safe control
train-only signal audit
blind/external once
Pd non-regression
```

越后面的 gate 越难，因为它们不再问：

```text
这个结果在某个 internal split 上是不是涨了？
```

而是在问：

```text
这个方法是否有稳定机制？
是否避免 post-hoc selection？
是否不牺牲 Pd？
是否能跨数据集泛化？
```

所以看起来是“越改越差”，本质是“越验证越真实”。

---

## 2.2 因为当前路线一直在 FA / Pd 的 Pareto 边界附近移动

IRSTD 里 Pd 是非常脆弱的指标。一个小目标可能只有很少像素，模型只要稍微保守一点，就可能：

```text
FA 降低
Precision 上升
mIoU 小幅上升
但某些弱目标整目标消失，Pd 下降
```

TCE-4 internal 的表现像是：

```text
保守化预测成功压掉内部 hard clutter。
```

但 external 的表现说明：

```text
这种保守化也压掉了外部数据中的弱目标 / 低对比目标 / 形态不同目标。
```

这就是为什么 external split 上 `min_delta_Pd < 0` 是致命问题。

---

## 2.3 因为 MSHNetOHEM anchor 太强，后续方法只有很小增量空间

MSHNetOHEM 作为 anchor 已经很强，内部 Full 的正收益只有：

```text
TCE-4 mean Full delta mIoU +0.002820
```

这个量级很小。相比之下，external Pd 下降是：

```text
-0.0183
-0.0135
```

也就是说：

```text
内部 mIoU 小增益 < 外部 Pd 风险
```

这不是一个稳定 AAAI 主结果。它说明候选方法没有形成足够大的泛化 margin。

---

## 2.4 因为多次修改其实是在追逐 gate failure，而不是从结构上解决问题

路线变化大致是：

```text
TWA-4 no-BN
  -> Gate-E 发现 best single checkpoint ep250 更强
  -> LateSnapshot ep250
  -> Gate-LS-A 发现 ep250 Full unsafe
  -> E2-FSC 选 ep300
  -> 发现 ep300 只比 TWA-4 高 +0.000204
  -> TCSR 想把 TCE signal 蒸馏为 train-only sparse bank
  -> Gate-A 发现 sparse bank 基本为空
  -> 回到 TCE-4 final
  -> external Pd 失败
```

这里每一步都有局部合理性，但整体有一个问题：

```text
每一步都在修补上一个 gate 暴露的问题，
却没有获得新的、足够强的、可泛化的结构性训练信号。
```

这会导致代码越写越多，但方法越来越像：

```text
post-hoc selection + diagnostic controls + failure guards
```

而不是一个简洁的主方法。

---

## 2.5 因为 train split 上 hard-clutter negative signal 太稀疏

TCSR-A 的失败非常关键。

它说明：如果坚持 train-only、target-safe、far-background、local-peak、TCE/OHEM disagreement 这些安全约束，那么真正可用的 negative supervision 几乎没有：

```text
697 张图里只有 1 张有 negative。
```

这意味着两件事：

第一，之前很多 hard-clutter suppression 类方法失败不是偶然。

第二，如果强行放宽 bank 条件，表面上会得到更多 negative，但很可能引入：

```text
target leakage
near-target suppression
weak target 被错当 clutter
external Pd 下降
```

所以 TCSR 不能继续救。

---

## 2.6 因为 TCE-4 是 oracle-like internal smoother，不是稳定外部方法

TCE-4 的强项是：

```text
多个 checkpoint 的轨迹共识可以压低不稳定 prediction peaks。
```

这在 internal HC-Val / HC-Test 上有效，因为 hard clutter 的分布和开发过程一致。

但 external 数据里，某些真实小目标本身也可能表现为：

```text
低对比
小面积
形态不稳定
跨 checkpoint 不一致
```

TCE 共识会把这些目标也当成不稳定响应压低，于是：

```text
FA 可能继续下降，
但 Pd 掉了。
```

这正是 F3 external 的现象。

---

## 2.7 因为 SIRST3 数据完整性问题暴露了 external manifest 风险

SIRST3：

```text
test_SIRST3.txt 共 1079 条
365 条缺 mask
1 条缺 image
```

这不是方法失败的主要原因，因为前两个 external split 已经因 Pd drop 触发 F3_FAIL_NO_REDESIGN。

但它暴露了另一个问题：

```text
external split manifest 没有在 once-lock 前完成完整 data integrity audit。
```

以后 external once 之前必须先做：

```text
manifest-only integrity preflight
不跑模型
不算指标
只检查 image/mask/list 是否完整
```

这不属于 rescue，也不改变模型，但属于实验卫生问题。

---

## 3. 哪些不是主要原因

### 3.1 不是 pytest 少了

你们每一步都做了：

```text
py_compile
pytest
git diff --check
```

这些只能证明：

```text
脚本语法、schema、简单逻辑没有明显错误。
```

它不能证明：

```text
方法会提升指标。
```

所以 pytest 通过但 gate fail 是正常的。

### 3.2 不是 checker 太严格

如果 AAAI 主张是：

```text
降低 false alarms，同时不牺牲 Pd。
```

那么 external Pd negative 不能放宽。

放宽 Pd gate 会把方法变成：

```text
以 recall 换 false alarm。
```

这与当前目标相反。

### 3.3 不是没有继续调 threshold / seed / epoch

不能这样救：

```text
哪个 seed 好用哪个
哪个 epoch 好用哪个
哪个 threshold 外部好用哪个
SIRST3 缺 mask 就换 labeled subset
```

这些都会把最终结果变成选择过的结果，不再是 frozen method evaluation。

---

## 4. 真正的根因归纳

### 根因一：目标太难，而且 baseline 太强

当前目标同时要求：

```text
Full 不退化
HC false alarm 下降
Precision 上升
Pd 不下降
单模型 / 单 forward 优先
外部泛化
```

对一个已经很强的 MSHNetOHEM baseline 来说，这些目标几乎是在 Pareto frontier 上继续挤增益。

### 根因二：内部 hard-clutter 收益主要来自保守化，而不是可靠 target/clutter 解耦

TWA、ep250、ep300、TCE-4 的共同特征都是：

```text
让预测更保守或更共识化。
```

它们并没有真正学习：

```text
哪些 high-confidence local peaks 是 clutter，哪些是弱真实目标。
```

所以 external 上一旦真实目标也呈现“不稳定响应”，Pd 就会下降。

### 根因三：train-only hard-clutter supervision source 几乎为空

TCSR-A 失败说明：

```text
如果不泄漏、不调参、不用 external/val label，train split 无法提供足够 sparse hard-clutter negatives。
```

这堵死了“安全蒸馏 TCE clutter reliability 到单模型”的路线。

### 根因四：后期方案越来越像选择器，不像方法

E2-FSC、ep300、best snapshot、best seed 都是选择机制，不是结构机制。

选择机制能提高 internal number，但很难形成 AAAI 方法贡献，也容易在 external 上失败。

### 根因五：external once 才是真实裁决

F3 的 external Pd drop 说明：

```text
内部结果可以报告为 diagnostic，
但不能支撑 final method claim。
```

---

## 5. 现在应该怎么处理

## 5.1 不要继续开发新方法

当前只剩 15 天，不建议再开新结构。

特别不要做：

```text
TCE threshold rescue
external-specific calibration
SIRST3 labeled subset 替换
seed / checkpoint / epoch 搜索
TCSR threshold 放宽
重新设计 loss / verifier / suppression head
```

这些很可能继续产生新的局部 PASS 和最终 FAIL。

## 5.2 应该生成一个 F3 fail final report

虽然 F3 final report 未生成，但现在应该生成一个“停止型 final report”，不是继续 evaluation。

建议新增或运行只读汇总：

```text
docs/internal/tce_final/gate_tce_f3_fail_summary.json
```

内容应包括：

```json
{
  "gate": "Gate-TCE-F3-blind-external-once",
  "decision": "F3_FAIL_NO_REDESIGN",
  "once_lock_created": true,
  "preflight_pass": true,
  "failed_splits": {
    "external_nuaa_sirst": {
      "min_delta_Pd": -0.018348624,
      "fail_reason": "Pd regression"
    },
    "external_irstd_1k": {
      "min_delta_Pd": -0.013468013,
      "fail_reason": "Pd regression"
    }
  },
  "not_completed_splits": {
    "external_sirst3": {
      "reason": "manifest integrity failure",
      "missing_masks": 365,
      "missing_images": 1,
      "total_entries": 1079
    }
  },
  "forbidden_next_actions": [
    "threshold search",
    "seed search",
    "checkpoint search",
    "split redefinition",
    "external rescue",
    "new model training"
  ]
}
```

这样 README 里就不会留下 `once-lock STARTED` 的悬空状态。

建议把 once-lock 状态从：

```text
STARTED
```

只读更新为：

```text
STOPPED_BY_F3_PD_REGRESSION
```

这不是继续实验，而是关闭状态机。

## 5.3 README 状态应改为最终停止，而不是 pending

README 顶部应写：

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Final frozen candidate TCE-4-OHEM reached F3 blind/external once and failed.

Decision:
- STOP_TCE4_AT_F3_EXTERNAL_PD_REGRESSION
- STOP_TCSR_AT_BANK_AUDIT
- STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD

Reason:
- TCE-4 internal evidence is positive across Full / HC-Val / HC-Test and threshold-matched hard splits.
- However, F3 external once produced Pd regression on external_nuaa_sirst and external_irstd_1k.
- SIRST3 also has manifest integrity issues and is not used for rescue or split redefinition.

Forbidden:
- threshold search
- seed search
- checkpoint search
- external split redefinition
- BN tuning
- new training
- new verifier / suppression / loss structure
```

---

## 6. 如果还要投稿，应该如何定位

严格按当前结果，不能再把 TCE-4 写成最终成功主方法。

还可以保留的论文素材是：

```text
1. MSHNetOHEM 是强 anchor。
2. TCE trajectory consensus 在 internal Full / HC-Val / HC-Test 上稳定降低 FA。
3. TWA / late snapshot / TCSR 逐步证明：
   - weight-space compression 不稳；
   - single checkpoint promotion 是 post-hoc；
   - train-only sparse hard-clutter labels 不足；
   - external Pd 是最终瓶颈。
4. 这可以形成一个很完整的 negative-analysis / reliability study。
```

但如果 AAAI full paper 需要一个清晰、外部泛化成功的新方法，那么当前严格 gate 下还没有。

---

## 7. 一句话总结

```text
越修改越差，不是因为代码质量越来越差，
而是因为每次修改都在解决上一个 internal gate 暴露的局部问题，
却没有获得一个足够强、train-only、target-safe、跨域稳定的 hard-clutter/target 解耦机制。

MSHNetOHEM 已经很强；
后续方法主要通过保守化降低 FA；
内部 hard-clutter split 上这有效；
外部数据上它会压掉弱真实目标，导致 Pd 下降。

所以当前正确动作不是继续救，
而是关闭 TCE-4 F3 状态、记录失败原因、停止新开发。
```
