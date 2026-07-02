# OHCM-MSHNet：停止 seed / checkpoint 选择，转向结构性 hard-clutter 方案 TCSR

> 适用场景：AAAI 冲刺期，已有 TWA / LateSnapshot / E2-FSC 结果，但当前路线开始把“checkpoint / seed / epoch 选择”当作方法。  
> 本文只处理当前失败点：**E2-FSC 虽然 PASS 并选中 ep300，但这个选择本身不是结构性方法，不能继续推进到 seed43/44 作为 AAAI 主线。**

---

## 0. 当前输入事实

### 0.1 已有 gate 结果

```text
TWA-4 no-BN:
  seed42 Full safe
  seed42 HC-Val positive
  Gate-E failed because best single late checkpoint ep250 HC-Val much stronger

LateSnapshot-ep250:
  HC-Val very strong
  Full unsafe
  Gate-LS-A failed

Gate-TWA-E2-FSC:
  PASS
  selected_candidate = LateSnapshot-ep300
```

### 0.2 ep300 当前指标

```text
ep300 Full delta vs OHEM:
  mIoU      +0.004852
  Precision +0.002231
  Pd        +0.001058
  FA ppm    -1.378806

ep300 HC-Val delta vs OHEM:
  mIoU      +0.029305
  Precision +0.032006
  Pd        +0.0
  FA ppm    -50.862630

TWA-4 HC-Val delta mIoU:
  +0.029101

ep300 - TWA-4 HC-Val delta mIoU:
  +0.000204
```

这个 `+0.000204` 是核心问题。它在方法学上不能支撑“切换到 ep300”。

---

## 1. 先回答：哪个随机种子好就用哪个，固定就好了吧？

不行。

### 1.1 随机种子不是方法结构

随机种子控制的是：

```text
initialization
数据加载顺序
augmentation 顺序
CUDA / cuDNN 非确定性路径
优化轨迹细节
```

如果做法是：

```text
跑 seed42 / seed43 / seed44 / seed45 / ...
哪个结果最好就说方法使用哪个 seed
```

那这个 seed 已经变成了一个被 validation / test 结果选择过的超参数。固定它只能让结果可复现，不能让选择过程干净。

### 1.2 对 AAAI 来说，seed 只能有两种合法角色

```text
Role A: development seed
  例如 seed42 只用于设计和早停 gate。
  方法、epoch、threshold、loss、结构冻结后，seed42 不再参与选择。

Role B: confirmation seeds
  例如 seed42/43/44 作为预注册固定集合。
  报告 mean/std 或逐 seed paired delta。
```

不能有第三种角色：

```text
Role C: seed search space
  选最好的 seed 当最终方法。
```

Role C 对论文很危险。审稿人会认为这是 training stochasticity selection，而不是 method improvement。

### 1.3 当前 ep300 也是类似问题

ep300 不是新结构，只是同一个 MSHNetOHEM 训练轨迹上的某个 epoch。

如果当前结论变成：

```text
我们发现 seed42 的 ep300 比 TWA-4 高 0.000204 mIoU，所以主方法是 ep300。
```

这不是一个 AAAI 方法。它最多是一个 diagnostic observation：

```text
hard-clutter 性能沿训练轨迹存在波动；某些 late checkpoint 更偏向 hard-clutter。
```

这条 observation 有价值，但不能直接当主方法。

---

## 2. 当前 E2-FSC 方案到底解决了什么问题？

### 2.1 E2-FSC 解决的是“Full-safe single checkpoint control”问题

原 Gate-E 失败原因是：

```text
TWA-4 HC-Val 不如 best single late checkpoint ep250。
```

但是 ep250 后来被证明：

```text
HC-Val 很强
Full unsafe
```

所以 E2-FSC 的修正逻辑是合理的：

```text
不要拿 Full-unsafe checkpoint 否定 TWA；
只在 Full-safe late checkpoints 里面比较 HC-Val。
```

这解决的是一个统计控制问题：

```text
避免被 ep250 这种 hard-clutter specialist 误导。
```

### 2.2 但 E2-FSC 没有解决“方法结构”问题

E2-FSC 没有提供新的 false-alarm 机制。它只是在已有候选中排序：

```text
ep300
ep350
TWA-4
OHEM-400
```

这类排序 gate 可以防止错误推进，但不能产生可投稿的结构贡献。

### 2.3 当前 checker 的具体缺陷

现在 checker 采用：

```text
在 eligible candidates 中按 HC-Val mIoU 最大者选择。
```

这导致：

```text
ep300 HC-Val delta mIoU = +0.029305
TWA-4 HC-Val delta mIoU = +0.029101
差值 = +0.000204
```

于是 checker 选择 ep300。

这个选择太敏感，应该被视为 numerical tie，而不是 method switch。

### 2.4 应立即修正 E2-FSC checker

不要把 `+0.000204` 当成切换证据。应加入最小实用差异阈值：

```text
min_practical_hc_miou_switch_margin = 0.005
```

如果 single checkpoint 只比 TWA-4 高不到 0.005 mIoU，则输出：

```text
selection_status = "TIE_OR_NUMERICAL_NOISE"
promotion_allowed = false
next_allowed_gate = "STOP_POSTHOC_CHECKPOINT_SELECTION"
```

不是让 TWA-4 自动复活，也不是让 ep300 继续推进，而是说明：

```text
post-hoc epoch / checkpoint selection 不足以构成 AAAI 主方法。
```

---

## 3. 当前真正要解决的问题是什么？

总目标不是选 seed，也不是选 epoch。总目标是：

```text
在保持 MSHNetOHEM 小目标 evidence anchor 的情况下，降低复杂背景 false alarms，提升 Precision，不牺牲 Pd，并保持单模型单 forward。
```

结合所有结果，现在可以归纳为三个硬约束。

### 3.1 约束一：不能污染 evidence branch

之前 PFR / ERD / suppression-style 方案多次出现：

```text
Full mIoU / Precision / FA 退化
Pd 或 target support 被破坏
```

所以任何新结构都必须满足：

```text
MSHNetOHEM 原始 segmentation evidence path 保持主导。
新信号只能作为受控辅助训练信号，不能在 inference 阶段覆盖 evidence。
```

### 3.2 约束二：false alarm 是 hard-clutter local peak / component 问题，不是普通 dense BCE 问题

OHEM 已经在做 hard negative mining，但它主要是 pixel-level hard negative。当前现象说明：

```text
像素级 OHEM 可以保持 Full 强基线；
但 HC-Val 上仍然存在 target-like clutter false alarms。
```

所以新方法需要对齐 evaluation failure：

```text
局部峰值
连通 component
far-background 高置信区域
```

而不是在全图 dense soft label 上平均。

### 3.3 约束三：TCE 有真实机制信号，但不能直接作为 4x inference 方法

TCE 的价值不是“ensemble 可以提高指标”，而是它暴露了一个机制：

```text
训练轨迹共识可以区分稳定 target evidence 与不稳定 clutter activation。
```

TWA 和 ep300 都是在尝试压缩这个机制，但它们失败的原因是：

```text
它们没有显式学习“哪里是 TCE 认为不可靠的 hard clutter”。
```

---

## 4. 为什么现有 TWA / ep300 路线不应继续

### 4.1 TWA 的问题

TWA 的方法含义是：

```text
weight-space trajectory averaging
```

它可以解释为 trajectory compression，但 Gate-E 暴露：

```text
HC-Val 收益未必来自 averaging；
单个 late checkpoint 也能达到类似甚至略高表现。
```

所以它的机制解释不够稳。

### 4.2 ep300 的问题

ep300 的方法含义更弱：

```text
same architecture
same loss
same training
only stop epoch changed from 400 to 300
```

如果把 ep300 当主方法，论文贡献会退化成：

```text
early stopping helps hard clutter.
```

这不足以支撑 AAAI 主线，尤其是 ep300 比 TWA-4 只高 `0.000204`。

### 4.3 随机 seed / checkpoint / epoch 都不能成为核心结构

这三者都属于 training trajectory realization，不是 architecture / loss / inference mechanism：

```text
seed selection      -> stochastic selection
checkpoint selection -> epoch selection
TWA selection        -> checkpoint combination selection
```

它们都可以做 diagnostic control，但不应该继续作为最终方法。

---

## 5. 推荐的新结构：TCSR-MSHNet

### 5.1 名称

```text
TCSR-MSHNet
Trajectory-Consensus Sparse Reliability MSHNet
```

也可以写成更论文化的名字：

```text
Trajectory-Consensus Sparse Reliability Distillation for IRSTD
```

### 5.2 一句话方法

> 用已有 TCE 轨迹共识在 train split 上构建 target-safe sparse reliability bank，只对 far-background 的不稳定 high-confidence local peaks 施加辅助抑制，同时用 GT / consensus target support 保护小目标 evidence；测试时仍然是单个 MSHNetOHEM-style forward。

### 5.3 它解决的问题

TCSR 不是选 seed，也不是选 checkpoint。它直接解决：

```text
hard-clutter false alarm local peaks 如何被训练目标识别和抑制，
同时不破坏 target evidence。
```

### 5.4 为什么它比 dense TCE distillation 合理

之前 dense TCE distillation 失败的根因是：

```text
TCE 与 OHEM 在全图大部分背景像素上几乎相同；
全图平均 diff 很小；
真正有用的差异集中在极少数 local peaks / hard clutter pixels 上。
```

所以不能做：

```text
dense TCE soft label regression
```

应该做：

```text
sparse local-peak reliability distillation
```

也就是只学习 TCE 与 OHEM 在 hard-clutter 高风险位置上的差异。

---

## 6. TCSR 的结构设计

### 6.1 Inference 结构

不改 inference graph：

```text
input image
  -> MSHNet backbone / decoder
  -> final logit
  -> sigmoid foreground probability
  -> threshold 0.5
```

测试阶段没有：

```text
TCE ensemble
extra verifier
post-hoc suppression
threshold search
BN recalibration
checkpoint averaging
multi-forward
```

### 6.2 Training 结构

训练阶段增加一个 sparse reliability auxiliary loss：

```text
L_total = L_MSHNetOHEM
        + lambda_neg(t)     * L_sparse_clutter_neg
        + lambda_protect(t) * L_target_protect
        + lambda_agree(t)   * L_consensus_agree
```

其中：

```text
L_MSHNetOHEM:
  原有 segmentation + OHEM loss，保持 evidence anchor。

L_sparse_clutter_neg:
  只在 train split 中 target-safe far-background hard local peaks 上，
  让 student logit 靠近 background。

L_target_protect:
  在 GT target support / dilated target support 上，
  禁止 auxiliary loss 降低 target probability。

L_consensus_agree:
  在 TCE 与 OHEM 高一致区域，弱约束 student 不偏离 anchor。
```

### 6.3 Sparse reliability bank

离线构建 train-only bank。每张训练图保存：

```text
image_id
neg_weight:       far-background sparse hard-clutter weight map
protect_weight:   target / consensus-target protection weight map
tce_mean:         TCE mean foreground probability, optional
tce_std:          TCE trajectory uncertainty, optional
anchor_prob:      OHEM-400 foreground probability, optional
metadata:         counts / leakage / thresholds
```

### 6.4 hard negative 定义

核心思想：

```text
OHEM / student 高置信
TCE consensus 低或不稳定
距离 GT target 足够远
位于 local peak / component-risk 区域
```

建议先用完全离线、固定定义：

```python
far_bg = dilate(gt, radius=7) == 0
anchor_high = p_ohem >= 0.50
tce_low = p_tce_mean <= 0.35
disagreement = (p_ohem - p_tce_mean) >= 0.15
local_peak = p_ohem == maxpool(p_ohem, kernel=7)
neg_seed = far_bg & anchor_high & tce_low & disagreement & local_peak
```

然后把 local peak 膨胀成小区域：

```python
neg_weight = dilate(neg_seed, radius=2) * clamp(p_ohem - p_tce_mean, 0, 1)
```

如果 `p_ohem` 太稀疏，可以改用 TCE variance：

```python
uncertain_clutter = far_bg & (p_tce_std >= std_threshold) & local_peak
```

但第一版只允许一种固定定义，不要 grid search。

### 6.5 target protection 定义

```python
target_support = dilate(gt, radius=2)
consensus_target = (p_tce_mean >= 0.60) & near_gt
protect_weight = target_support | consensus_target
```

保护损失不要强行提高所有 target 外扩像素，只做下界保护：

```python
p_student >= p_ref - margin
```

其中：

```python
p_ref = max(gt_mask, p_tce_mean, p_ohem)
margin = 0.05
```

---

## 7. 为什么 TCSR 比继续 ep300 更像 AAAI 方法

| 路线 | 本质 | 是否结构性解决 hard clutter | AAAI 风险 |
|---|---|---:|---|
| 选好 seed | stochastic selection | 否 | 极高 |
| ep300 | epoch selection | 否 | 很高 |
| TWA-4 | weight averaging / model soup style | 间接 | 中高 |
| TCE-4 | ensemble / trajectory oracle | 有信号但 4x inference | 中 |
| TCSR | train-time trajectory-consensus sparse reliability supervision | 是 | 相对最低 |

TCSR 的论文叙述可以是：

```text
TCE reveals reliable trajectory consensus but is computationally expensive.
Dense distillation fails because the useful discrepancy is sparse.
We therefore distill trajectory reliability only on target-safe hard-clutter local peaks, while protecting target evidence with an anchor-preserving constraint.
```

这个叙述有机制、有结构、有失败路线动机，也能解释为什么之前 dense TCD / TWA / ep300 都不够。

---

## 8. 立即停止和保留的分支

### 8.1 停止

```text
STOP_POSTHOC_SEED_SELECTION
STOP_LATE_SNAPSHOT_EP300_PROMOTION
STOP_TWA_E2_FSC_AS_FINAL_METHOD
```

### 8.2 保留为 diagnostic controls

```text
ep250:
  hard-clutter specialist, Full unsafe

ep300:
  Full-safe diagnostic late snapshot, not method

TWA-4:
  trajectory averaging diagnostic, not final method unless later structural branch fails and paper降级

TCE-4:
  trajectory consensus oracle / teacher source
```

---

## 9. 对现有 E2-FSC checker 的修正

文件：

```text
tools/official/check_twa_gate_e2_fullsafe_single_control.py
```

### 9.1 新增参数

```python
parser.add_argument(
    "--min_hc_miou_switch_margin",
    type=float,
    default=0.005,
    help="Minimum HC-Val mIoU advantage required to switch from TWA-4 to a single late snapshot.",
)
parser.add_argument(
    "--disable_posthoc_single_promotion",
    action="store_true",
    help="If set, do not promote a single late snapshot to the next multi-seed gate; keep it diagnostic only.",
)
```

### 9.2 修改选择逻辑

替换原来的“谁 HC-Val mIoU 最大选谁”：

```python
best = max(eligible, key=lambda x: x["hcval_delta"]["mIoU"])
twa4 = next(x for x in eligible if x["name"] == "TWA-4-noBN")

best_adv_over_twa = (
    best["hcval_delta"]["mIoU"] - twa4["hcval_delta"]["mIoU"]
)

is_single_snapshot = best["name"].startswith("LateSnapshot-")

if is_single_snapshot and best_adv_over_twa < args.min_hc_miou_switch_margin:
    selected = None
    selection_status = "TIE_OR_NUMERICAL_NOISE_NO_SWITCH"
    promotion_allowed = False
    next_allowed_gate = "STOP_POSTHOC_CHECKPOINT_SELECTION"
elif is_single_snapshot and args.disable_posthoc_single_promotion:
    selected = best["name"]
    selection_status = "DIAGNOSTIC_SINGLE_SNAPSHOT_WINNER_NOT_PROMOTED"
    promotion_allowed = False
    next_allowed_gate = "STOP_POSTHOC_CHECKPOINT_SELECTION"
else:
    selected = best["name"]
    selection_status = "SELECTED_WITH_PRACTICAL_MARGIN"
    promotion_allowed = True
    next_allowed_gate = f"Gate-multiseed-{selected}"
```

### 9.3 当前数据下应该输出

```json
{
  "gate": "Gate-TWA-E2-FSC",
  "gate_pass": true,
  "best_eligible_by_hcval": "LateSnapshot-ep300",
  "twa4_hcval_delta_miou": 0.029101,
  "best_hcval_delta_miou": 0.029305,
  "best_advantage_over_twa4": 0.000204,
  "min_hc_miou_switch_margin": 0.005,
  "selection_status": "TIE_OR_NUMERICAL_NOISE_NO_SWITCH",
  "promotion_allowed": false,
  "next_allowed_gate": "STOP_POSTHOC_CHECKPOINT_SELECTION"
}
```

---

## 10. 新增 TCSR 代码结构

### 10.1 新增文件清单

```text
utils/tcsr_bank.py
loss_tcsr.py 或直接合入 loss.py
tools/official/build_tcsr_sparse_bank.py
tools/official/check_tcsr_bank_gate_a.py
tools/official/check_tcsr_activation_gate_b.py
scripts/official/run_tcsr_gate_a_seed42_bank.sh
scripts/official/run_tcsr_gate_b_seed42_activation.sh
scripts/official/run_tcsr_seed42_train_eval.sh
tests/test_tcsr_bank.py
tests/test_tcsr_loss.py
tests/test_tcsr_gate_a_checker.py
tests/test_tcsr_status_schema.py
```

建议先不要直接训练 400 epoch。先只实现：

```text
Gate-TCSR-A: train-only sparse bank audit
```

Gate-A 过了再写训练脚本。

---

## 11. `utils/tcsr_bank.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class TCSRBankItem:
    image_id: str
    neg_weight: torch.Tensor
    protect_weight: torch.Tensor
    ref_prob: Optional[torch.Tensor] = None


class TCSRBank:
    """Load train-only trajectory-consensus sparse reliability maps.

    Files are expected as:
        bank_root/<image_id>.pt

    Each .pt contains:
        neg_weight: Tensor[1,H,W] or Tensor[H,W]
        protect_weight: Tensor[1,H,W] or Tensor[H,W]
        ref_prob: optional Tensor[1,H,W] or Tensor[H,W]
    """

    def __init__(self, bank_root: str | Path):
        self.bank_root = Path(bank_root)
        if not self.bank_root.exists():
            raise FileNotFoundError(f"TCSR bank not found: {self.bank_root}")

    @staticmethod
    def _normalize_image_id(image_id) -> str:
        if isinstance(image_id, bytes):
            image_id = image_id.decode("utf-8")
        image_id = str(image_id)
        return Path(image_id).stem

    @staticmethod
    def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected CHW or HW tensor, got shape={tuple(x.shape)}")
        return x.float()

    def get(self, image_id: str, device=None, size=None) -> TCSRBankItem:
        key = self._normalize_image_id(image_id)
        path = self.bank_root / f"{key}.pt"
        if not path.exists():
            raise KeyError(f"TCSR bank item missing for image_id={image_id}, path={path}")

        data = torch.load(path, map_location="cpu")
        neg_weight = self._ensure_chw(data["neg_weight"])
        protect_weight = self._ensure_chw(data["protect_weight"])
        ref_prob = data.get("ref_prob", None)
        if ref_prob is not None:
            ref_prob = self._ensure_chw(ref_prob)

        if size is not None:
            neg_weight = F.interpolate(
                neg_weight[None], size=size, mode="nearest"
            )[0]
            protect_weight = F.interpolate(
                protect_weight[None], size=size, mode="nearest"
            )[0]
            if ref_prob is not None:
                ref_prob = F.interpolate(
                    ref_prob[None], size=size, mode="bilinear", align_corners=False
                )[0]

        if device is not None:
            neg_weight = neg_weight.to(device)
            protect_weight = protect_weight.to(device)
            if ref_prob is not None:
                ref_prob = ref_prob.to(device)

        return TCSRBankItem(
            image_id=key,
            neg_weight=neg_weight,
            protect_weight=protect_weight,
            ref_prob=ref_prob,
        )

    def batch(self, image_ids, device, size) -> Dict[str, torch.Tensor]:
        items = [self.get(i, device=device, size=size) for i in image_ids]
        neg_weight = torch.stack([x.neg_weight for x in items], dim=0)
        protect_weight = torch.stack([x.protect_weight for x in items], dim=0)
        if items[0].ref_prob is not None:
            ref_prob = torch.stack([x.ref_prob for x in items], dim=0)
        else:
            ref_prob = None
        return {
            "neg_weight": neg_weight,
            "protect_weight": protect_weight,
            "ref_prob": ref_prob,
        }
```

---

## 12. `loss.py` 新增 TCSR loss

建议直接加到 `loss.py`，避免新文件 import 路径复杂。

```python
class TrajectoryConsensusSparseLoss(nn.Module):
    """TCSR auxiliary loss on top of MSHNetOHEM.

    This loss does not change inference. It only adds train-time sparse
    reliability supervision from a prebuilt trajectory-consensus bank.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        bank_path: str,
        lambda_neg: float = 0.05,
        lambda_protect: float = 0.10,
        lambda_agree: float = 0.02,
        start_epoch: int = 40,
        ramp_epochs: int = 40,
        protect_margin: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        from utils.tcsr_bank import TCSRBank

        self.base_loss = base_loss
        self.bank = TCSRBank(bank_path)
        self.lambda_neg = float(lambda_neg)
        self.lambda_protect = float(lambda_protect)
        self.lambda_agree = float(lambda_agree)
        self.start_epoch = int(start_epoch)
        self.ramp_epochs = int(ramp_epochs)
        self.protect_margin = float(protect_margin)
        self.eps = float(eps)

    def _ramp(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        return min(1.0, float(epoch - self.start_epoch) / float(max(1, self.ramp_epochs)))

    @staticmethod
    def _weighted_mean(x, w, eps=1e-6):
        return (x * w).sum() / w.sum().clamp_min(eps)

    def forward(self, masks, final_logit, gt_mask, epoch=0, image_ids=None, **kwargs):
        base_out = self.base_loss(
            masks,
            final_logit,
            gt_mask,
            epoch=epoch,
            image_ids=image_ids,
            **kwargs,
        )
        base_total = base_out["total"] if isinstance(base_out, dict) else base_out

        if image_ids is None:
            raise ValueError("TCSR loss requires image_ids to load sparse reliability bank items.")

        size = final_logit.shape[-2:]
        bank = self.bank.batch(image_ids, device=final_logit.device, size=size)
        neg_w = bank["neg_weight"].float()
        protect_w = bank["protect_weight"].float()
        ref_prob = bank["ref_prob"]

        # Make shapes BCHW.
        if neg_w.ndim == 3:
            neg_w = neg_w[:, None]
        if protect_w.ndim == 3:
            protect_w = protect_w[:, None]
        if ref_prob is not None and ref_prob.ndim == 3:
            ref_prob = ref_prob[:, None]

        ramp = self._ramp(epoch)
        zero = final_logit.sum() * 0.0

        # Sparse clutter negative: push selected far-background local peaks to background.
        loss_neg_map = F.binary_cross_entropy_with_logits(
            final_logit,
            torch.zeros_like(final_logit),
            reduction="none",
        )
        loss_neg = (
            self._weighted_mean(loss_neg_map, neg_w, self.eps)
            if float(neg_w.detach().sum().cpu()) > 0
            else zero
        )

        # Target / consensus protection: do not let auxiliary suppression lower protected prob.
        if ref_prob is not None:
            p = torch.sigmoid(final_logit)
            protect_violation = F.relu((ref_prob - self.protect_margin) - p).pow(2)
            loss_protect = (
                self._weighted_mean(protect_violation, protect_w, self.eps)
                if float(protect_w.detach().sum().cpu()) > 0
                else zero
            )

            agree_w = torch.clamp(protect_w + (1.0 - neg_w) * 0.05, 0.0, 1.0)
            loss_agree = self._weighted_mean((p - ref_prob).pow(2), agree_w, self.eps)
        else:
            loss_protect = zero
            loss_agree = zero

        total = (
            base_total
            + ramp * self.lambda_neg * loss_neg
            + ramp * self.lambda_protect * loss_protect
            + ramp * self.lambda_agree * loss_agree
        )

        out = {
            "total": total,
            "base": base_total.detach(),
            "tcsr_neg": loss_neg.detach(),
            "tcsr_protect": loss_protect.detach(),
            "tcsr_agree": loss_agree.detach(),
            "tcsr_ramp": torch.tensor(ramp, device=final_logit.device),
            "tcsr_neg_pixels": neg_w.sum().detach(),
            "tcsr_protect_pixels": protect_w.sum().detach(),
        }
        if isinstance(base_out, dict):
            for k, v in base_out.items():
                if k not in out and k != "total":
                    out[f"base_{k}"] = v
        return out
```

---

## 13. `net.py` 修改

### 13.1 增加 model name

```python
SUPPORTED_MODEL_NAMES = (
    ...
    "MSHNetOHEM",
    "MSHNetTCSR",
    ...
)
```

### 13.2 保持同一 MSHNet inference graph

在 MSHNet variant branch 里加入：

```python
MSHNET_VARIANT_NAMES = (
    "MSHNet",
    "MSHNetFocal",
    "MSHNetOHEM",
    "MSHNetTopKNeg",
    "MSHNetSPSOHEM",
    "MSHNetTCSR",
)
```

variant map：

```python
variant = {
    "MSHNet": "baseline",
    "MSHNetFocal": "focal",
    "MSHNetOHEM": "ohem",
    "MSHNetTopKNeg": "topk_neg",
    "MSHNetSPSOHEM": "sps_ohem",
    "MSHNetTCSR": "ohem",
}[model_name]
```

然后在 base loss 之后包一层 TCSR：

```python
base_loss = MSHNetVariantLoss(
    variant=variant,
    mshnet_warm_epoch=self.mshnet_warm_epoch,
    lambda_variant=float(loss_cfg.get("lambda_variant", 0.2)),
    ohem_ratio=float(loss_cfg.get("ohem_ratio", 0.01)),
    # keep existing args
)

if model_name == "MSHNetTCSR":
    self.cal_loss = TrajectoryConsensusSparseLoss(
        base_loss=base_loss,
        bank_path=loss_cfg["tcsr_bank_path"],
        lambda_neg=float(loss_cfg.get("tcsr_lambda_neg", 0.05)),
        lambda_protect=float(loss_cfg.get("tcsr_lambda_protect", 0.10)),
        lambda_agree=float(loss_cfg.get("tcsr_lambda_agree", 0.02)),
        start_epoch=int(loss_cfg.get("tcsr_start_epoch", 40)),
        ramp_epochs=int(loss_cfg.get("tcsr_ramp_epochs", 40)),
        protect_margin=float(loss_cfg.get("tcsr_protect_margin", 0.05)),
    )
else:
    self.cal_loss = base_loss
```

### 13.3 不改 forward

`forward` 继续走已有 MSHNet variant 路径：

```python
if self.model_name in MSHNET_VARIANT_NAMES:
    warm_flag = True if not self.training else epoch > self.mshnet_warm_epoch
    masks, pred = self.model(img, warm_flag)
    if self.training:
        return masks, pred
    return foreground_probability(pred)
```

这点很关键：TCSR 是 training objective，不是 inference structure。

---

## 14. `train.py` 修改点

### 14.1 新增参数

```python
parser.add_argument("--tcsr_bank_path", type=str, default=None)
parser.add_argument("--tcsr_lambda_neg", type=float, default=0.05)
parser.add_argument("--tcsr_lambda_protect", type=float, default=0.10)
parser.add_argument("--tcsr_lambda_agree", type=float, default=0.02)
parser.add_argument("--tcsr_start_epoch", type=int, default=40)
parser.add_argument("--tcsr_ramp_epochs", type=int, default=40)
parser.add_argument("--tcsr_protect_margin", type=float, default=0.05)
```

### 14.2 写入 loss_cfg

```python
loss_cfg.update({
    "tcsr_bank_path": args.tcsr_bank_path,
    "tcsr_lambda_neg": args.tcsr_lambda_neg,
    "tcsr_lambda_protect": args.tcsr_lambda_protect,
    "tcsr_lambda_agree": args.tcsr_lambda_agree,
    "tcsr_start_epoch": args.tcsr_start_epoch,
    "tcsr_ramp_epochs": args.tcsr_ramp_epochs,
    "tcsr_protect_margin": args.tcsr_protect_margin,
})
```

### 14.3 确保传入 image_ids

如果当前 dataloader batch 已经有 image id / path：

```python
loss_out = net.loss(pred, masks, epoch=epoch, image_ids=image_ids)
```

如果 batch 只返回 image path，需要取 stem：

```python
image_ids = [Path(p).stem for p in image_paths]
```

如果 dataset 目前不返回 id，需要最小修改 dataset：

```python
return img, mask, image_name
```

不要把 bank index 写成 batch 顺序。必须按 `image_id` 对齐，否则会造成 silent label corruption。

---

## 15. `tools/official/build_tcsr_sparse_bank.py`

### 15.1 功能

输入：

```text
train image list
GT masks
OHEM-400 train predictions 或 checkpoint
ep250/300/350/400 train predictions 或 checkpoints
```

输出：

```text
docs/internal/tcsr/seed42_nudt/bank_train/*.pt
docs/internal/tcsr/seed42_nudt/gate_tcsr_a_bank_summary.json
```

### 15.2 推荐先用 prediction maps，而不是每次跑 checkpoint

如果已有 train prediction maps：

```text
p_ohem
p_ep250
p_ep300
p_ep350
p_ep400
```

直接构建 bank，避免重复 forward。

### 15.3 bank 构建伪代码

```python
def build_bank_item(gt, p_ohem, p_tce_list):
    p_tce = torch.stack(p_tce_list, dim=0).mean(dim=0)
    s_tce = torch.stack(p_tce_list, dim=0).std(dim=0)

    far_bg = 1.0 - dilate(gt, radius=7)
    local_peak = p_ohem >= maxpool(p_ohem, kernel=7)

    anchor_high = p_ohem >= 0.50
    tce_low = p_tce <= 0.35
    gap = (p_ohem - p_tce) >= 0.15

    neg_seed = far_bg.bool() & local_peak.bool() & anchor_high.bool() & tce_low.bool() & gap.bool()
    neg_weight = dilate(neg_seed.float(), radius=2) * torch.clamp(p_ohem - p_tce, 0.0, 1.0)

    target_support = dilate(gt, radius=2)
    consensus_target = ((p_tce >= 0.60) & (target_support > 0)).float()
    protect_weight = torch.clamp(target_support + consensus_target, 0.0, 1.0)

    ref_prob = torch.maximum(gt.float(), torch.maximum(p_ohem, p_tce))

    # strict safety
    leakage = (neg_weight > 0) & (dilate(gt, radius=2) > 0)
    neg_weight[leakage] = 0.0

    return {
        "neg_weight": neg_weight.cpu(),
        "protect_weight": protect_weight.cpu(),
        "ref_prob": ref_prob.cpu(),
        "meta": {
            "neg_pixels": int((neg_weight > 0).sum()),
            "protect_pixels": int((protect_weight > 0).sum()),
            "target_leakage_pixels": int(leakage.sum()),
        }
    }
```

---

## 16. Gate-TCSR-A：bank audit

新增：

```text
tools/official/check_tcsr_bank_gate_a.py
```

### 16.1 必须检查

```text
target_leakage_pixels_total == 0
num_images == train_images
num_images_with_neg >= min_images_with_neg
neg_pixels_total >= min_neg_pixels_total
protect_pixels_total > 0
neg/protect overlap == 0 after safety cleanup
```

### 16.2 建议阈值

第一版不要调。建议：

```text
min_images_with_neg = 50
min_neg_pixels_total = 500
max_target_leakage_pixels = 0
```

这比“每图都要有负样本”更稳，因为 TCE sparse signal 本来就是稀疏的。

如果 Gate-A 失败，停止 TCSR：

```text
STOP_TCSR_AT_BANK_AUDIT
```

不要降低阈值去救。

---

## 17. Gate-TCSR-B：1 epoch activation sanity

只在 Gate-A 通过后运行。

目标不是看指标，而是看 loss 是否真的在作用。

检查：

```text
tcsr_neg_pixels_mean > 0
tcsr_ramp 在 start_epoch 前为 0，在 start_epoch 后 > 0
selected_neg_logit_mean_after <= selected_neg_logit_mean_before
protected_target_prob_drop <= 0.01
loss_total finite
```

如果 B 失败：

```text
STOP_TCSR_AT_ACTIVATION_SANITY
```

不跑 400 epoch。

---

## 18. Gate-TCSR-C：seed42 official train/eval

只有 Gate-A/B 通过，才训练 seed42。

固定配置：

```text
model_name = MSHNetTCSR
seed = 42
epoch = 400
threshold = 0.5
bank = seed42 train-only TCE sparse bank
lambda_neg = 0.05
lambda_protect = 0.10
lambda_agree = 0.02
start_epoch = 40
ramp_epochs = 40
```

评估：

```text
Full seed42
HC-Val seed42
```

不跑：

```text
seed43/44
HC-Test
blind
external
threshold search
```

通过条件：

```text
Full:
  mIoU >= OHEM - 0.001
  Precision >= OHEM - 0.001
  Pd >= OHEM
  FA ppm <= OHEM + 1.0

HC-Val:
  mIoU >= OHEM + 0.010
  Precision >= OHEM
  Pd >= OHEM
  FA ppm <= OHEM - 20
```

为什么 HC-Val mIoU 要求 +0.010？

因为现有 TWA / ep300 已经有约 +0.029 HC-Val mIoU。如果新结构连 +0.010 都达不到，它就没有替代 post-hoc checkpoint 路线的价值。

---

## 19. Gate-TCSR-D：mechanism comparison

seed42 通过后，必须证明 TCSR 的收益来自 sparse hard-clutter mechanism，而不是普通 early stopping。

比较：

```text
OHEM-400
LateSnapshot-ep300 diagnostic
TWA-4 diagnostic
TCE-4 oracle
TCSR-MSHNet
```

必须输出：

```text
selected TCSR neg pixels 的 student probability 是否下降
protected target pixels 的 probability 是否保持
HC-Val false-alarm component 是否下降
Full split target lost count 是否不增加
```

通过条件：

```text
TCSR Full safe
TCSR HC-Val positive
TCSR 在 selected bank negatives 上有显著 probability drop
TCSR 在 target support 上没有显著 probability drop
TCSR 的收益不是 threshold-only
```

---

## 20. Gate-TCSR-E：seed43/44

只有 Gate-C/D 通过后，才允许：

```text
seed43
seed44
```

注意：这里不是选 seed，而是确认 frozen method。

要求：

```text
Full:
  3/3 no-regression or at least strict mean positive with no Pd drop

HC-Val:
  at least 2/3 positive in mIoU and FA
  mean HC-Val mIoU positive
  mean HC-Val FA lower

Pd:
  no seed has negative Pd delta
```

---

## 21. README 状态更新

把 README 顶部改为：

```markdown
## Current Official Status

Strong anchor: MSHNetOHEM.

Stopped / diagnostic branches:
- TWA with BN recalibration: stopped.
- TWA-4 no-BN: diagnostic only; Gate-E exposed stronger single-checkpoint control.
- LateSnapshot-ep250: stopped at Gate-LS-A because Full split is unsafe.
- LateSnapshot-ep300: diagnostic only; E2-FSC selected it by HC-Val mIoU, but the advantage over TWA-4 is only +0.000204 and is treated as numerical tie / post-hoc checkpoint selection.
- Post-hoc seed / checkpoint / epoch selection: stopped as AAAI main method.

Current active structural route:
- TCSR-MSHNet: Trajectory-Consensus Sparse Reliability MSHNet.
- Inference graph: same single-forward MSHNet-style model.
- Training signal: train-only TCE sparse reliability bank on target-safe hard-clutter local peaks.
- Next allowed gate: Gate-TCSR-A train-only sparse bank audit on seed42 NUDT.

Forbidden before Gate-TCSR-A/B/C:
- seed43 / seed44
- HC-Test
- blind / external
- threshold search
- BN recalibration tuning
- seed search
- checkpoint selection as final method
```

---

## 22. STOPPED_BRANCHES_SUMMARY 更新

新增：

```markdown
## Post-hoc checkpoint / seed selection stopped

Decision: STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD.

Reason:
- Gate-TWA-E2-FSC selected LateSnapshot-ep300, but its HC-Val mIoU advantage over TWA-4 is only +0.000204.
- The selected candidate is not a new architecture, loss, or inference mechanism.
- Choosing a seed / epoch / checkpoint after validation is a selection procedure, not a robust method.

Retained as diagnostics:
- ep250: hard-clutter specialist but Full unsafe.
- ep300: Full-safe late snapshot diagnostic.
- TWA-4: trajectory-averaging diagnostic.
- TCE-4: trajectory-consensus oracle.

Next structural route:
- TCSR-MSHNet, starting with Gate-TCSR-A bank audit.
```

---

## 23. 最小执行清单

### 23.1 现在立刻做

```bash
# 1. 修正 E2-FSC checker，加入 practical margin 和 no-promotion 状态
python -m py_compile tools/official/check_twa_gate_e2_fullsafe_single_control.py
pytest tests/test_twa_gate_e2_fullsafe_single_control.py -q
git diff --check

# 2. 更新 README / STOPPED_BRANCHES_SUMMARY
# 3. 新增 TCSR bank loader / bank builder / bank checker / tests
python -m py_compile utils/tcsr_bank.py tools/official/build_tcsr_sparse_bank.py tools/official/check_tcsr_bank_gate_a.py
pytest tests/test_tcsr_bank.py tests/test_tcsr_gate_a_checker.py -q
git diff --check

# 4. 只跑 train-only bank audit
bash scripts/official/run_tcsr_gate_a_seed42_bank.sh
```

### 23.2 Gate-A 通过后再做

```bash
python -m py_compile loss.py net.py train.py
pytest tests/test_tcsr_loss.py tests/test_tcsr_status_schema.py -q
git diff --check
bash scripts/official/run_tcsr_gate_b_seed42_activation.sh
```

### 23.3 Gate-B 通过后才训练 seed42

```bash
bash scripts/official/run_tcsr_seed42_train_eval.sh
```

---

## 24. 最终决策

当前正式决策应写成：

```text
Decision: STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD

Reason:
  Gate-TWA-E2-FSC selected LateSnapshot-ep300, but the advantage over TWA-4
  on HC-Val mIoU is only +0.000204, which is a numerical tie rather than
  a meaningful method switch.

Do not run:
  Gate-LS-B-ep300-seed43-44-Full-HCVal

Do not use:
  best random seed selection
  best checkpoint selection
  best epoch selection

Retain diagnostics:
  ep250 = Full-unsafe hard-clutter specialist
  ep300 = Full-safe late-snapshot diagnostic
  TWA-4 = weight-averaging diagnostic
  TCE-4 = trajectory-consensus oracle

Next allowed structural route:
  Gate-TCSR-A train-only sparse reliability bank audit.

Structural hypothesis:
  Hard-clutter false alarms are sparse local-peak / component-risk events.
  Dense TCE distillation fails because the useful trajectory signal is sparse.
  TCSR distills only target-safe trajectory-disagreement local peaks while
  protecting target evidence, preserving single-forward inference.
```

一句话：

> 现在不要再让 checker 从 seed / epoch / checkpoint 里挑赢家。真正能解决问题的结构是：把 TCE 暴露出的 hard-clutter reliability signal 做成 **train-only sparse local-peak supervision**，同时保护 MSHNetOHEM 的 target evidence；测试仍然保持单模型单 forward。
