# OHCM-MSHNet：TCSR-v1 冻结版结构方案与分步代码修改计划

> 目标：把当前路线从 **seed / epoch / checkpoint selection** 收敛到一个真正可写成 AAAI 方法的结构性方案。  
> 原则：**不一次性大改；每一节通过后，只重点修改下一节。代码问题修代码，机制失败就停止。**

---

## 0. 当前正式判断

```text
Decision:
  FREEZE_TCSR_V1_AS_FINAL_STRUCTURAL_ATTEMPT

Meaning:
  TCSR-v1 是下一步唯一结构性主线。
  但 TCSR-v1 还不是最终 AAAI 方法，必须通过 gate 才能进入论文主结果。

Stop:
  STOP_POSTHOC_SEED_CHECKPOINT_SELECTION_AS_MAIN_METHOD
  STOP_LATE_SNAPSHOT_EP300_PROMOTION
  STOP_TWA_E2_FSC_AS_FINAL_METHOD

Do not run now:
  Gate-LS-B-ep300-seed43-44-Full-HCVal
  seed43 / seed44
  HC-Test
  blind / external
  threshold search
  BN tuning
  new checkpoint search
  best random seed selection
```

当前 `Gate-TWA-E2-FSC` 虽然 PASS 并选中 `LateSnapshot-ep300`，但这个选择不应继续推进到三种子。核心原因是：

```text
TWA-4 HC-Val delta mIoU = +0.029101
ep300 HC-Val delta mIoU = +0.029305

Difference = +0.000204
```

`+0.000204` 是 numerical tie，不是方法切换证据。把 `ep300` 推到 seed43/44，本质上会变成 **post-hoc epoch promotion**，而不是结构性方法。

---

## 1. 先定边界：可以定什么，不能定什么

### 1.1 可以定

```text
TCSR-v1 作为最后一个结构性主线。
Gate-TCSR-A/B/C/D/E 作为唯一推进路径。
固定 seed42 为 development seed。
固定 seed42/43/44 为后续 confirmation seeds，但现在不跑 seed43/44。
固定 threshold = 0.5。
固定 TCSR-v1 第一版 loss 配置。
```

### 1.2 不能定

```text
不能定 TCSR 已经成功。
不能定 ep300 是主方法。
不能定哪个 seed 好就用哪个 seed。
不能继续按 HC-Val 赢家切换 checkpoint。
不能在 Gate-A/B/C 失败后调阈值、调 lambda、换 seed 救火。
```

### 1.3 代码问题与机制失败的边界

允许修代码：

```text
image_id 对齐错误
path / filename / extension 错误
summary_metrics key 不一致
tensor shape / dtype / device 错误
bank item 缺失导致 KeyError
py_compile / pytest / git diff --check 问题
JSON schema 字段缺失
train.py 没有把 image_ids 传给 loss
```

不允许把机制失败伪装成代码问题：

```text
Gate-A bank 太稀疏后降低阈值救火
Gate-B activation 不明显后调 lambda 救火
Gate-C seed42 指标失败后换 epoch / lr / seed
seed42 不好就改用 seed43 当 development seed
ep300 / TWA-4 / TCSR 之间继续按 HC-Val 赢家切换
```

---

## 2. 当前方案到底解决什么问题？

### 2.1 原目标没有变

```text
保持 MSHNetOHEM 的真实小目标 detection evidence。
降低 hard-clutter false alarms。
提升 Precision。
不牺牲 Pd。
保持单模型、单 forward。
```

### 2.2 TWA / ep300 解决不了结构问题

`TWA-4` 和 `ep300` 的共同问题：

```text
它们都没有显式学习 hard-clutter false alarm 的结构。
它们只是在训练轨迹上找一个更好的点或平均点。
```

`ep300` 的论文风险更高：

```text
same architecture
same loss
same training
only stop epoch changed from 400 to 300
```

这最多能写成 diagnostic observation：

```text
hard-clutter 性能沿训练轨迹波动。
某些 late checkpoint 更偏向 hard-clutter split。
```

但它不能成为 AAAI 主方法。

### 2.3 TCSR 要解决的真实结构问题

```text
hard-clutter false alarms 不是普通 dense pixel-level BCE 问题，
而是 far-background target-like local peaks / connected components 问题。
```

TCSR 的结构假设：

```text
TCE trajectory consensus 能区分：
  稳定 target evidence
  vs
  不稳定 clutter activation

Dense TCE distillation 失败，是因为有用差异非常稀疏。
所以应该只蒸馏 target-safe hard-clutter local peaks。
```

---

## 3. TCSR-v1 定义

### 3.1 方法名

```text
TCSR-MSHNet
Trajectory-Consensus Sparse Reliability MSHNet
```

论文标题候选：

```text
Trajectory-Consensus Sparse Reliability Distillation for Infrared Small Target Detection
```

### 3.2 一句话方法

> 用 train-only TCE trajectory consensus 构建 target-safe sparse reliability bank，只在 far-background hard-clutter local peaks 上施加辅助抑制，同时用 GT / consensus target support 保护小目标 evidence；测试时保持单个 MSHNetOHEM-style forward。

### 3.3 Inference 不变

测试阶段：

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

### 3.4 Training 增加 sparse auxiliary loss

第一版只保留最小必要项：

```text
L_total = L_MSHNetOHEM
        + lambda_neg(t)     * L_sparse_clutter_neg
        + lambda_protect(t) * L_target_protect
```

第一版关闭：

```text
lambda_agree = 0.00
```

原因：先只验证一个核心假设，避免把方案重新拖回 dense TCE matching。

---

## 4. TCSR-v1 冻结配置

建议写入：

```text
docs/internal/tcsr/seed42_nudt/tcsr_v1_frozen_plan.json
```

内容：

```json
{
  "method_name": "TCSR-v1",
  "base": "MSHNetOHEM",
  "inference": "unchanged single-forward MSHNet-style inference",
  "teacher_source": "train-only TCE trajectory predictions",
  "bank_split": "train_only",
  "development_seed": 42,
  "threshold": 0.5,
  "forbidden_data_for_design": [
    "HC-Test",
    "blind",
    "external",
    "seed43/44 result selection"
  ],
  "bank_definition": {
    "far_bg_radius": 7,
    "local_peak_kernel": 7,
    "anchor_high_threshold": 0.50,
    "tce_low_threshold": 0.35,
    "disagreement_threshold": 0.15,
    "neg_dilation_radius": 2,
    "target_support_radius": 2,
    "consensus_target_threshold": 0.60,
    "protect_margin": 0.05
  },
  "loss": {
    "lambda_neg": 0.05,
    "lambda_protect": 0.10,
    "lambda_agree": 0.00,
    "start_epoch": 40,
    "ramp_epochs": 40
  },
  "next_allowed_gate": "Gate-TCSR-A-train-only-bank-audit",
  "forbidden_before_gate_tcsr_c_pass": [
    "seed43",
    "seed44",
    "HC-Test",
    "blind",
    "external",
    "threshold search",
    "BN tuning",
    "checkpoint selection",
    "seed selection"
  ]
}
```

---

## 5. 分阶段执行总览

```text
Stage 0: 修正 E2-FSC checker，停止 ep300 promotion。
Stage 1: 只做 Gate-TCSR-A bank audit，不改 train/loss/net。
Stage 2: Gate-A 通过后，才写 TCSR loss 和 activation sanity。
Stage 3: Gate-B 通过后，才训练 seed42。
Stage 4: seed42 Full + HC-Val 通过后，才做 mechanism comparison。
Stage 5: C/D 都通过后，才允许 seed43/44 confirmation。
```

严格顺序：

```text
E2-FSC no-promotion patch
  -> Gate-TCSR-A bank audit
    -> Gate-TCSR-B activation sanity
      -> Gate-TCSR-C seed42 Full + HC-Val
        -> Gate-TCSR-D mechanism comparison
          -> Gate-TCSR-E seed43/44 confirmation
```

---

# Stage 0：修正 E2-FSC checker，停止 ep300 promotion

## 0.1 修改文件

```text
tools/official/check_twa_gate_e2_fullsafe_single_control.py
tests/test_twa_gate_e2_fullsafe_single_control.py
README.md
STOPPED_BRANCHES_SUMMARY.md
```

## 0.2 目的

不把 `+0.000204` 的 HC-Val mIoU 差异当作方法切换证据。

## 0.3 新增参数

在 checker argparse 中加入：

```python
parser.add_argument(
    "--min_hc_miou_switch_margin",
    type=float,
    default=0.005,
    help=(
        "Minimum HC-Val mIoU advantage required to promote a single late "
        "snapshot over TWA-4. Smaller differences are treated as numerical ties."
    ),
)

parser.add_argument(
    "--disable_posthoc_single_promotion",
    action="store_true",
    help=(
        "If set, do not promote a single late snapshot to a multi-seed gate; "
        "retain it as diagnostic only."
    ),
)
```

## 0.4 替换选择逻辑

把原来的：

```python
best = max(eligible, key=lambda x: x["hcval_delta"]["mIoU"])
selected = best["name"]
```

替换为：

```python
best = max(eligible, key=lambda x: x["hcval_delta"]["mIoU"])
twa4 = next((x for x in eligible if x["name"] == "TWA-4-noBN"), None)

if twa4 is None:
    raise SystemExit("TWA-4-noBN must be present as the trajectory-averaging control.")

best_advantage_over_twa4 = (
    float(best["hcval_delta"]["mIoU"])
    - float(twa4["hcval_delta"]["mIoU"])
)

is_single_snapshot = str(best["name"]).startswith("LateSnapshot-")

if is_single_snapshot and best_advantage_over_twa4 < args.min_hc_miou_switch_margin:
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

在 output JSON 里加入：

```python
result.update({
    "best_eligible_by_hcval": best["name"],
    "twa4_hcval_delta_miou": float(twa4["hcval_delta"]["mIoU"]),
    "best_hcval_delta_miou": float(best["hcval_delta"]["mIoU"]),
    "best_advantage_over_twa4": best_advantage_over_twa4,
    "min_hc_miou_switch_margin": args.min_hc_miou_switch_margin,
    "selection_status": selection_status,
    "promotion_allowed": promotion_allowed,
    "next_allowed_gate": next_allowed_gate,
})
```

## 0.5 当前数据下期望输出

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

## 0.6 Stage 0 验证

```bash
python -m py_compile tools/official/check_twa_gate_e2_fullsafe_single_control.py
pytest tests/test_twa_gate_e2_fullsafe_single_control.py -q
git diff --check
```

通过后才进入 Stage 1。

---

# Stage 1：Gate-TCSR-A train-only sparse bank audit

## 1.1 这一阶段只新增这些文件

```text
utils/tcsr_bank.py
tools/official/build_tcsr_sparse_bank.py
tools/official/check_tcsr_bank_gate_a.py
scripts/official/run_tcsr_gate_a_seed42_bank.sh
tests/test_tcsr_bank.py
tests/test_tcsr_gate_a_checker.py
docs/internal/tcsr/seed42_nudt/tcsr_v1_frozen_plan.json
```

不要改：

```text
loss.py
net.py
train.py
```

## 1.2 Gate-TCSR-A 只回答一个问题

```text
train split 上是否存在足够多 target-safe sparse hard-clutter reliability signals？
```

如果没有，TCSR 的训练信号不成立，必须停止。

## 1.3 bank item 定义

每张 train 图保存：

```text
bank_root/<image_id>.pt
```

其中包含：

```python
{
    "neg_weight": Tensor[1,H,W] or Tensor[H,W],
    "protect_weight": Tensor[1,H,W] or Tensor[H,W],
    "ref_prob": Tensor[1,H,W] or Tensor[H,W],
    "meta": {
        "image_id": str,
        "neg_pixels": int,
        "protect_pixels": int,
        "target_leakage_pixels": int,
        "neg_protect_overlap_pixels": int
    }
}
```

## 1.4 `utils/tcsr_bank.py`

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
    """Load train-only trajectory-consensus sparse reliability maps."""

    def __init__(self, bank_root: str | Path):
        self.bank_root = Path(bank_root)
        if not self.bank_root.exists():
            raise FileNotFoundError(f"TCSR bank not found: {self.bank_root}")

    @staticmethod
    def normalize_image_id(image_id) -> str:
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
        key = self.normalize_image_id(image_id)
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
            neg_weight = F.interpolate(neg_weight[None], size=size, mode="nearest")[0]
            protect_weight = F.interpolate(protect_weight[None], size=size, mode="nearest")[0]
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

## 1.5 `tools/official/build_tcsr_sparse_bank.py` 关键逻辑

第一版建议使用已经导出的 train prediction maps，避免重复 forward。输入应是：

```text
GT masks
OHEM-400 train probability maps
TCE trajectory maps: ep250 / ep300 / ep350 / ep400 train probability maps
```

核心函数：

```python
import torch
import torch.nn.functional as F


def maxpool2d(x: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    return F.max_pool2d(x[None, None], kernel_size=kernel, stride=1, padding=pad)[0, 0]


def dilate_binary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return (mask > 0).float()
    kernel = 2 * radius + 1
    pad = radius
    out = F.max_pool2d(mask.float()[None, None], kernel_size=kernel, stride=1, padding=pad)[0, 0]
    return (out > 0).float()


def build_bank_item(
    gt: torch.Tensor,
    p_ohem: torch.Tensor,
    p_tce_list: list[torch.Tensor],
    far_bg_radius: int = 7,
    local_peak_kernel: int = 7,
    anchor_high_threshold: float = 0.50,
    tce_low_threshold: float = 0.35,
    disagreement_threshold: float = 0.15,
    neg_dilation_radius: int = 2,
    target_support_radius: int = 2,
    consensus_target_threshold: float = 0.60,
):
    gt = (gt > 0.5).float()
    p_ohem = p_ohem.float().clamp(0.0, 1.0)
    tce_stack = torch.stack([x.float().clamp(0.0, 1.0) for x in p_tce_list], dim=0)
    p_tce = tce_stack.mean(dim=0)

    far_bg = 1.0 - dilate_binary(gt, radius=far_bg_radius)
    local_peak = p_ohem >= maxpool2d(p_ohem, kernel=local_peak_kernel)

    anchor_high = p_ohem >= anchor_high_threshold
    tce_low = p_tce <= tce_low_threshold
    disagreement = (p_ohem - p_tce) >= disagreement_threshold

    neg_seed = far_bg.bool() & local_peak.bool() & anchor_high.bool() & tce_low.bool() & disagreement.bool()
    neg_weight = dilate_binary(neg_seed.float(), radius=neg_dilation_radius)
    neg_weight = neg_weight * torch.clamp(p_ohem - p_tce, min=0.0, max=1.0)

    target_support = dilate_binary(gt, radius=target_support_radius)
    consensus_target = ((p_tce >= consensus_target_threshold) & (target_support > 0)).float()
    protect_weight = torch.clamp(target_support + consensus_target, 0.0, 1.0)

    ref_prob = torch.maximum(gt, torch.maximum(p_ohem, p_tce))

    # strict cleanup: no negative supervision near protected target support
    leakage_mask = (neg_weight > 0) & (target_support > 0)
    target_leakage_pixels = int(leakage_mask.sum().item())
    neg_weight = neg_weight.masked_fill(leakage_mask, 0.0)

    overlap_mask = (neg_weight > 0) & (protect_weight > 0)
    neg_protect_overlap_pixels = int(overlap_mask.sum().item())
    neg_weight = neg_weight.masked_fill(overlap_mask, 0.0)

    item = {
        "neg_weight": neg_weight.cpu(),
        "protect_weight": protect_weight.cpu(),
        "ref_prob": ref_prob.cpu(),
        "meta": {
            "neg_pixels": int((neg_weight > 0).sum().item()),
            "protect_pixels": int((protect_weight > 0).sum().item()),
            "target_leakage_pixels": target_leakage_pixels,
            "neg_protect_overlap_pixels": neg_protect_overlap_pixels,
        },
    }
    return item
```

完整脚本可以围绕这个函数做三件事：

```text
1. 遍历 train image_id。
2. 加载 gt / p_ohem / p_ep250 / p_ep300 / p_ep350 / p_ep400。
3. 每张图保存 <image_id>.pt，并汇总 gate_tcsr_a_bank_summary.json。
```

## 1.6 `tools/official/check_tcsr_bank_gate_a.py`

```python
import argparse
import json
from pathlib import Path

import torch


def load_item(path: Path):
    return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_num_images", type=int, required=True)
    parser.add_argument("--min_images_with_neg", type=int, default=50)
    parser.add_argument("--min_neg_pixels_total", type=int, default=500)
    parser.add_argument("--max_target_leakage_pixels", type=int, default=0)
    args = parser.parse_args()

    bank_root = Path(args.bank_root)
    paths = sorted(bank_root.glob("*.pt"))

    num_images = len(paths)
    images_with_neg = 0
    neg_pixels_total = 0
    protect_pixels_total = 0
    target_leakage_pixels_total = 0
    neg_protect_overlap_pixels_total = 0
    bad_items = []

    for path in paths:
        data = load_item(path)
        neg = data["neg_weight"]
        protect = data["protect_weight"]
        meta = data.get("meta", {})

        neg_pixels = int((neg > 0).sum().item())
        protect_pixels = int((protect > 0).sum().item())
        overlap_pixels = int(((neg > 0) & (protect > 0)).sum().item())
        leakage_pixels = int(meta.get("target_leakage_pixels", 0))

        if neg_pixels > 0:
            images_with_neg += 1
        neg_pixels_total += neg_pixels
        protect_pixels_total += protect_pixels
        target_leakage_pixels_total += leakage_pixels
        neg_protect_overlap_pixels_total += overlap_pixels

        if leakage_pixels > 0 or overlap_pixels > 0:
            bad_items.append(path.name)

    checks = {
        "num_images_matches": num_images == args.expected_num_images,
        "enough_images_with_neg": images_with_neg >= args.min_images_with_neg,
        "enough_neg_pixels_total": neg_pixels_total >= args.min_neg_pixels_total,
        "has_protect_pixels": protect_pixels_total > 0,
        "no_target_leakage": target_leakage_pixels_total <= args.max_target_leakage_pixels,
        "no_neg_protect_overlap": neg_protect_overlap_pixels_total == 0,
    }

    gate_pass = all(checks.values())
    result = {
        "gate": "Gate-TCSR-A",
        "method": "TCSR-v1",
        "split": "train-only",
        "bank_root": str(bank_root),
        "num_images": num_images,
        "expected_num_images": args.expected_num_images,
        "images_with_neg": images_with_neg,
        "neg_pixels_total": neg_pixels_total,
        "protect_pixels_total": protect_pixels_total,
        "target_leakage_pixels_total": target_leakage_pixels_total,
        "neg_protect_overlap_pixels_total": neg_protect_overlap_pixels_total,
        "bad_items_sample": bad_items[:20],
        "thresholds": {
            "min_images_with_neg": args.min_images_with_neg,
            "min_neg_pixels_total": args.min_neg_pixels_total,
            "max_target_leakage_pixels": args.max_target_leakage_pixels,
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "next_allowed_gate": "Gate-TCSR-B-activation-sanity" if gate_pass else "STOP_TCSR_AT_BANK_AUDIT",
        "forbidden_if_fail": [
            "threshold tuning",
            "lambda tuning",
            "seed43/44",
            "HC-Test",
            "blind/external",
            "full training"
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not gate_pass:
        raise SystemExit("Gate-TCSR-A failed. Stop TCSR-v1 at bank audit.")


if __name__ == "__main__":
    main()
```

## 1.7 Gate-TCSR-A 阈值

第一版固定：

```text
expected_num_images = train_images
min_images_with_neg = 50
min_neg_pixels_total = 500
max_target_leakage_pixels = 0
protect_pixels_total > 0
neg_protect_overlap_pixels_total = 0
```

如果失败：

```text
STOP_TCSR_AT_BANK_AUDIT
```

不要降低阈值救火。

## 1.8 `scripts/official/run_tcsr_gate_a_seed42_bank.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/OHCM-MSHNet-main}

BANK_ROOT="${ROOT}/docs/internal/tcsr/seed42_nudt/bank_train"
SUMMARY="${ROOT}/docs/internal/tcsr/seed42_nudt/gate_tcsr_a_bank_summary.json"

# 这里的路径按你本地实际 prediction map 目录替换。
# 第一版只允许 train split，不允许 HC-Val/HC-Test/blind/external 参与 bank 构建。
python tools/official/build_tcsr_sparse_bank.py \
  --split train \
  --output_bank_root "${BANK_ROOT}" \
  --output_summary "${SUMMARY}.build.json" \
  --far_bg_radius 7 \
  --local_peak_kernel 7 \
  --anchor_high_threshold 0.50 \
  --tce_low_threshold 0.35 \
  --disagreement_threshold 0.15 \
  --neg_dilation_radius 2 \
  --target_support_radius 2 \
  --consensus_target_threshold 0.60

python tools/official/check_tcsr_bank_gate_a.py \
  --bank_root "${BANK_ROOT}" \
  --output "${SUMMARY}" \
  --expected_num_images 697 \
  --min_images_with_neg 50 \
  --min_neg_pixels_total 500 \
  --max_target_leakage_pixels 0
```

如果 train image count 不是 697，以当前本地 NUDT train split 为准，但必须写入 summary。

## 1.9 Stage 1 测试

`tests/test_tcsr_bank.py`：

```python
import torch

from utils.tcsr_bank import TCSRBank


def test_tcsr_bank_loads_and_resizes(tmp_path):
    item = {
        "neg_weight": torch.ones(8, 8),
        "protect_weight": torch.zeros(8, 8),
        "ref_prob": torch.full((8, 8), 0.5),
    }
    torch.save(item, tmp_path / "img001.pt")

    bank = TCSRBank(tmp_path)
    out = bank.get("/any/path/img001.png", size=(16, 16))

    assert out.image_id == "img001"
    assert tuple(out.neg_weight.shape) == (1, 16, 16)
    assert tuple(out.protect_weight.shape) == (1, 16, 16)
    assert tuple(out.ref_prob.shape) == (1, 16, 16)


def test_tcsr_bank_missing_item_raises(tmp_path):
    bank = TCSRBank(tmp_path)
    try:
        bank.get("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for missing bank item")
```

`tests/test_tcsr_gate_a_checker.py` 至少覆盖：

```text
PASS: enough sparse negatives, zero leakage, zero overlap
FAIL: not enough images_with_neg
FAIL: target leakage > 0
FAIL: neg/protect overlap > 0
```

## 1.10 Stage 1 执行

```bash
python -m py_compile \
  utils/tcsr_bank.py \
  tools/official/build_tcsr_sparse_bank.py \
  tools/official/check_tcsr_bank_gate_a.py

pytest tests/test_tcsr_bank.py tests/test_tcsr_gate_a_checker.py -q
git diff --check

bash scripts/official/run_tcsr_gate_a_seed42_bank.sh
```

---

# Stage 2：Gate-TCSR-B activation sanity

只有 Gate-TCSR-A PASS，才进入这一节。

## 2.1 修改文件

```text
loss.py
net.py
train.py
tools/official/check_tcsr_activation_gate_b.py
scripts/official/run_tcsr_gate_b_seed42_activation.sh
tests/test_tcsr_loss.py
tests/test_tcsr_status_schema.py
```

## 2.2 `loss.py` 新增 TCSR loss

第一版 `lambda_agree = 0.00`，但可以保留代码字段，默认关闭。

```python
class TrajectoryConsensusSparseLoss(nn.Module):
    """Train-time TCSR auxiliary loss.

    This loss does not change inference. It only adds sparse reliability
    supervision from a prebuilt train-only TCE consensus bank.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        bank_path: str,
        lambda_neg: float = 0.05,
        lambda_protect: float = 0.10,
        lambda_agree: float = 0.00,
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

        if neg_w.ndim == 3:
            neg_w = neg_w[:, None]
        if protect_w.ndim == 3:
            protect_w = protect_w[:, None]
        if ref_prob is not None and ref_prob.ndim == 3:
            ref_prob = ref_prob[:, None]

        ramp = self._ramp(epoch)
        zero = final_logit.sum() * 0.0

        loss_neg_map = F.binary_cross_entropy_with_logits(
            final_logit,
            torch.zeros_like(final_logit),
            reduction="none",
        )
        loss_neg = self._weighted_mean(loss_neg_map, neg_w, self.eps) if neg_w.sum() > 0 else zero

        if ref_prob is not None:
            p = torch.sigmoid(final_logit)
            protect_violation = F.relu((ref_prob - self.protect_margin) - p).pow(2)
            loss_protect = (
                self._weighted_mean(protect_violation, protect_w, self.eps)
                if protect_w.sum() > 0 else zero
            )
            loss_agree = zero
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

注意：这段需要 `loss.py` 顶部已有或新增：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 2.3 `net.py` 最小接入

只加 model name 和 loss wrapper，不改 forward graph。

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

base loss 后包一层：

```python
if model_name == "MSHNetTCSR":
    self.cal_loss = TrajectoryConsensusSparseLoss(
        base_loss=base_loss,
        bank_path=loss_cfg["tcsr_bank_path"],
        lambda_neg=float(loss_cfg.get("tcsr_lambda_neg", 0.05)),
        lambda_protect=float(loss_cfg.get("tcsr_lambda_protect", 0.10)),
        lambda_agree=float(loss_cfg.get("tcsr_lambda_agree", 0.00)),
        start_epoch=int(loss_cfg.get("tcsr_start_epoch", 40)),
        ramp_epochs=int(loss_cfg.get("tcsr_ramp_epochs", 40)),
        protect_margin=float(loss_cfg.get("tcsr_protect_margin", 0.05)),
    )
else:
    self.cal_loss = base_loss
```

## 2.4 `train.py` 参数

```python
parser.add_argument("--tcsr_bank_path", type=str, default=None)
parser.add_argument("--tcsr_lambda_neg", type=float, default=0.05)
parser.add_argument("--tcsr_lambda_protect", type=float, default=0.10)
parser.add_argument("--tcsr_lambda_agree", type=float, default=0.00)
parser.add_argument("--tcsr_start_epoch", type=int, default=40)
parser.add_argument("--tcsr_ramp_epochs", type=int, default=40)
parser.add_argument("--tcsr_protect_margin", type=float, default=0.05)
```

写入 `loss_cfg`：

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

确保 loss 调用传入 `image_ids`：

```python
loss_out = net.loss(pred, masks, epoch=epoch, image_ids=image_ids)
```

如果 dataloader 当前只返回 image path：

```python
from pathlib import Path
image_ids = [Path(p).stem for p in image_paths]
```

如果 dataset 不返回 image id，则最小修改 dataset：

```python
return img, mask, image_name
```

绝对不要用 batch 顺序对齐 bank。

## 2.5 Gate-TCSR-B 检查内容

只做 1 epoch 或 very short activation sanity，不看 official metrics。

必须检查：

```text
tcsr_neg_pixels_mean > 0
tcsr_ramp 在 start_epoch 前为 0，在 start_epoch 后 > 0
loss_total finite
selected_neg_logit_mean_after <= selected_neg_logit_mean_before
protected_target_prob_drop <= 0.01
```

如果失败：

```text
STOP_TCSR_AT_ACTIVATION_SANITY
```

不要跑 400 epoch。

## 2.6 Stage 2 执行

```bash
python -m py_compile loss.py net.py train.py tools/official/check_tcsr_activation_gate_b.py
pytest tests/test_tcsr_loss.py tests/test_tcsr_status_schema.py -q
git diff --check
bash scripts/official/run_tcsr_gate_b_seed42_activation.sh
```

---

# Stage 3：Gate-TCSR-C seed42 official train/eval

只有 Gate-A/B 都 PASS，才训练 seed42。

## 3.1 固定配置

```text
model_name = MSHNetTCSR
seed = 42
epoch = 400
threshold = 0.5
bank = docs/internal/tcsr/seed42_nudt/bank_train
lambda_neg = 0.05
lambda_protect = 0.10
lambda_agree = 0.00
start_epoch = 40
ramp_epochs = 40
```

## 3.2 只评估

```text
seed42 Full
seed42 HC-Val
```

不评估：

```text
seed43/44
HC-Test
blind
external
threshold search
```

## 3.3 Gate-TCSR-C 通过条件

Full split：

```text
mIoU      >= OHEM - 0.001
Precision >= OHEM - 0.001
Pd        >= OHEM
FA ppm    <= OHEM + 1.0
```

HC-Val：

```text
mIoU      >= OHEM + 0.010
Precision >= OHEM
Pd        >= OHEM
FA ppm    <= OHEM - 20
```

为什么 HC-Val mIoU 只要求 +0.010：

```text
TWA / ep300 目前约 +0.029，TCSR 是新结构，第一关先证明有效而且 Full-safe。
若连 +0.010 都没有，就没有替代 post-hoc checkpoint 路线的价值。
```

## 3.4 如果 Gate-C 失败

```text
STOP_TCSR_AT_SEED42_OFFICIAL_GATE
```

不允许：

```text
调 lambda
换 start_epoch
换 seed
换 checkpoint
跑 seed43/44
看 HC-Test / blind / external
```

---

# Stage 4：Gate-TCSR-D mechanism comparison

只有 Gate-C PASS，才做机制比较。

比较对象：

```text
OHEM-400
LateSnapshot-ep300 diagnostic only
TWA-4 diagnostic only
TCE-4 oracle / teacher source
TCSR-MSHNet
```

必须输出：

```text
selected TCSR neg pixels 的 student probability 是否下降
protected target pixels 的 probability 是否保持
HC-Val false-alarm component 是否下降
Full split target lost count 是否不增加
Pd-matched FA 是否仍下降
mIoU-matched Pd 是否不下降
```

通过条件：

```text
TCSR Full safe
TCSR HC-Val positive
TCSR 在 selected bank negatives 上有 probability drop
TCSR 在 target support 上没有 probability drop
收益不是 threshold-only
```

如果失败：

```text
STOP_TCSR_AT_MECHANISM_GATE
```

---

# Stage 5：Gate-TCSR-E seed43/44 confirmation

只有 Gate-C/D 都 PASS，才允许：

```text
seed43
seed44
```

这里不是选 seed，而是确认 frozen method。

要求：

```text
Full:
  3/3 no-regression，或至少 strict mean positive 且 no Pd drop

HC-Val:
  at least 2/3 positive in mIoU and FA
  mean HC-Val mIoU positive
  mean HC-Val FA lower

Pd:
  no seed has negative Pd delta
```

如果通过，才进入：

```text
threshold-matched
FP component analysis
blind / external final once
```

---

## 6. README 更新块

把 README 顶部更新为：

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

## 7. STOPPED_BRANCHES_SUMMARY 更新块

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

## 8. 15 天内的最小可执行路线

### Day 1：只做 Stage 0 + Stage 1 代码

```bash
python -m py_compile tools/official/check_twa_gate_e2_fullsafe_single_control.py
pytest tests/test_twa_gate_e2_fullsafe_single_control.py -q
git diff --check

python -m py_compile \
  utils/tcsr_bank.py \
  tools/official/build_tcsr_sparse_bank.py \
  tools/official/check_tcsr_bank_gate_a.py
pytest tests/test_tcsr_bank.py tests/test_tcsr_gate_a_checker.py -q
git diff --check
```

### Day 2：只跑 Gate-TCSR-A

```bash
bash scripts/official/run_tcsr_gate_a_seed42_bank.sh
```

结果分支：

```text
PASS -> Stage 2
FAIL -> STOP_TCSR_AT_BANK_AUDIT
```

### Day 3：Gate-B activation sanity

```bash
python -m py_compile loss.py net.py train.py tools/official/check_tcsr_activation_gate_b.py
pytest tests/test_tcsr_loss.py tests/test_tcsr_status_schema.py -q
git diff --check
bash scripts/official/run_tcsr_gate_b_seed42_activation.sh
```

结果分支：

```text
PASS -> Stage 3
FAIL -> STOP_TCSR_AT_ACTIVATION_SANITY
```

### Day 4-6：seed42 official train/eval

```bash
bash scripts/official/run_tcsr_seed42_train_eval.sh
```

结果分支：

```text
PASS -> Stage 4 mechanism comparison
FAIL -> STOP_TCSR_AT_SEED42_OFFICIAL_GATE
```

### Day 7-8：mechanism comparison

```text
selected neg probability drop
protected target support stability
component FP reduction
threshold-matched sanity
```

结果分支：

```text
PASS -> seed43/44 confirmation
FAIL -> STOP_TCSR_AT_MECHANISM_GATE
```

### Day 9-12：seed43/44 confirmation only if frozen method passed

```text
只跑 frozen TCSR-v1。
不换 lambda。
不换 threshold。
不换 checkpoint。
不选 seed。
```

### Day 13-15：整理论文材料

```text
method diagram
ablation table
failure branch table
gate protocol table
mechanism visualization
final result table
```

---

## 9. 最终可写入计划的正式决策

```text
Decision:
  FREEZE_TCSR_V1_AS_FINAL_STRUCTURAL_ATTEMPT

Rationale:
  Gate-TWA-E2-FSC selected LateSnapshot-ep300, but its HC-Val mIoU advantage
  over TWA-4 is only +0.000204. This is a numerical tie, not sufficient
  evidence for promoting a single checkpoint as the main method.

Stopped:
  Post-hoc seed selection
  Post-hoc checkpoint selection
  Post-hoc epoch selection
  LateSnapshot-ep300 promotion

Retained diagnostics:
  ep250 = Full-unsafe hard-clutter specialist
  ep300 = Full-safe late-snapshot diagnostic
  TWA-4 = trajectory-averaging diagnostic
  TCE-4 = trajectory-consensus oracle / teacher source

Next allowed work:
  Stage 0: patch E2-FSC no-promotion logic
  Stage 1: Gate-TCSR-A train-only sparse bank audit

Core structural hypothesis:
  Hard-clutter false alarms are sparse local-peak / component-risk events.
  Dense TCE distillation fails because useful trajectory signal is sparse.
  TCSR distills only target-safe trajectory-disagreement local peaks while
  protecting target evidence, preserving single-forward inference.
```

一句话：

> 现在不要再让 checker 从 seed / epoch / checkpoint 里挑赢家。TCSR-v1 可以定为最后一个结构性主线，但下一步只做 **E2-FSC no-promotion patch + Gate-TCSR-A bank audit**；代码错就修代码，机制 gate 失败就停止。
