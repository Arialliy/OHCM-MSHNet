#!/usr/bin/env python3
from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT = Path("/home/ly/AAAI/OHCM-MSHNet/OHCM_MSHNet_research_summary_20260617.docx")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def run(text: object, bold: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    props.append('<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Microsoft YaHei"/>')
    return (
        "<w:r>"
        f"<w:rPr>{''.join(props)}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t>"
        "</w:r>"
    )


def para(text: object = "", bold: bool = False, size: int | None = None, spacing: int = 120) -> str:
    return (
        "<w:p>"
        f'<w:pPr><w:spacing w:after="{spacing}"/></w:pPr>'
        f"{run(text, bold=bold, size=size)}"
        "</w:p>"
    )


def heading(text: object, level: int = 1) -> str:
    size = {1: 32, 2: 26, 3: 23}.get(level, 22)
    before = {1: 240, 2: 180, 3: 120}.get(level, 120)
    return (
        "<w:p>"
        f'<w:pPr><w:spacing w:before="{before}" w:after="100"/></w:pPr>'
        f"{run(text, bold=True, size=size)}"
        "</w:p>"
    )


def bullet(text: object) -> str:
    return para(f"- {text}", spacing=60)


def table(rows: list[list[object]]) -> str:
    if not rows:
        return ""
    cols = max(len(row) for row in rows)
    grid = "".join('<w:gridCol w:w="2200"/>' for _ in range(cols))
    body = [
        "<w:tbl>",
        '<w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '</w:tblBorders></w:tblPr>',
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for r_idx, row in enumerate(rows):
        body.append("<w:tr>")
        for value in row + [""] * (cols - len(row)):
            body.append(
                "<w:tc>"
                '<w:tcPr><w:tcW w:w="2200" w:type="dxa"/></w:tcPr>'
                f"{para(value, bold=(r_idx == 0), spacing=0)}"
                "</w:tc>"
            )
        body.append("</w:tr>")
    body.append("</w:tbl>")
    return "".join(body) + para("")


def document_xml() -> str:
    parts: list[str] = []
    parts.append(heading("OHCM-MSHNet 当前研究总结", 1))
    parts.append(para("日期：2026-06-17"))
    parts.append(para("项目目录：/home/ly/AAAI/OHCM-MSHNet"))
    parts.append(para("研究方向：红外小目标检测；核心问题是复杂背景中高响应伪目标导致的 false alarm。"))

    parts.append(heading("0. 项目准备与代码基础", 2))
    parts.append(bullet("项目由 /home/ly/AAAI/BasicIRSTD-main1 复制到 /home/ly/AAAI/OHCM-MSHNet，所有后续代码和结果均放在 OHCM-MSHNet 下。"))
    parts.append(bullet("代码基础是 BasicIRSTD + MSHNet。MSHNet 被注册到 BasicIRSTD 的 model/__init__.py 和 net.py，支持 train.py/test.py 训练与测试。"))
    parts.append(bullet("默认输入为单通道红外图像：mshnet_in_channels=1；MSHNet warm-up 默认为 5 epoch；训练使用 Adagrad、batch size 4、patch size 256。"))
    parts.append(bullet("数据集位于 /home/ly/AAAI/OHCM-MSHNet/datasets，主要使用 IRSTD-1K、NUAA-SIRST、NUDT-SIRST。"))
    parts.append(bullet("主要运行容器：docker 3ca2917d9c0c，名称 RCCN1，镜像 rccn-5090-pytorch270-cu128。"))
    parts.append(bullet("Smoke test 已完成：MSHNet 在 NUAA-SIRST 上 1 epoch 训练、checkpoint、test、export 均可运行；参数量约 4.065M，FLOPs 约 6.10G。"))

    parts.append(heading("1. 当前正式决策", 2))
    parts.append(para("Decision: STOP_FULL_BRANCH / STOP_PROTO.", bold=True))
    parts.append(bullet("OHCM-full 不保留。虽然 full inference/export/evaluation path 在 checkpoint swap 下可以复现 OHCM-light，但独立训练即使关闭 prototype loss 也无法复现 OHCM-light。"))
    parts.append(bullet("因此，退化不是 prototype loss 单独导致，full/prototype 分支从最终框架中移除。"))
    parts.append(bullet("最终内部方法名：OHCM-light。论文中方法名：OHCM: Online Hard Clutter Mining for Infrared Small Target Detection。"))
    parts.append(bullet("后续不再跑 OHCM-full，不再调 lambda_proto，不做 prototype retry/t-SNE，不把 memory bank/prototype 写进主方法。"))

    parts.append(heading("2. 最终 OHCM 方法定义", 2))
    parts.append(bullet("Online Hard Clutter Mining：从 targetness map 中挖高响应背景伪目标。"))
    parts.append(bullet("保留 connected component mining、target dilation protection、area prior、local contrast、top-K hard clutter selection。"))
    parts.append(bullet("Clutterness Head：Z_c = Head_c(F)。"))
    parts.append(bullet("Clutter-aware Inhibition：Z_final = Z_t - gamma Z_c，P_final = sigmoid(Z_final)。"))
    parts.append(bullet("保留 L_clu、L_sup、L_margin；最终 loss 为 L = L_SLS + lambda_1 L_clu + lambda_2 L_sup + lambda_3 L_margin。"))
    parts.append(bullet("删除 L_proto、memory bank、target prototype、clutter prototype、prototype projection、prototype contrastive loss。"))

    parts.append(heading("3. 从 Step0 开始的完整实验时间线", 2))
    parts.append(heading("3.1 Step0: 原版 MSHNet / BasicIRSTD 复现", 3))
    parts.append(para("目标：先复现原版 MSHNet，在不加入 OHCM 的情况下建立 baseline、导出概率图、生成 threshold curve，并为后续 FP 诊断提供固定来源。"))
    parts.append(table([
        ["Dataset", "Seed", "mIoU", "nIoU", "Pd", "FA ppm", "Precision", "F1", "FP comps"],
        ["IRSTD-1K", "42", "0.5886", "0.6234", "0.8586", "45.74", "0.8055", "0.7410", "36"],
        ["NUAA-SIRST", "42", "0.6814", "0.7757", "0.9358", "70.99", "0.8639", "0.8105", "7"],
        ["NUDT-SIRST", "42", "0.7655", "0.7951", "0.9757", "82.04", "0.8724", "0.8672", "67"],
        ["NUDT-SIRST", "43", "0.8175", "0.8372", "0.9714", "68.69", "0.8954", "0.8996", "42"],
        ["NUDT-SIRST", "44", "0.8369", "0.8616", "0.9831", "62.00", "0.9058", "0.9112", "60"],
    ]))
    parts.append(para("NUDT-SIRST 三种子 baseline mean/std：mIoU 0.8066 ± 0.0369，nIoU 0.8313 ± 0.0337，Pd 0.9767 ± 0.0059，FA 70.91 ± 10.20 ppm，Precision 0.8912 ± 0.0171。"))
    parts.append(bullet("Step0 期间曾因容器暂停/恢复中断，保留 250 epoch checkpoint 后通过 RESUME 机制继续训练。"))
    parts.append(bullet("Step0 输出包括 masks/probs/logits/features/vis、metrics_per_image.csv、threshold_curve.csv、summary_metrics.json。"))
    parts.append(bullet("Step0 结论：原版 MSHNet 复现可运行，Full-set 指标正常，但还需要分析 false alarm 类型和 hard clutter 场景。"))

    parts.append(heading("3.2 Step1: False Alarm / Hard Clutter 诊断", 3))
    parts.append(para("目标：从 Step0 的 probability maps 中提取 false-positive connected components，分析伪目标是否与真实小目标相似。"))
    parts.append(bullet("实现工具：tools/analyze_step1_hard_clutter.py。"))
    parts.append(bullet("记录字段包括 area、bbox、mean/max probability、local contrast、target distance、scale similarity、multi-scale response proxy 和 clutter type。"))
    parts.append(table([
        ["Dataset", "Seed", "Images", "GT comps", "Pred comps", "FP comps", "target-like hard clutter", "Hard clutter fraction"],
        ["IRSTD-1K", "42", "201", "297", "292", "36", "22", "0.6111"],
        ["NUAA-SIRST", "42", "86", "109", "109", "7", "3", "0.4286"],
        ["NUDT-SIRST", "42", "664", "945", "1014", "67", "27", "0.4030"],
        ["NUDT-SIRST", "43", "664", "945", "977", "42", "19", "0.4524"],
        ["NUDT-SIRST", "44", "664", "945", "1000", "60", "28", "0.4667"],
    ]))
    parts.append(table([
        ["Dataset / Seed", "sensor_noise_hot_pixel", "target_like_hard_clutter", "weak_background_fp"],
        ["IRSTD-1K / 42", "3", "22", "11"],
        ["NUAA-SIRST / 42", "3", "3", "1"],
        ["NUDT-SIRST / 42", "23", "27", "17"],
        ["NUDT-SIRST / 43", "11", "19", "12"],
        ["NUDT-SIRST / 44", "10", "28", "22"],
    ]))
    parts.append(para("Step1 结论：baseline 的 false alarms 中存在大量 target-like hard clutter，说明论文主问题“background looks like target”成立。"))

    parts.append(heading("3.3 Step2: HC-Set 构建", 3))
    parts.append(table([
        ["Dataset", "HC images", "HC components", "说明"],
        ["IRSTD-1K", "13", "22", "hard clutter subset"],
        ["NUAA-SIRST", "3", "3", "hard clutter subset"],
        ["NUDT-SIRST", "17", "27", "后续主要门控数据集"],
    ]))
    parts.append(para("NUDT-SIRST baseline HC 指标：mIoU 0.4029，HC-FA 528.67 ppm，Precision 0.5000。"))

    parts.append(heading("3.4 Step3: OHCM-light 初始 gate 与 baseline 对比", 3))
    parts.append(table([
        ["Method", "Split", "mIoU", "FA ppm", "Precision", "Pd"],
        ["MSHNet", "Full", "0.7655", "82.04", "0.8724", "0.9757"],
        ["MSHNet", "HC", "0.4029", "528.67", "0.5000", "0.7647"],
        ["MSHNet + Focal", "HC", "0.4488", "532.26", "0.5260", "0.7647"],
        ["MSHNet + OHEM", "HC", "0.5608", "330.31", "0.6541", "0.8235"],
        ["MSHNet + Top-k Neg", "HC", "0.5847", "318.64", "0.6692", "0.8824"],
        ["OHCM-light", "Full", "0.7942", "81.17", "0.8775", "0.9672"],
        ["OHCM-light", "HC", "0.6087", "211.83", "0.7409", "0.8235"],
    ]))
    parts.append(para("结论：seed42 下 OHCM-light 在 HC-Set 上显著优于普通 hard negative mining，是当时的最优候选。"))

    parts.append(heading("3.5 Step4: full/prototype 分支筛选", 3))
    parts.append(table([
        ["Method", "HC-mIoU", "HC-FA ppm", "HC-Precision", "结论"],
        ["OHCM-light", "0.6087", "211.83", "0.7409", "保留为最终候选"],
        ["OHCM-full, lambda_proto > 0", "0.5274", "331.21", "0.6396", "退化"],
        ["OHCM-full, lambda_proto = 0", "0.5796", "367.11", "0.6450", "仍退化"],
        ["F0-train", "0.4237", "699.21", "0.4733", "不能复现 light 训练轨迹"],
    ]))
    parts.append(bullet("Checkpoint-swap 和 forward parity 说明 full 推理/导出/评估路径能复现 OHCM-light。"))
    parts.append(bullet("独立训练无法复现，说明问题在 full 分支训练轨迹/结构稳定性，而不是 prototype loss 单点。"))
    parts.append(bullet("正式决策：STOP_FULL_BRANCH / STOP_PROTO。"))

    parts.append(heading("3.6 Step5 前置：OHCM-light 三种子复现", 3))
    parts.append(table([
        ["Method", "Seed", "Full mIoU", "Full FA ppm", "Full Precision", "HC-mIoU", "HC-FA ppm", "HC-Precision"],
        ["OHCM", "0", "0.8023", "60.62", "0.9040", "0.3642", "379.67", "0.5274"],
        ["OHCM", "1", "0.7753", "89.58", "0.8650", "0.4189", "561.88", "0.5008"],
        ["OHCM", "2", "0.7326", "107.13", "0.8382", "0.4273", "464.05", "0.5347"],
        ["Mean", "-", "0.7701", "85.78", "0.8691", "0.4035", "468.53", "0.5209"],
    ]))
    parts.append(para("Gate 要求：HC-mIoU mean >= 0.59，HC-FA mean <= 240 ppm，HC-Precision mean >= 0.72。"))
    parts.append(para("结论：Decision = HOLD_OHCM_STABILITY。Full 集表现尚可，但 HC-Set 三种子明显不稳定，不能进入正式 Step5。", bold=True))

    parts.append(heading("3.7 Checkpoint 早停规则诊断", 3))
    parts.append(table([
        ["Seed", "Selected Epoch", "Full mIoU", "Full FA ppm", "Full Precision", "HC-mIoU", "HC-FA ppm", "HC-Precision"],
        ["0", "150", "0.7809", "72.34", "0.8864", "0.4390", "385.96", "0.5709"],
        ["1", "400", "0.7753", "89.58", "0.8650", "0.4189", "561.88", "0.5008"],
        ["2", "400", "0.7326", "107.13", "0.8382", "0.4273", "464.05", "0.5347"],
        ["Mean", "-", "0.7629", "89.68", "0.8632", "0.4284", "470.63", "0.5354"],
    ]))
    parts.append(para("结论：Decision = CHECKPOINT_RULE_NOT_ENOUGH。固定 best checkpoint 选择规则不能解决稳定性问题。"))

    parts.append(heading("4. OHCM-light 稳定性调参", 2))
    parts.append(para("调参只围绕 OHCM-light/OHCM，full/prototype 保持停止。当前失败模式是 HC-FA 高、HC-Precision 低。"))
    parts.append(heading("4.1 tau06-topk2", 3))
    parts.append(table([
        ["Variant", "Seed", "Full mIoU", "Full FA ppm", "Full Precision", "HC-mIoU", "HC-FA ppm", "HC-Precision"],
        ["OHCM-tau06-topk2", "0", "0.8130", "62.90", "0.9022", "0.5357", "333.90", "0.6420"],
        ["OHCM-tau06-topk2", "1", "0.8057", "78.96", "0.8816", "0.5003", "536.75", "0.5517"],
    ]))
    parts.append(para("结论：比原始 seed0/seed1 有提升，但 HC-FA 仍高；seed0/1 已无法满足三种子 FA mean <= 240 ppm，因此不跑 seed2。"))

    parts.append(heading("4.2 tau07-topk1", 3))
    parts.append(table([
        ["Variant", "Seed", "Full mIoU", "Full FA ppm", "Full Precision", "HC-mIoU", "HC-FA ppm", "HC-Precision"],
        ["OHCM-tau07-topk1", "0", "0.7733", "86.45", "0.8683", "0.5922", "345.57", "0.6593"],
        ["OHCM-tau07-topk1", "1", "0.7828", "84.75", "0.8717", "0.5622", "356.34", "0.6427"],
    ]))
    parts.append(para("结论：HC-mIoU 进一步提升，但 HC-FA 仍超过 240 ppm，HC-Precision 仍低于 0.72；threshold curve 说明单纯提高推理阈值不能同时满足 mIoU/FA/Precision。"))

    parts.append(heading("4.3 tau07-topk1-gamma04", 3))
    parts.append(table([
        ["Variant", "Seed", "状态", "当前进度", "说明"],
        ["OHCM-tau07-topk1-gamma04", "0", "running", "约 epoch 5/400", "tau=0.7, topK=1, gamma_max=0.4"],
        ["OHCM-tau07-topk1-gamma04", "1", "running", "约 epoch 5/400", "tau=0.7, topK=1, gamma_max=0.4"],
    ]))
    parts.append(para("目的：保留 tau07-topk1 的 mining 设置，同时增强 clutter-aware inhibition，尝试进一步压低 HC-FA。"))

    parts.append(heading("5. 当前总体判断", 2))
    parts.append(bullet("OHCM-light/OHCM 的单次 seed42 结果很好，但三种子复现未过门槛，投稿前不能进入正式 Step5。"))
    parts.append(bullet("full/prototype 分支已经被证伪并停止；当前所有工作只围绕 OHCM-light 稳定性。"))
    parts.append(bullet("tau/topK 调参能改善 HC-mIoU 和 Precision，但 HC-FA 仍是主要瓶颈。"))
    parts.append(bullet("当前正在验证更强 inhibition 的 gamma04 版本。"))

    parts.append(heading("6. 后续执行顺序", 2))
    parts.append(bullet("等待 OHCM-tau07-topk1-gamma04 的 seed0/seed1 完成。"))
    parts.append(bullet("若 seed0/seed1 仍无法满足 HC-FA 趋势，则不跑 seed2，继续只调 OHCM-light。"))
    parts.append(bullet("可优先尝试：gamma 继续小幅提高、增大 dilation radius 保护 Pd、调整 lambda_sup/lambda_margin、检查 hard_comp 过少问题。"))
    parts.append(bullet("只有当稳定性 gate 通过后，才进入正式 Step5：主表、HC-Set、FA 类型分解、跨域实验、消融实验。"))

    parts.append(heading("7. 关键证据文件", 2))
    for path in [
        "/home/ly/AAAI/OHCM-MSHNet/results/STEP4_FINAL_DECISION_STOP_FULL_PROTO.md",
        "/home/ly/AAAI/OHCM-MSHNet/results/CURRENT_OHCM_DECISION_AND_STEP5_PRE_STATUS.md",
        "/home/ly/AAAI/OHCM-MSHNet/results/step3_ohcm_light_gate/20260613_step3_gate/step3_gate_table.csv",
        "/home/ly/AAAI/OHCM-MSHNet/results/step5_pre_ohcm_seed_repro/20260615_ohcm_three_seed/OHCM_THREE_SEED_REPORT.md",
        "/home/ly/AAAI/OHCM-MSHNet/results/step5_pre_ohcm_stability/20260616_checkpoint_sweep_hc_first/OHCM_CHECKPOINT_SWEEP_REPORT.md",
        "/home/ly/AAAI/OHCM-MSHNet/results/step5_pre_ohcm_stability/20260616_tune_round1",
        "/home/ly/AAAI/OHCM-MSHNet/results/step5_pre_ohcm_stability/20260617_tune_round2",
    ]:
        parts.append(bullet(path))

    body = "".join(parts)
    sect = (
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="900" w:bottom="1440" w:left="900" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{body}{sect}</w:body>"
        "</w:document>"
    )


def write_docx(path: Path) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/document.xml", document_xml())


if __name__ == "__main__":
    write_docx(OUT)
    print(OUT)
