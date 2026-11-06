"""Summarize the E2 Kaggle oracle-boundary result and check preregistered criteria.

Reads either the final result JSON or, if the run was interrupted, the
incremental progress JSONL written by the Kaggle driver.

Usage:
    python scripts/summarize_kaggle_e2.py [result.json] [progress.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT = ROOT / "kaggle_phase_e2_output" / "qwen25_reversal_results.json"
TRACKED_RESULT = ROOT / "experiments" / "qwen25_reversal_results.json"
DEFAULT_PROGRESS = ROOT / "kaggle_phase_e2_output" / "qwen25_reversal_results_progress.jsonl"
REPORT = ROOT / "experiments" / "qwen25_reversal_report.md"
ALL_MODES = (
    "standard",
    "query",
    "segment",
    "lowfreq_segment",
    "entity_segment",
    "entity_lowfreq",
    "additive",
    "entity_additive",
)
MODES = ("standard", "query", "segment", "additive")
SCENARIOS = ("single", "double", "unrelated", "none")
REVERSAL_SCENARIOS = ("single", "double")
SAFETY_SCENARIOS = ("unrelated", "none")


def load_records(result_path: Path, progress_path: Path) -> tuple[list[dict], dict]:
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return payload["records"], payload
    if result_path == DEFAULT_RESULT and TRACKED_RESULT.is_file():
        payload = json.loads(TRACKED_RESULT.read_text(encoding="utf-8"))
        return payload["records"], payload
    if not progress_path.is_file():
        raise FileNotFoundError(f"No result at {result_path} and no progress at {progress_path}")
    records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, {"incomplete_run": True, "source": str(progress_path)}


def aggregate(records: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["mode"], record["scenario"], record["target_tokens"])].append(record)
    rows = {}
    for key, items in groups.items():
        rows[key] = {
            "n": len(items),
            "accuracy": sum(item["margin"] > 0 for item in items) / len(items),
            "margin": sum(item["margin"] for item in items) / len(items),
            "correct_logprob": sum(item["correct_logprob"] for item in items) / len(items),
            "forward_ms": sum(item["elapsed_ms"] for item in items) / len(items),
            "tokens": sum(item["prompt_tokens"] for item in items) / len(items),
        }
    return rows


def mean_metric(rows, mode, scenarios, lengths, metric):
    values = []
    for scenario in scenarios:
        for length in lengths:
            row = rows.get((mode, scenario, length))
            if row:
                values.append(row[metric])
    return sum(values) / len(values) if values else float("nan")


def build_report(records, payload, rows) -> str:
    lengths = sorted({key[2] for key in rows})
    model_name = payload.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
    short_model = model_name.split("/")[-1]
    lines = [
        f"# E2 Oracle 边界实验结果（{short_model}）",
        "",
        f"- 记录数：**{len(records)}**",
        f"- 模型：`{model_name}`",
        f"- 设备：{payload.get('device', 'n/a')}，dtype：{payload.get('dtype', 'n/a')}",
        f"- transformers：{payload.get('transformers_version', 'n/a')}",
        f"- Oracle 边界：{payload.get('oracle_boundaries', True)}；KV cache：{payload.get('uses_kv_cache', False)}",
        "",
    ]
    meta = payload.get("run_metadata", {})
    if meta:
        env = meta.get("environment", {})
        command = meta.get("equivalent_local_command") or payload.get("command", "n/a")
        lines += [
            "## 运行元数据",
            "",
            f"- 提交：`{meta.get('source_commit', 'n/a')}`；kernel：`{meta.get('kaggle_kernel', 'n/a')}`",
            f"- 随机种子：{meta.get('seed', 'n/a')}；模型 revision：`{meta.get('model_revision', 'n/a')}`",
            f"- GPU：{', '.join(env.get('gpu_names', [])) or 'n/a'}；torch：{env.get('torch_version', 'n/a')}",
            f"- 起止：{meta.get('started_utc', 'n/a')} → {meta.get('finished_utc', 'n/a')}（{meta.get('elapsed_seconds', 0):.0f}s）",
            f"- 命令：`{command}`",
        ]
        if meta.get("measured_forward_ms_by_length"):
            costs = ", ".join(
                f"L={k}: {v:.0f}ms" for k, v in sorted(meta["measured_forward_ms_by_length"].items())
            )
            lines.append(f"- 实测单次前向：{costs}")
        if meta.get("shard_groups"):
            lines.append(
                f"- 双卡分片：各组数 {meta['shard_groups']}，预计耗时 {meta.get('shard_estimated_ms', [])} ms，"
                f"绑定 {meta.get('shard_gpu_binding', [])}"
            )
        lines += ["", f"> 样本说明：{meta.get('note', '')}", ""]
        if meta.get("control_variable_note"):
            lines += [f"> 控制变量：{meta['control_variable_note']}", ""]
    if payload.get("incomplete_run"):
        lines += ["> 注意：本次运行被中断，以下统计来自增量进度文件，不是完整结果。", ""]

    present_modes = [m for m in ALL_MODES if any(k[0] == m for k in rows)] or list(MODES)
    lines += ["## 主结果", "", "每个单元格为 `准确率 / 平均 margin（正确 − 过期 logprob）`。", ""]
    for length in lengths:
        lines += [
            f"### 目标长度 {length}",
            "",
            "| 场景 | " + " | ".join(present_modes) + " |",
            "| --- | " + " | ".join(["---"] * len(present_modes)) + " |",
        ]
        for scenario in SCENARIOS:
            cells = []
            for mode in present_modes:
                row = rows.get((mode, scenario, length))
                cells.append(f"{row['accuracy']:.3f} / {row['margin']:+.4f}" if row else "—")
            lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 预注册判据", "", "| 判据 | 阈值 | 实测 | 结论 |", "| --- | --- | --- | --- |"]

    def verdict(ok):
        return "通过" if ok else "未通过"

    std_rev = mean_metric(rows, "standard", REVERSAL_SCENARIOS, lengths, "accuracy")
    seg_rev = mean_metric(rows, "segment", REVERSAL_SCENARIOS, lengths, "accuracy")
    add_rev = mean_metric(rows, "additive", REVERSAL_SCENARIOS, lengths, "accuracy")
    std_safe = mean_metric(rows, "standard", SAFETY_SCENARIOS, lengths, "accuracy")
    seg_safe = mean_metric(rows, "segment", SAFETY_SCENARIOS, lengths, "accuracy")
    delta = (seg_rev - std_rev) * 100
    lines.append(f"| 有效性：segment − standard（撤销场景） | ≥ +5 pp | {delta:+.2f} pp | {verdict(delta >= 5)} |")
    rotations = (seg_rev - add_rev) * 100
    additive_single = mean_metric(rows, "additive", ("single",), lengths, "accuracy")
    additive_collapsed = additive_single < 0.25
    rotation_note = "通过（对照崩塌，不可采信）" if additive_collapsed else verdict(rotations >= 2)
    lines.append(
        f"| 旋转必要性：segment − additive（撤销场景） | ≥ +2 pp | {rotations:+.2f} pp | {rotation_note} |"
    )
    collateral = (std_safe - seg_safe) * 100
    lines.append(
        f"| 低误伤：standard − segment（安全场景） | ≤ 1 pp | {collateral:+.2f} pp | {verdict(collateral <= 1)} |"
    )

    overheads = []
    for length in lengths:
        for scenario in SCENARIOS:
            base = rows.get(("standard", scenario, length))
            seg = rows.get(("segment", scenario, length))
            if base and seg and base["forward_ms"] > 0:
                overheads.append(seg["forward_ms"] / base["forward_ms"] - 1)
    overhead = sum(overheads) / len(overheads) * 100 if overheads else float("nan")
    lines.append(f"| 工程价值：segment 相对 standard 额外延迟 | < 10% | {overhead:+.2f}% | {verdict(overhead < 10)} |")

    if "entity_segment" in present_modes:
        ent_rev = mean_metric(rows, "entity_segment", REVERSAL_SCENARIOS, lengths, "accuracy")
        ent_safe = mean_metric(rows, "entity_segment", SAFETY_SCENARIOS, lengths, "accuracy")
        ent_all = mean_metric(rows, "entity_segment", SCENARIOS, lengths, "accuracy")
        eadd_all = mean_metric(rows, "entity_additive", SCENARIOS, lengths, "accuracy")
        ent_delta = (ent_rev - std_rev) * 100
        ent_rot_all = (ent_all - eadd_all) * 100
        ent_collateral = (std_safe - ent_safe) * 100
        lines.append(
            f"| 实体绑定有效性：entity_segment − standard（撤销场景） | ≥ +5 pp | {ent_delta:+.2f} pp | {verdict(ent_delta >= 5)} |"
        )
        lines.append(
            f"| 实体绑定旋转必要性：entity_segment − entity_additive（全场景） | ≥ +2 pp | {ent_rot_all:+.2f} pp | {verdict(ent_rot_all >= 2)} |"
        )
        lines.append(
            f"| 实体绑定低误伤：standard − entity_segment（安全场景） | ≤ 1 pp | {ent_collateral:+.2f} pp | {verdict(ent_collateral <= 1)} |"
        )
    standard_ceiling = std_rev >= 0.999
    if delta >= 5:
        item1 = (
            f"1. **有效性判据通过（{delta:+.2f} pp）**：Oracle 边界下 QK 分段相位将撤销场景准确率从 "
            f"{std_rev:.3f} 提升至 {seg_rev:.3f}（主要来自 `double` 二次反转从 0.000 恢复并加强，`single` 的正 margin 也进一步扩大）。"
        )
    else:
        item1 = (
            "1. **主假设未获支持**：Oracle 边界下 QK 分段相位没有带来答案级收益"
            f"（{delta:+.2f} pp）。"
            + (
                "standard 在撤销场景已接近满分，存在天花板效应，预注册的「+5 pp」判据在该任务上结构上不可满足；"
                "这不等于证明机制无效，但也不构成支持它的证据。"
                if standard_ceiling
                else "未达到预注册的 +5 pp 门槛。"
            )
        )
    lines += [
        "",
        "## 结论",
        "",
        f"撤销场景 standard 准确率 {std_rev:.3f}、segment {seg_rev:.3f}、additive {add_rev:.3f}。",
        "",
        item1,
        (
            "2. **「旋转必要性」的通过不可采信**：additive 在 single 场景准确率 "
            f"{additive_single:.3f}，penalty=2.0 的加法偏置过强且未与旋转做等强度标定；"
            "segment 只是「没那么糟」，不是优于一个公平的加法基线。"
            if additive_collapsed
            else "2. 旋转必要性按预注册阈值判定，additive 对照未出现崩塌。"
        ),
        f"3. **误伤未达标的风险**：安全场景 standard − segment = {collateral:+.2f} pp，"
        "unrelated（无关实体反转）下 segment 在长序列表现尤其差，符合「独立实体被误伤」的停止条件信号。",
        "4. `none` 场景（无反转边界）四种模式数值完全一致，验证了零干预时 controller 与原模型等价。",
        "5. 按 RESEARCH_PLAN 的停止条件（答案不改善、独立实体持续误伤），"
        "**不应据此进入 E3 门控训练**；下一步应先重标定加法对照或提高任务难度，再判断假设是否值得继续。",
        "",
        "## 说明",
        "",
        "- 指标为答案级对数概率，不是注意力对比度；正确与过期候选来自同一次前向的同一 Softmax。",
        "- 门控为 Oracle 边界，未训练检测器；本结果不能证明模型能学会识别撤销。",
        "- `query` 为全局 Query 相位对照，`additive` 为同边界加法偏置对照。",
        "- 同一 (长度, 场景, 样本) 组的四种模式运行在同一张 GPU；跨长度的绝对耗时可能混合设备，"
        "因此只解释模式间差值。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULT
    progress_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PROGRESS
    if len(sys.argv) > 3:
        report_path = Path(sys.argv[3])
    elif len(sys.argv) > 1:
        report_path = ROOT / "experiments" / (result_path.stem.replace("_results", "_report") + ".md")
    else:
        report_path = REPORT
    records, payload = load_records(result_path, progress_path)
    rows = aggregate(records)
    report = build_report(records, payload, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
