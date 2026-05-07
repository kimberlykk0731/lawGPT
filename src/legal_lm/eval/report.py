from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_lm.utils.io import ensure_dir


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        items = "; ".join(f"{k}: {_format_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    return str(value)


def _render_legacy(summary: dict[str, Any]) -> list[str]:
    overall = summary["overall"]
    lines = [
        "## 总体指标",
        "",
        f"- 样本数：{summary['num_examples']}",
        f"- 综合得分：{overall['overall']:.4f}",
        f"- 结论重合度：{overall['answer_overlap']:.4f}",
        f"- 推理重合度：{overall['reasoning_overlap']:.4f}",
        f"- 引用得分：{overall['citation_score']:.4f}",
        f"- 结构得分：{overall['structure_score']:.4f}",
        "",
        "## 分领域指标",
        "",
    ]
    for domain, metrics in summary.get("by_domain", {}).items():
        lines.extend(
            [
                f"### {domain}",
                "",
                f"- 综合得分：{metrics['overall']:.4f}",
                f"- 结论重合度：{metrics['answer_overlap']:.4f}",
                f"- 推理重合度：{metrics['reasoning_overlap']:.4f}",
                f"- 引用得分：{metrics['citation_score']:.4f}",
                f"- 结构得分：{metrics['structure_score']:.4f}",
                "",
            ]
        )
    return lines


def _render_lawbench(summary: dict[str, Any]) -> list[str]:
    lines = [
        "## 总体指标",
        "",
        f"- 样本数：{summary.get('num_examples', 0)}",
        f"- 综合得分：{summary['overall']:.4f}",
        "",
        "## 按能力维度",
        "",
    ]
    for cap, score in summary.get("by_capability", {}).items():
        lines.append(f"- {cap}: {score:.4f}")
    lines.append("")
    if summary.get("by_task"):
        lines.extend(["## 子任务", "", "| 任务 | 类型 | 能力 | 样本数 | 得分 |", "|---|---|---|---|---|"])
        for task_id, info in summary["by_task"].items():
            lines.append(
                f"| {task_id} | {info.get('type', '-')} | {info.get('capability', '-')} | "
                f"{info.get('n', 0)} | {info.get('score', 0.0):.4f} |"
            )
        lines.append("")
    return lines


def _render_cail(summary: dict[str, Any]) -> list[str]:
    return [
        "## 总体指标",
        "",
        f"- 样本数：{summary.get('num_examples', 0)}",
        f"- 罪名 macro-F1：{summary.get('charge_macro_f1', 0.0):.4f}",
        f"- 法条 macro-F1：{summary.get('article_macro_f1', 0.0):.4f}",
        f"- 量刑 MAE（月）：{summary.get('sentencing_mae', 0.0):.2f}",
        "",
    ]


def render_markdown_report(
    benchmark_result: dict[str, Any],
    cot_ablation: dict[str, Any] | None = None,
) -> str:
    label = benchmark_result.get("label", "benchmark")
    summary = benchmark_result["summary"]
    lines: list[str] = [f"# 评测报告 — {label}", ""]

    if "by_capability" in summary:
        lines.extend(_render_lawbench(summary))
    elif "charge_macro_f1" in summary:
        lines.extend(_render_cail(summary))
    elif isinstance(summary.get("overall"), dict):
        lines.extend(_render_legacy(summary))
    else:
        lines.append("## 总体指标")
        lines.append("")
        for key, value in summary.items():
            lines.append(f"- {key}: {_format_value(value)}")
        lines.append("")

    if cot_ablation:
        lines.extend(["## CoT 消融", ""])
        if cot_ablation.get("best_mode"):
            lines.append(f"- 最优模式：`{cot_ablation['best_mode']}`")
            lines.append("")
        for run in cot_ablation.get("runs", []):
            lines.append(f"### {run.get('label', '-')}")
            lines.append("")
            for key, value in run.items():
                if key == "label":
                    continue
                lines.append(f"- {key}: {_format_value(value)}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_reports(
    output_dir: str | Path,
    benchmark_result: dict[str, Any],
    cot_ablation: dict[str, Any] | None = None,
) -> dict[str, Path]:
    target_dir = ensure_dir(output_dir)
    metrics_path = target_dir / "metrics.json"
    report_path = target_dir / "report.md"

    payload = dict(benchmark_result)
    if cot_ablation is not None:
        payload["cot_ablation"] = cot_ablation

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(benchmark_result, cot_ablation))

    return {"metrics": metrics_path, "report": report_path}
