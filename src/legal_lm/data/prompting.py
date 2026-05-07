from __future__ import annotations

from typing import Any

from legal_lm.data.schema import LegalSample, SFTMode


DEFAULT_SYSTEM_PROMPT = (
    "你是一名审慎、准确、遵循中国法律文本的法律分析助手。"
    "你的回答必须以案件事实为基础，优先给出法律结论、法律依据、推理过程和不确定性提示。"
)


def _render_issue_block(sample: LegalSample) -> str:
    if not sample.issues:
        return "未显式提供争点，请先识别争点。"
    return "\n".join(f"- {item}" for item in sample.issues)


def _render_statute_block(sample: LegalSample) -> str:
    if sample.statutes:
        return "\n".join(f"- {item}" for item in sample.statutes)
    if sample.citations:
        return "\n".join(f"- {item}" for item in sample.citations)
    return "未提供明确法条，请结合案情说明需要检索的法律依据。"


def render_user_prompt(sample: LegalSample, rag_block: str = "") -> str:
    instruction = sample.instruction or "请对以下中文法律案件进行分析。"
    parts = [
        instruction,
        f"【案件事实】\n{sample.facts or '未提供'}",
        f"【争点】\n{_render_issue_block(sample)}",
        f"【候选法律依据】\n{_render_statute_block(sample)}",
    ]
    if rag_block:
        parts.append(rag_block)
    return "\n\n".join(parts)


def build_rag_query(sample: LegalSample) -> str:
    """构造 RAG 查询：争点优先 + 事实摘要 + 候选法条名。"""
    pieces: list[str] = []
    if sample.issues:
        pieces.append("；".join(map(str, sample.issues)))
    if sample.facts:
        pieces.append(sample.facts[:500])
    if sample.statutes:
        pieces.append("；".join(sample.statutes[:5]))
    return " ".join(piece for piece in pieces if piece).strip()


def render_assistant_target(sample: LegalSample, mode: str) -> str:
    citations = "\n".join(f"- {item}" for item in sample.citations) or "未明确给出"
    caution = sample.metadata.get("caution") or "以上结论仅基于当前提供事实，若关键事实变化，结论可能变化。"

    if mode == SFTMode.ANSWER_ONLY.value:
        return f"【结论】\n{sample.gold_answer}\n\n【法律依据】\n{citations}"

    if mode == SFTMode.DISTILLED_COT.value:
        reasoning = sample.distilled_cot or sample.gold_reasoning or sample.brief_reasoning or sample.gold_answer
        rules = "\n".join(f"- {item}" for item in sample.statutes or sample.citations) or "请补充相关法条。"
        issues = "\n".join(f"- {item}" for item in sample.issues) or "- 需要先识别争点"
        return (
            f"【案件事实】\n{sample.facts}\n\n"
            f"【争点识别】\n{issues}\n\n"
            f"【规则适用】\n{rules}\n\n"
            f"【逐步推理】\n{reasoning}\n\n"
            f"【结论】\n{sample.gold_answer}\n\n"
            f"【风险提示】\n{caution}"
        )

    reasoning = sample.brief_reasoning or sample.gold_reasoning or sample.gold_answer
    return (
        f"【结论】\n{sample.gold_answer}\n\n"
        f"【法律依据】\n{citations}\n\n"
        f"【推理摘要】\n{reasoning}\n\n"
        f"【风险提示】\n{caution}"
    )


def build_messages(
    sample: LegalSample,
    mode: str,
    system_prompt: str | None = None,
    rag_block: str = "",
) -> list[dict[str, Any]]:
    prompt = render_user_prompt(sample, rag_block=rag_block)
    target = render_assistant_target(sample, mode)
    return [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
