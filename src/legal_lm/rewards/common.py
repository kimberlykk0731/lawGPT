from __future__ import annotations

import json
from typing import Any, Iterable

from legal_lm.utils.text import completion_to_text, extract_articles, normalize_text, token_overlap_ratio


COMMON_HEADINGS = ["【结论】", "【法律依据】", "【推理摘要】"]
CAUTION_MARKERS = ["基于当前事实", "需补充", "可能", "建议进一步", "存在不确定性", "风险提示"]


def _coerce_rows(value: Any, size: int, default: Any) -> list[Any]:
    if value is None:
        return [default for _ in range(size)]
    if isinstance(value, list):
        if len(value) == size:
            return value
        if len(value) == 1:
            return value * size
    return [value for _ in range(size)]


def _parse_metadata_rows(metadata_json: Any, size: int) -> list[dict[str, Any]]:
    rows = _coerce_rows(metadata_json, size, "{}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            parsed.append(row)
            continue
        try:
            parsed.append(json.loads(str(row)))
        except json.JSONDecodeError:
            parsed.append({})
    return parsed


def _section_score(text: str, headings: list[str]) -> float:
    if not headings:
        return 0.0
    hits = sum(1 for heading in headings if heading in text)
    return hits / len(headings)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {normalize_text(item) for item in left if normalize_text(item)}
    right_set = {normalize_text(item) for item in right if normalize_text(item)}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def citation_accuracy_reward(
    completions: list[Any],
    citations: Any = None,
    answer: Any = None,
    **_: Any,
) -> list[float]:
    answers = _coerce_rows(answer, len(completions), "")
    citation_rows = _coerce_rows(citations, len(completions), [])
    rewards: list[float] = []
    for completion, gold_citations, gold_answer in zip(completions, citation_rows, answers):
        pred_text = completion_to_text(completion)
        pred_articles = extract_articles(pred_text)
        gold_articles = extract_articles(" ".join(gold_citations) if isinstance(gold_citations, list) else str(gold_citations))
        if not gold_articles:
            gold_articles = extract_articles(str(gold_answer))
        rewards.append(_jaccard(pred_articles, gold_articles))
    return rewards


_NLI_PIPELINE = None
_NLI_INIT_FAILED = False


def _get_nli_pipeline():
    """Lazily load a small Chinese NLI model. Returns None if unavailable."""
    global _NLI_PIPELINE, _NLI_INIT_FAILED
    if _NLI_PIPELINE is not None or _NLI_INIT_FAILED:
        return _NLI_PIPELINE
    try:
        from transformers import pipeline

        _NLI_PIPELINE = pipeline(
            "text-classification",
            model="IDEA-CCNL/Erlangshen-Roberta-330M-NLI",
            return_all_scores=True,
        )
    except Exception:  # noqa: BLE001
        _NLI_INIT_FAILED = True
        _NLI_PIPELINE = None
    return _NLI_PIPELINE


def _nli_entailment_score(premise: str, hypothesis: str) -> float | None:
    pipe = _get_nli_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe({"text": premise, "text_pair": hypothesis})
    except Exception:  # noqa: BLE001
        return None
    if isinstance(result, list) and result:
        scores = result[0] if isinstance(result[0], list) else result
        for item in scores:
            label = str(item.get("label", "")).lower()
            if "entail" in label or label in {"0", "label_0"}:
                return float(item.get("score", 0.0))
    return None


def fact_consistency_reward(
    completions: list[Any],
    facts: Any = None,
    **_: Any,
) -> list[float]:
    """事实一致性奖励：优先 NLI 蕴含分；不可用时回退到归一化 token overlap。"""
    fact_rows = _coerce_rows(facts, len(completions), "")
    rewards: list[float] = []
    for completion, fact_text in zip(completions, fact_rows):
        completion_text = completion_to_text(completion)
        fact_str = str(fact_text)
        nli_score = _nli_entailment_score(fact_str, completion_text) if fact_str else None
        if nli_score is not None:
            rewards.append(max(0.0, min(nli_score, 1.0)))
            continue
        overlap = token_overlap_ratio(completion_text, fact_str)
        rewards.append(min(overlap * 1.5, 1.0))
    return rewards


def reasoning_structure_reward(
    completions: list[Any],
    issues: Any = None,
    **_: Any,
) -> list[float]:
    issue_rows = _coerce_rows(issues, len(completions), [])
    rewards: list[float] = []
    for completion, sample_issues in zip(completions, issue_rows):
        text = completion_to_text(completion)
        structure = _section_score(text, COMMON_HEADINGS)
        issue_bonus = 0.0
        if isinstance(sample_issues, list) and sample_issues:
            covered = sum(1 for item in sample_issues if normalize_text(item) and normalize_text(item) in text)
            issue_bonus = covered / len(sample_issues)
        rewards.append(min(0.7 * structure + 0.3 * issue_bonus, 1.0))
    return rewards


def caution_compliance_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        has_marker = any(marker in text for marker in CAUTION_MARKERS)
        rewards.append(1.0 if has_marker else 0.0)
    return rewards


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        has_conclusion = "【结论】" in text
        has_law = "【法律依据】" in text or "【规则适用】" in text
        rewards.append(1.0 if has_conclusion and has_law else 0.0)
    return rewards


def build_common_reward_functions() -> list:
    return [
        citation_accuracy_reward,
        fact_consistency_reward,
        reasoning_structure_reward,
        caution_compliance_reward,
        format_reward,
    ]


__all__ = [
    "build_common_reward_functions",
    "citation_accuracy_reward",
    "fact_consistency_reward",
    "format_reward",
    "reasoning_structure_reward",
    "caution_compliance_reward",
    "_coerce_rows",
    "_parse_metadata_rows",
]
