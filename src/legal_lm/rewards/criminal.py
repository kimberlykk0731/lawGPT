from __future__ import annotations

from typing import Any

from legal_lm.rewards.common import _coerce_rows, _parse_metadata_rows
from legal_lm.utils.text import completion_to_text, extract_articles, extract_number, normalize_text, token_overlap_ratio


def charge_match_reward(
    completions: list[Any],
    answer: Any = None,
    metadata_json: Any = None,
    **_: Any,
) -> list[float]:
    answers = _coerce_rows(answer, len(completions), "")
    metadata_rows = _parse_metadata_rows(metadata_json, len(completions))
    rewards: list[float] = []
    for completion, gold_answer, metadata in zip(completions, answers, metadata_rows):
        expected = metadata.get("charges") or metadata.get("charge") or []
        expected_text = " ".join(map(str, expected)) if expected else str(gold_answer)
        rewards.append(token_overlap_ratio(completion_to_text(completion), expected_text))
    return rewards


def article_match_reward(
    completions: list[Any],
    citations: Any = None,
    metadata_json: Any = None,
    **_: Any,
) -> list[float]:
    citation_rows = _coerce_rows(citations, len(completions), [])
    metadata_rows = _parse_metadata_rows(metadata_json, len(completions))
    rewards: list[float] = []
    for completion, gold_citations, metadata in zip(completions, citation_rows, metadata_rows):
        pred = set(extract_articles(completion_to_text(completion)))
        gold = set(extract_articles(" ".join(gold_citations) if isinstance(gold_citations, list) else str(gold_citations)))
        if not gold:
            gold = {str(item) for item in metadata.get("articles", [])}
        if not pred and not gold:
            rewards.append(1.0)
        elif not pred or not gold:
            rewards.append(0.0)
        else:
            rewards.append(len(pred & gold) / len(pred | gold))
    return rewards


def sentencing_band_reward(
    completions: list[Any],
    answer: Any = None,
    metadata_json: Any = None,
    **_: Any,
) -> list[float]:
    answers = _coerce_rows(answer, len(completions), "")
    metadata_rows = _parse_metadata_rows(metadata_json, len(completions))
    rewards: list[float] = []
    for completion, gold_answer, metadata in zip(completions, answers, metadata_rows):
        pred_text = completion_to_text(completion)
        gold_text = str(gold_answer)
        if "死刑" in gold_text:
            rewards.append(1.0 if "死刑" in pred_text else 0.0)
            continue
        if "无期徒刑" in gold_text:
            rewards.append(1.0 if "无期徒刑" in pred_text else 0.0)
            continue

        pred_value = extract_number(pred_text)
        gold_value = extract_number(gold_text)
        if gold_value is None:
            imprisonment = metadata.get("imprisonment", {})
            if isinstance(imprisonment, dict) and imprisonment.get("imprisonment") is not None:
                gold_value = float(imprisonment["imprisonment"])

        if pred_value is None or gold_value is None:
            rewards.append(0.0)
            continue

        relative_error = abs(pred_value - gold_value) / max(abs(gold_value), 1.0)
        rewards.append(max(0.0, 1.0 - relative_error))
    return rewards


def element_coverage_reward(
    completions: list[Any],
    issues: Any = None,
    **_: Any,
) -> list[float]:
    issue_rows = _coerce_rows(issues, len(completions), [])
    keywords = ["构成", "主观", "客观", "情节", "法条", "量刑"]
    rewards: list[float] = []
    for completion, sample_issues in zip(completions, issue_rows):
        text = completion_to_text(completion)
        keyword_score = sum(1 for item in keywords if item in text) / len(keywords)
        issue_score = 0.0
        if isinstance(sample_issues, list) and sample_issues:
            issue_score = sum(1 for item in sample_issues if normalize_text(item) in text) / len(sample_issues)
        rewards.append(min(0.6 * keyword_score + 0.4 * issue_score, 1.0))
    return rewards


def build_criminal_reward_functions() -> list:
    return [
        charge_match_reward,
        article_match_reward,
        sentencing_band_reward,
        element_coverage_reward,
    ]
