"""民事/侵权领域奖励函数（TRL GRPO 风格）。

- liability_allocation_reward: 责任比例划分准确性
- compensation_items_reward:   赔偿项目完整性
- causation_reward:            因果关系论证
- burden_of_proof_reward:      举证责任分配
- statute_of_limitations_reward: 诉讼时效判断
"""

from __future__ import annotations

import re
from typing import Any

from legal_lm.rewards.common import _coerce_rows, _parse_metadata_rows
from legal_lm.utils.text import completion_to_text


COMPENSATION_ITEMS = [
    "医疗费", "误工费", "护理费", "交通费", "住宿费", "住院伙食补助费",
    "营养费", "残疾赔偿金", "残疾辅助器具费", "丧葬费", "死亡赔偿金",
    "被扶养人生活费", "精神损害抚慰金", "鉴定费", "后续治疗费",
]

BURDEN_OF_PROOF_KEYWORDS = [
    "举证责任", "举证责任倒置", "谁主张谁举证", "推定过错",
    "无过错责任", "过错推定", "证据", "举证不能",
]

CAUSATION_KEYWORDS = [
    "因果关系", "相当因果关系", "直接因果", "间接原因",
    "损害结果", "侵权行为", "因果链",
]

LIMITATION_KEYWORDS = [
    "诉讼时效", "三年", "一年", "二十年", "最长诉讼时效",
    "时效中断", "时效中止", "除斥期间",
]

LIABILITY_PATTERN = re.compile(r"(\d{1,3})\s*%|百分之([一二三四五六七八九十百零\d]+)")

_CN_NUMBER_MAP = {
    "十": 10, "二十": 20, "三十": 30, "四十": 40, "五十": 50,
    "六十": 60, "七十": 70, "八十": 80, "九十": 90, "一百": 100,
}


def _parse_liability_pct(text: str) -> int | None:
    matches = LIABILITY_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[0][0] or matches[0][1]
    if not raw:
        return None
    if raw.isdigit():
        try:
            return int(raw)
        except ValueError:
            return None
    return _CN_NUMBER_MAP.get(raw)


def liability_allocation_reward(
    completions: list[Any],
    answer: Any = None,
    metadata_json: Any = None,
    **_: Any,
) -> list[float]:
    answers = _coerce_rows(answer, len(completions), "")
    metadata_rows = _parse_metadata_rows(metadata_json, len(completions))
    rewards: list[float] = []
    for completion, gold_answer, metadata in zip(completions, answers, metadata_rows):
        text = completion_to_text(completion)
        pred_pct = _parse_liability_pct(text)
        gold_pct = metadata.get("liability_pct")
        if gold_pct is None:
            gold_pct = _parse_liability_pct(str(gold_answer))
        if gold_pct is None:
            rewards.append(0.5 if pred_pct is not None else 0.0)
            continue
        if pred_pct is None:
            rewards.append(0.0)
            continue
        diff = abs(pred_pct - int(gold_pct))
        if diff <= 10:
            rewards.append(1.0)
        elif diff <= 20:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def compensation_items_reward(
    completions: list[Any],
    metadata_json: Any = None,
    **_: Any,
) -> list[float]:
    metadata_rows = _parse_metadata_rows(metadata_json, len(completions))
    rewards: list[float] = []
    for completion, metadata in zip(completions, metadata_rows):
        text = completion_to_text(completion)
        gold = metadata.get("compensation_items") or COMPENSATION_ITEMS
        gold_set = set(map(str, gold))
        if not gold_set:
            rewards.append(0.0)
            continue
        found = sum(1 for item in gold_set if item in text)
        rewards.append(found / len(gold_set))
    return rewards


def causation_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        found = sum(1 for kw in CAUSATION_KEYWORDS if kw in text)
        rewards.append(min(found / 2.0, 1.0))
    return rewards


def burden_of_proof_reward(
    completions: list[Any],
    metadata_json: Any = None,
    **_: Any,
) -> list[float]:
    metadata_rows = _parse_metadata_rows(metadata_json, len(completions))
    rewards: list[float] = []
    for completion, metadata in zip(completions, metadata_rows):
        text = completion_to_text(completion)
        found = sum(1 for kw in BURDEN_OF_PROOF_KEYWORDS if kw in text)
        base = min(found / 2.0, 1.0)
        if "burden_inverted" not in metadata:
            rewards.append(base)
            continue
        inverted = bool(metadata["burden_inverted"])
        mentions_inversion = ("举证责任倒置" in text) or ("推定过错" in text)
        if inverted and mentions_inversion:
            rewards.append(1.0)
        elif (not inverted) and (not mentions_inversion):
            rewards.append(base)
        else:
            rewards.append(base * 0.5)
    return rewards


def statute_of_limitations_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        found = sum(1 for kw in LIMITATION_KEYWORDS if kw in text)
        rewards.append(min(found / 2.0, 1.0))
    return rewards


def build_civil_reward_functions() -> list:
    return [
        liability_allocation_reward,
        compensation_items_reward,
        causation_reward,
        burden_of_proof_reward,
        statute_of_limitations_reward,
    ]


__all__ = [
    "build_civil_reward_functions",
    "liability_allocation_reward",
    "compensation_items_reward",
    "causation_reward",
    "burden_of_proof_reward",
    "statute_of_limitations_reward",
]
