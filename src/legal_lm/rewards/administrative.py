from __future__ import annotations

from typing import Any

from legal_lm.utils.text import completion_to_text


def administrative_act_type_reward(completions: list[Any], **_: Any) -> list[float]:
    keywords = ["行政处罚", "行政许可", "行政强制", "行政确认", "行政裁决"]
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(sum(1 for item in keywords if item in text) / len(keywords))
    return rewards


def procedure_legality_reward(completions: list[Any], **_: Any) -> list[float]:
    keywords = ["程序合法", "告知", "听证", "送达", "法定程序", "权限"]
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(sum(1 for item in keywords if item in text) / len(keywords))
    return rewards


def review_standard_reward(completions: list[Any], **_: Any) -> list[float]:
    keywords = ["事实清楚", "证据确凿", "适用法律正确", "程序合法", "裁量适当"]
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(sum(1 for item in keywords if item in text) / len(keywords))
    return rewards


def remedy_path_reward(completions: list[Any], **_: Any) -> list[float]:
    keywords = ["行政复议", "行政诉讼", "撤销", "变更", "确认违法", "赔偿"]
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        rewards.append(sum(1 for item in keywords if item in text) / len(keywords))
    return rewards


def build_administrative_reward_functions() -> list:
    return [
        administrative_act_type_reward,
        procedure_legality_reward,
        review_standard_reward,
        remedy_path_reward,
    ]
