"""RLVR (Reinforcement Learning with Verifiable Rewards) for legal tasks.

Designed for GRPO on tasks whose ground truth can be programmatically scored:
- crime_prediction: JSON list of charges, F1 over set
- contract_review:  JSON list of risk labels, F1 over set

Anti-hacking: format reward capped at +0.1 so the model can't ignore content.

A module-level `METRICS_BUFFER` accumulates per-completion stats so a Trainer
callback can flush them to swanlab at log time (`flush_rlvr_metrics()`).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from threading import Lock
from typing import Any

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def parse_json_array(text: str) -> list[str] | None:
    """Robustly extract a JSON list of strings from a model completion."""
    match = _JSON_BLOCK_RE.search(text) or _JSON_ARRAY_RE.search(text)
    if match is None:
        return None
    candidate = match.group(1) if match.re is _JSON_BLOCK_RE else match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item).strip() for item in parsed if str(item).strip()]


def _f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def crime_reward(completion: str, ground_truth: dict[str, Any]) -> float:
    pred = parse_json_array(completion)
    if pred is None:
        return -0.5
    return _f1(set(pred), set(ground_truth.get("crimes", [])))


def contract_reward(completion: str, ground_truth: dict[str, Any]) -> float:
    pred = parse_json_array(completion)
    if pred is None:
        return -0.5
    return _f1(set(pred), set(ground_truth.get("risks", [])))


def format_reward(completion: str) -> float:
    return 0.1 if parse_json_array(completion) is not None else 0.0


_TASK_REWARDS = {
    "crime_prediction": crime_reward,
    "contract_review": contract_reward,
}


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


_BUFFER_LOCK = Lock()
METRICS_BUFFER: dict[str, list[float]] = defaultdict(list)


def _record(key: str, value: float) -> None:
    with _BUFFER_LOCK:
        METRICS_BUFFER[key].append(value)


def flush_rlvr_metrics() -> dict[str, float]:
    """Aggregate buffered per-completion stats into scalar tracker-ready metrics."""
    with _BUFFER_LOCK:
        if not METRICS_BUFFER:
            return {}
        agg: dict[str, float] = {}
        crime = METRICS_BUFFER.get("legal/crime_f1", [])
        contract = METRICS_BUFFER.get("legal/contract_f1", [])
        if crime:
            agg["legal/crime_f1"] = sum(crime) / len(crime)
        if contract:
            agg["legal/contract_f1"] = sum(contract) / len(contract)
        parse_fail = METRICS_BUFFER.get("legal/parse_fail", [])
        if parse_fail:
            agg["legal/parse_failure_rate"] = sum(parse_fail) / len(parse_fail)
        empty = METRICS_BUFFER.get("legal/empty_pred", [])
        if empty:
            agg["legal/empty_pred_rate"] = sum(empty) / len(empty)
        main_r = METRICS_BUFFER.get("critic/main_reward", [])
        if main_r:
            agg["critic/main_reward/mean"] = sum(main_r) / len(main_r)
        fmt_r = METRICS_BUFFER.get("critic/format_reward", [])
        if fmt_r:
            agg["critic/format_reward/mean"] = sum(fmt_r) / len(fmt_r)
            agg["critic/format_reward/frac"] = sum(1 for r in fmt_r if r > 0) / len(fmt_r)
        METRICS_BUFFER.clear()
    return agg


def legal_reward_fn(prompts: Any, completions: list[Any], **kwargs: Any) -> list[float]:
    """GRPO reward dispatcher. trl injects extra fields from the dataset as kwargs."""
    tasks = kwargs.get("task") or []
    gts = kwargs.get("ground_truth") or []
    rewards: list[float] = []
    for completion, task, gt in zip(completions, tasks, gts):
        text = _completion_text(completion)
        pred = parse_json_array(text)
        _record("legal/parse_fail", 1.0 if pred is None else 0.0)
        _record("legal/empty_pred", 1.0 if (pred is not None and not pred) else 0.0)

        scorer = _TASK_REWARDS.get(task)
        main = scorer(text, gt) if scorer else 0.0
        fmt = format_reward(text)

        _record("critic/main_reward", main)
        _record("critic/format_reward", fmt)
        if task == "crime_prediction":
            _record("legal/crime_f1", max(main, 0.0))
        elif task == "contract_review":
            _record("legal/contract_f1", max(main, 0.0))

        rewards.append(main + fmt)
    return rewards


__all__ = [
    "contract_reward",
    "crime_reward",
    "flush_rlvr_metrics",
    "format_reward",
    "legal_reward_fn",
    "METRICS_BUFFER",
    "parse_json_array",
]
