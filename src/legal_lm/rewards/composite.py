"""规则奖励 + 学习型 RM 的可参数化组合。

GRPO 通常用纯规则；PPO 默认 rule:learned = 0.3:0.7；权重在配置中可调。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from legal_lm.rewards import build_reward_functions
from legal_lm.rewards.learned_rm import LearnedRMReward


def _safe_mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else 0.0


@dataclass
class CompositeWeights:
    rule: float = 0.3
    learned: float = 0.7

    def normalized(self) -> "CompositeWeights":
        total = self.rule + self.learned
        if total <= 0:
            return CompositeWeights(rule=0.5, learned=0.5)
        return CompositeWeights(rule=self.rule / total, learned=self.learned / total)


@dataclass
class CompositeReward:
    rule_funcs: list[Callable[..., list[float]]] = field(default_factory=list)
    learned: LearnedRMReward | None = None
    weights: CompositeWeights = field(default_factory=CompositeWeights)

    @classmethod
    def from_paths(
        cls,
        rm_path: str | None,
        rule_profile: str = "general",
        weights: dict[str, float] | None = None,
    ) -> "CompositeReward":
        rule_funcs = build_reward_functions(rule_profile)
        learned = LearnedRMReward(model_path=rm_path) if rm_path else None
        w = CompositeWeights(**(weights or {}))
        return cls(rule_funcs=rule_funcs, learned=learned, weights=w.normalized())

    def __call__(
        self,
        completions: list[Any],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        n = len(completions)
        rule_matrix: list[list[float]] = []
        for fn in self.rule_funcs:
            try:
                vals = fn(completions, prompts=prompts, **kwargs)
            except TypeError:
                vals = fn(completions, **kwargs)
            rule_matrix.append([float(v) for v in vals])
        if rule_matrix:
            rule_scores = [
                _safe_mean(rule_matrix[r][i] for r in range(len(rule_matrix)))
                for i in range(n)
            ]
        else:
            rule_scores = [0.0] * n

        if self.learned is None:
            return rule_scores
        learned_scores = self.learned(completions, prompts=prompts, **kwargs)
        w = self.weights
        return [w.rule * r + w.learned * l for r, l in zip(rule_scores, learned_scores)]

    def as_callable(self) -> Callable[..., list[float]]:
        return self.__call__


__all__ = ["CompositeReward", "CompositeWeights"]
