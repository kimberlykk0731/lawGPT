"""学习型奖励模型包装器，输出 [0,1] 标量奖励。

接口对齐 TRL GRPO/PPO 的 `reward_funcs` 调用约定：
    reward_fn(completions, prompts=..., **kwargs) -> list[float]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from legal_lm.utils.text import completion_to_text

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = 1.0 / (1.0 + (1.0 / (2.71828 ** x)))
    else:
        z = (2.71828 ** x) / (1.0 + (2.71828 ** x))
    return float(z)


@dataclass
class LearnedRMReward:
    model_path: str
    device: str | None = None
    batch_size: int = 4
    max_length: int = 4096
    normalize: bool = True

    def __post_init__(self) -> None:
        if AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required for LearnedRMReward.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, num_labels=1, trust_remote_code=True
        )
        self.model.eval()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.inference_mode()
    def score(self, prompts: Iterable[str], responses: Iterable[str]) -> list[float]:
        prompts = list(prompts)
        responses = list(responses)
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk_p = prompts[start : start + self.batch_size]
            chunk_r = responses[start : start + self.batch_size]
            texts = [f"{p}\n{r}" for p, r in zip(chunk_p, chunk_r)]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1).float().cpu().tolist()
            scores.extend(logits)
        if self.normalize:
            scores = [_sigmoid(float(s)) for s in scores]
        return scores

    def __call__(
        self,
        completions: list[Any],
        prompts: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        prompt_list = list(prompts) if prompts else [""] * len(completions)
        responses = [completion_to_text(item) for item in completions]
        if len(prompt_list) != len(responses):
            prompt_list = (prompt_list * len(responses))[: len(responses)]
        return self.score(prompt_list, responses)


__all__ = ["LearnedRMReward"]
