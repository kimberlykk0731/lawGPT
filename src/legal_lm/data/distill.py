"""CoT 蒸馏 client：用强模型为法律样本生成 <think> 推理。

支持 DeepSeek API / OpenAI 兼容 API / 本地 vLLM。
强制结构校验：<think> 内必须出现"争点 / 大前提 / 小前提 / 结论"四段。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


SYSTEM_PROMPT = (
    "你是一位资深中国法律专家，具有丰富的法律实务经验。"
    "请对用户提出的法律问题进行严谨、专业的分析，输出结构化的法律推理。"
)

DISTILL_PROMPT_TEMPLATE = """请对以下法律问题进行详细分析。

要求：
1. 用 <think>...</think> 标签包裹完整的推理过程，标签外给出最终结论。
2. <think> 内必须依次出现以下四段（用方括号标题标识）：
   [争点] 列出本案核心争议焦点
   [大前提] 引用具体法律条文（写明法律名称与条文编号）
   [小前提] 用案件事实匹配大前提的构成要件
   [结论] 在三段论框架下得出结论
3. 标签外给出 4 段输出：【结论】【法律依据】【推理摘要】【风险提示】

问题：{question}
"""


API_CONFIGS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-reasoner",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "QWEN_API_KEY",
        "model": "qwen-max",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "env_key": "",
        "model": "default",
    },
}


REQUIRED_THINK_SECTIONS = ("[争点]", "[大前提]", "[小前提]", "[结论]")
REQUIRED_OUTPUT_SECTIONS = ("【结论】", "【法律依据】", "【推理摘要】", "【风险提示】")
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class DistillClient:
    api: str = "deepseek"
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise ImportError("Please `pip install openai` to use the distill client.")
        if self.api not in API_CONFIGS:
            raise ValueError(f"Unsupported api={self.api}; expected one of {list(API_CONFIGS)}")
        cfg = API_CONFIGS[self.api]
        api_key = os.environ.get(cfg["env_key"], "EMPTY") if cfg["env_key"] else "EMPTY"
        self._client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
        self.model = self.model or cfg["model"]

    def generate(self, question: str) -> str | None:
        prompt = DISTILL_PROMPT_TEMPLATE.format(question=question)
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None


def passes_structure_check(content: str) -> bool:
    """Verify <think> contains the four required sections AND output has the four headings."""
    if not content:
        return False
    match = THINK_PATTERN.search(content)
    if not match:
        return False
    think = match.group(1)
    if not all(section in think for section in REQUIRED_THINK_SECTIONS):
        return False
    body = THINK_PATTERN.sub("", content)
    return all(heading in body for heading in REQUIRED_OUTPUT_SECTIONS)


def filter_distilled(
    items: Iterable[dict],
    *,
    enforce_structure: bool = True,
) -> list[dict]:
    """Drop items that fail structure check (when enforce_structure=True)."""
    kept: list[dict] = []
    for item in items:
        answer = item.get("answer") or item.get("response") or ""
        if enforce_structure and not passes_structure_check(answer):
            continue
        kept.append(item)
    return kept


__all__ = [
    "API_CONFIGS",
    "DistillClient",
    "filter_distilled",
    "passes_structure_check",
    "REQUIRED_OUTPUT_SECTIONS",
    "REQUIRED_THINK_SECTIONS",
    "SYSTEM_PROMPT",
    "DISTILL_PROMPT_TEMPLATE",
]
