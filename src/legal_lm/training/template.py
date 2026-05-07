"""模板注册中心：覆盖主流中文/通用基座的 chat template。

设计：HuggingFace tokenizer 自带 chat_template 时直接使用；缺失或想覆盖时，
通过 `apply_template(model_name, messages)` 主动渲染。换基座只需在 base.yaml
里改 `model.model_name_or_path`，模板由本注册中心自动选择。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

ChatMessage = dict[str, str]
TemplateRenderer = Callable[[Iterable[ChatMessage], bool], str]


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    matchers: tuple[str, ...]
    renderer: TemplateRenderer


def _qwen_renderer(messages: Iterable[ChatMessage], add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _deepseek_renderer(messages: Iterable[ChatMessage], add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n\n".join(parts)


def _glm_renderer(messages: Iterable[ChatMessage], add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _internlm_renderer(messages: Iterable[ChatMessage], add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _generic_renderer(messages: Iterable[ChatMessage], add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


_REGISTRY: tuple[TemplateSpec, ...] = (
    TemplateSpec(name="qwen", matchers=("qwen",), renderer=_qwen_renderer),
    TemplateSpec(name="deepseek", matchers=("deepseek",), renderer=_deepseek_renderer),
    TemplateSpec(name="glm", matchers=("glm", "chatglm"), renderer=_glm_renderer),
    TemplateSpec(name="internlm", matchers=("internlm",), renderer=_internlm_renderer),
    TemplateSpec(name="generic", matchers=("",), renderer=_generic_renderer),
)


def select_template(model_name_or_path: str) -> TemplateSpec:
    name = (model_name_or_path or "").lower()
    for spec in _REGISTRY:
        for token in spec.matchers:
            if token and token in name:
                return spec
    return _REGISTRY[-1]


def apply_template(
    model_name_or_path: str,
    messages: Iterable[ChatMessage],
    add_generation_prompt: bool = True,
) -> str:
    return select_template(model_name_or_path).renderer(messages, add_generation_prompt)


def render_with_tokenizer(
    tokenizer,
    messages: Iterable[ChatMessage],
    model_name_or_path: str,
    add_generation_prompt: bool = True,
) -> str:
    """优先 tokenizer.apply_chat_template，缺失时回退本地注册中心。"""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:  # noqa: BLE001
            pass
    return apply_template(model_name_or_path, messages, add_generation_prompt)


__all__ = [
    "TemplateSpec",
    "apply_template",
    "render_with_tokenizer",
    "select_template",
]
