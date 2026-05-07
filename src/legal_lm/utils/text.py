from __future__ import annotations

import re
from typing import Any


ARTICLE_PATTERN = re.compile(r"第([一二三四五六七八九十百千万0-9]+)条")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        if "text" in completion:
            return str(completion["text"])
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            parts.append(completion_to_text(item))
        return "\n".join(part for part in parts if part)
    return str(completion)


def extract_articles(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return sorted(set(ARTICLE_PATTERN.findall(normalized)))


def extract_number(text: str | None) -> float | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    match = NUMBER_PATTERN.search(normalized)
    return float(match.group(0)) if match else None


def _tokenize(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return re.findall(r"[\u4e00-\u9fff]{1,4}|[a-zA-Z0-9_]+", normalized)


def token_overlap_ratio(left: str | None, right: str | None) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
