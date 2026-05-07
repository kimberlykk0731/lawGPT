"""Shared utilities for the project."""

from .config import deep_merge, load_yaml, resolve_config_chain
from .io import dump_json, dump_jsonl, ensure_dir, ensure_parent, iter_jsonl, load_json, load_jsonl
from .text import (
    completion_to_text,
    extract_articles,
    extract_number,
    normalize_text,
    token_overlap_ratio,
)

__all__ = [
    "completion_to_text",
    "deep_merge",
    "dump_json",
    "dump_jsonl",
    "ensure_dir",
    "ensure_parent",
    "extract_articles",
    "extract_number",
    "iter_jsonl",
    "load_json",
    "load_jsonl",
    "load_yaml",
    "normalize_text",
    "resolve_config_chain",
    "token_overlap_ratio",
]
