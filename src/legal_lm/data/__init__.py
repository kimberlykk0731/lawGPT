"""Data layer for Chinese legal post-training."""

from .build_preference import build_grpo_rows, build_preference_dataset, build_preference_rows
from .build_sft import build_sft_dataset, build_sft_rows
from .registry import load_samples
from .schema import LegalDomain, LegalSample, SFTMode, TaskType

__all__ = [
    "LegalDomain",
    "LegalSample",
    "SFTMode",
    "TaskType",
    "build_grpo_rows",
    "build_preference_dataset",
    "build_preference_rows",
    "build_sft_dataset",
    "build_sft_rows",
    "load_samples",
]
