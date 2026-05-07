from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from legal_lm.data.schema import LegalSample


class BaseDatasetLoader(ABC):
    def __init__(self, dataset_name: str, dataset_config: dict[str, Any]):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.root = Path(dataset_config["root"])

    @abstractmethod
    def load_split(self, split: str) -> list[LegalSample]:
        raise NotImplementedError

    def _resolve_split_file(self, split: str) -> Path:
        split_files = self.dataset_config.get("split_files", {})
        file_name = split_files.get(split)
        if not file_name:
            raise KeyError(f"Split '{split}' is not configured for dataset '{self.dataset_name}'.")
        return self.root / file_name
