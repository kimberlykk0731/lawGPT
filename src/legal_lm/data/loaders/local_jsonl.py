from __future__ import annotations

from pathlib import Path

from legal_lm.data.loaders.base import BaseDatasetLoader
from legal_lm.data.schema import LegalSample
from legal_lm.utils.io import load_json, load_jsonl


class LocalJsonlLoader(BaseDatasetLoader):
    def load_split(self, split: str) -> list[LegalSample]:
        path = self._resolve_split_file(split)
        rows = self._load_rows(path)
        samples: list[LegalSample] = []
        for index, row in enumerate(rows):
            sample = LegalSample.from_dict(
                {
                    **row,
                    "source_dataset": self.dataset_name,
                    "domain": row.get("domain", self.dataset_config.get("domain", "general")),
                    "task_type": row.get("task_type", self.dataset_config.get("task_type", "case_reasoning")),
                    "split": split,
                }
            ).require_identifier(f"{self.dataset_name}-{split}-{index}")
            samples.append(sample)
        return samples

    @staticmethod
    def _load_rows(path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            return load_jsonl(path)
        payload = load_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        raise ValueError(f"Unsupported local dataset format: {path}")
