from __future__ import annotations

from pathlib import Path
from typing import Any

from legal_lm.data.loaders import CAILLoader, JECQALoader, LeCaRDv2Loader, LocalJsonlLoader
from legal_lm.data.loaders.base import BaseDatasetLoader
from legal_lm.data.schema import LegalSample
from legal_lm.utils.config import load_yaml


class DatasetRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseDatasetLoader]] = {}

    def register(self, name: str, loader_cls: type[BaseDatasetLoader]) -> None:
        self._registry[name] = loader_cls

    def create(self, name: str, dataset_name: str, dataset_config: dict[str, Any]) -> BaseDatasetLoader:
        if name not in self._registry:
            raise KeyError(f"Unknown dataset loader: {name}")
        return self._registry[name](dataset_name, dataset_config)


def build_default_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register("local_jsonl", LocalJsonlLoader)
    registry.register("cail", CAILLoader)
    registry.register("jec_qa", JECQALoader)
    registry.register("lecard_v2", LeCaRDv2Loader)
    return registry


def load_dataset_specs(config_path: str | Path) -> dict[str, dict[str, Any]]:
    config = load_yaml(config_path)
    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid dataset config: {config_path}")
    return datasets


def load_samples(dataset_config_path: str | Path, dataset_name: str, split: str) -> list[LegalSample]:
    dataset_specs = load_dataset_specs(dataset_config_path)
    if dataset_name not in dataset_specs:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    dataset_config = dataset_specs[dataset_name]
    loader_name = dataset_config.get("loader")
    registry = build_default_registry()
    loader = registry.create(loader_name, dataset_name, dataset_config)
    return loader.load_split(split)
