from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _resolve_child_path(base_path: Path, candidate: str | Path) -> Path:
    child = Path(candidate)
    if child.is_absolute():
        return child

    direct = child
    if direct.exists():
        return direct

    relative = base_path.parent / child
    if relative.exists():
        return relative

    return relative


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_config_chain(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = load_yaml(config_path)
    base_path = config.get("base_config")
    if not base_path:
        return config

    base_config = resolve_config_chain(_resolve_child_path(config_path, base_path))
    return deep_merge(base_config, {k: v for k, v in config.items() if k != "base_config"})


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path)
    config = resolve_config_chain(experiment_path)
    dataset_config = config.get("dataset_config")
    if dataset_config:
        config["dataset_config_path"] = str(_resolve_child_path(experiment_path, dataset_config))
    return config
