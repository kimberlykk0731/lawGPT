from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from legal_lm.utils.config import load_experiment_config
from legal_lm.utils.io import ensure_dir


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    normalized = str(dtype_name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    return config


def build_lora_config(config: dict[str, Any]) -> LoraConfig | None:
    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return None
    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(lora_cfg.get("target_modules", [])),
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = config["model"]
    model_name = model_cfg["model_name_or_path"]
    tokenizer_name = model_cfg.get("tokenizer_name_or_path", model_name)
    torch_dtype = parse_torch_dtype(model_cfg.get("torch_dtype"))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("load_in_4bit", False):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def load_json_dataset(train_file: str | Path, validation_file: str | Path | None = None) -> tuple[Dataset, Dataset | None]:
    data_files: dict[str, str] = {"train": str(train_file)}
    if validation_file:
        data_files["validation"] = str(validation_file)
    dataset_dict = load_dataset("json", data_files=data_files)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"] if "validation" in dataset_dict else None
    return train_dataset, eval_dataset


def ensure_output_dir(config: dict[str, Any]) -> Path:
    output_dir = config["run"]["output_dir"]
    return ensure_dir(output_dir)


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    path = ensure_dir(output_dir) / "resolved_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return path


def ensure_dataset_columns(dataset: Dataset | None, required_columns: list[str], dataset_name: str) -> None:
    if dataset is None:
        return
    column_names = set(getattr(dataset, "column_names", []))
    missing = [column for column in required_columns if column not in column_names]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def _supports_argument(target: Any, name: str) -> bool:
    signature = inspect.signature(target)
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def filter_supported_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None and _supports_argument(target, key)}


def instantiate_supported(target: Any, kwargs: dict[str, Any]) -> Any:
    return target(**filter_supported_kwargs(target, kwargs))


def build_interval_kwargs(
    *,
    has_eval: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
) -> dict[str, Any]:
    evaluation_strategy = "steps" if has_eval else "no"
    kwargs: dict[str, Any] = {
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_strategy": "steps",
        "logging_strategy": "steps",
    }
    if has_eval:
        kwargs["eval_steps"] = eval_steps
    for key in ("evaluation_strategy", "eval_strategy"):
        kwargs[key] = evaluation_strategy
    return kwargs


def attach_processing_class(kwargs: dict[str, Any], trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    if _supports_argument(trainer_cls, "processing_class"):
        kwargs["processing_class"] = tokenizer
    elif _supports_argument(trainer_cls, "tokenizer"):
        kwargs["tokenizer"] = tokenizer
    return kwargs


def trainer_supports_eval(trainer_cls: Any) -> bool:
    return _supports_argument(trainer_cls, "eval_dataset")
