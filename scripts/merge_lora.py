"""把 LoRA adapter 合并回基座，产出可直接被 vLLM/HF Pipeline 加载的完整模型。

仅在确认训练完成且 adapter 与基座对齐时使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def merge(base_model: str, adapter: str, output: str, dtype: str = "bfloat16") -> None:
    torch_dtype = _DTYPE_MAP.get(dtype.lower(), torch.bfloat16)
    print(f"[merge_lora] loading base={base_model} adapter={adapter} dtype={dtype}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[merge_lora] merged -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA into base model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    merge(args.base_model, args.adapter, args.output, args.dtype)


if __name__ == "__main__":
    main()
