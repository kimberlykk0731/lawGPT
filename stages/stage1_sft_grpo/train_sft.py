"""Qwen3-8B full-parameter SFT on the 350k legal corpus.

deepspeed --num_gpus=4 stages/stage1_sft_grpo/train_sft.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_path data/processed/sft_350k \
    --output_dir ckpts/legalgpt-8b-sft \
    --max_seq_length 4096 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed stages/stage1_sft_grpo/configs/ds_zero3_bf16.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _load_dataset(path: Path) -> Dataset:
    if path.is_dir() and (path / "dataset_info.json").exists():
        return load_from_disk(str(path))
    rows: list[dict] = []
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage1-sft-v1")
    args = parser.parse_args()

    import os
    os.environ.setdefault("SWANLAB_PROJECT", args.swanlab_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype="bfloat16",
    )
    dataset = _load_dataset(args.dataset_path)

    config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        report_to=["swanlab"],
        run_name=args.swanlab_run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
