"""Reference implementation: training Qwen3-30B-A3B with the domain-aware aux loss.

This is *reference-only* — Stage 2's main deliverable for the resume story
is `analyze_router.py`'s output. Running this end-to-end on Qwen3-30B-A3B
needs serious memory engineering (LoRA + ZeRO-3 with offload, ~4×H100 minimum),
and we don't claim trained numbers in the resume.

Kept around so an interviewer asking "show me the aux-loss training entry"
gets a concrete CLI to read instead of an `<unimplemented>`. To actually
exercise the aux loss in a runnable way, point `--model_name_or_path` at the
Stage-3 upcycled `qwen3-1.7b-moe-8e` (much smaller, easier to fit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from stages.stage2_moe_router.domain_aware_aux_loss import domain_aware_aux_loss

DOMAIN_TO_ID = {"刑事": 0, "民事": 1, "商事": 2, "行政": 3, "知产": 4}


def load_domain_dataset(jsonl_path: Path, tokenizer, max_length: int) -> Dataset:
    rows = []
    with jsonl_path.open(encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            domain = row.get("domain") or "民事"
            encoded = tokenizer(row["text"], truncation=True, max_length=max_length,
                                padding=False)
            rows.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": list(encoded["input_ids"]),
                "domain_id": DOMAIN_TO_ID.get(domain, 1),
            })
    return Dataset.from_list(rows)


class DomainAuxTrainer(Trainer):
    def __init__(self, *args, num_experts: int, num_domains: int,
                 intra_weight: float, inter_weight: float, aux_weight: float,
                 top_k: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts = num_experts
        self.num_domains = num_domains
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.aux_weight = aux_weight
        self.top_k = top_k

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        domain_ids = inputs.pop("domain_id")
        outputs = model(**inputs, output_router_logits=True)
        lm_loss = outputs.loss
        aux = domain_aware_aux_loss(
            router_logits_per_layer=outputs.router_logits,
            domain_ids=domain_ids.to(lm_loss.device),
            num_experts=self.num_experts,
            num_domains=self.num_domains,
            top_k=self.top_k,
            intra_weight=self.intra_weight,
            inter_weight=self.inter_weight,
        )
        loss = lm_loss + self.aux_weight * aux
        try:
            from swanlab.integration.wandb import wandb
            if wandb.run is not None and self.state.global_step % self.args.logging_steps == 0:
                wandb.log({
                    "loss/lm_ce": lm_loss.item(),
                    "loss/aux_total": aux.item(),
                    "loss/total": loss.item(),
                }, step=self.state.global_step)
        except ImportError:
            pass
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--num_domains", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--intra_weight", type=float, default=1.0)
    parser.add_argument("--inter_weight", type=float, default=0.5)
    parser.add_argument("--aux_weight", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--deepspeed", default="stages/stage1_sft_grpo/configs/ds_zero3_bf16.json")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage2-domain-aux-v1")
    args = parser.parse_args()

    import os
    os.environ.setdefault("SWANLAB_PROJECT", args.swanlab_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.output_router_logits = True

    dataset = load_domain_dataset(args.dataset_jsonl, tokenizer, args.max_length)

    config = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to=["swanlab"],
        run_name=args.swanlab_run_name,
        remove_unused_columns=False,
    )

    trainer = DomainAuxTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        num_experts=args.num_experts,
        num_domains=args.num_domains,
        intra_weight=args.intra_weight,
        inter_weight=args.inter_weight,
        aux_weight=args.aux_weight,
        top_k=args.top_k,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
