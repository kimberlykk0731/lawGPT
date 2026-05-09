"""Stage 4-B: Offline top-50 logits KL distillation.

Two sub-steps:
  1. dump_teacher_topk: run teacher once over all (prompt, response) sequences,
     persist top-50 (values, indices) per token. Cuts storage 99.96% vs. full vocab.
  2. train_student_kl: standard CE + KL on the top-50 positions.

Run dump on the teacher GPU pod, train_student on a separate pod that only
holds the student model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class JsonlMessages(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        text = self.tokenizer.apply_chat_template(
            self.rows[idx]["messages"],
            tokenize=False,
            enable_thinking=False,
        )
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_tensors="pt", padding="max_length")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def dump_teacher_topk(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()

    dataset = JsonlMessages(args.dataset_jsonl, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: list[dict] = []
    for batch in loader:
        with torch.no_grad():
            logits = teacher(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            ).logits  # (B, L, V)
            topk = logits.topk(args.top_k, dim=-1)
            cache.append({
                "topk_values": topk.values.half().cpu(),
                "topk_indices": topk.indices.cpu(),
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            })
    torch.save(cache, str(args.cache_path))
    print(f"[dump] cached {len(cache)} batches -> {args.cache_path}")


class CachedKLDataset(Dataset):
    def __init__(self, cache_path: Path):
        self.batches = torch.load(str(cache_path))

    def __len__(self) -> int:
        return sum(b["input_ids"].size(0) for b in self.batches)

    def __getitem__(self, idx: int) -> dict:
        for batch in self.batches:
            n = batch["input_ids"].size(0)
            if idx < n:
                return {
                    "input_ids": batch["input_ids"][idx],
                    "attention_mask": batch["attention_mask"][idx],
                    "labels": batch["input_ids"][idx],
                    "teacher_topk_values": batch["topk_values"][idx],
                    "teacher_topk_indices": batch["topk_indices"][idx],
                }
            idx -= n
        raise IndexError


class DistillTrainer(Trainer):
    def __init__(self, *args, temperature: float = 2.0, alpha: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        teacher_topk_values = inputs.pop("teacher_topk_values").to(model.device)
        teacher_topk_indices = inputs.pop("teacher_topk_indices").to(model.device)

        outputs = model(**inputs)
        student_logits = outputs.logits

        # Standard next-token shift for both CE and KL:
        # student_logits[:, :-1] predicts labels[:, 1:].
        shifted_logits = student_logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_teacher_values = teacher_topk_values[..., :-1, :].contiguous()
        shifted_teacher_indices = teacher_topk_indices[..., :-1, :].contiguous()
        student_topk = shifted_logits.gather(-1, shifted_teacher_indices)

        T = self.temperature
        teacher_probs = F.softmax(shifted_teacher_values.float() / T, dim=-1)
        student_log_probs = F.log_softmax(student_topk / T, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

        ce_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        try:
            from swanlab.integration.wandb import wandb
            if wandb.run is not None and self.state.global_step % self.args.logging_steps == 0:
                with torch.no_grad():
                    student_full_probs = F.softmax(student_logits.float(), dim=-1)
                    teacher_full_topk = F.softmax(teacher_topk_values.float(), dim=-1)
                    teacher_entropy = -(teacher_full_topk *
                                        teacher_full_topk.clamp(min=1e-12).log()).sum(-1).mean()
                    student_entropy = -(student_full_probs *
                                        student_full_probs.clamp(min=1e-12).log()).sum(-1).mean()
                    student_top1 = shifted_logits.argmax(-1)
                    in_topk = (student_top1.unsqueeze(-1) == shifted_teacher_indices).any(-1)
                    agreement_top1 = in_topk.float().mean()
                    student_mass_in_topk = student_full_probs[..., :-1, :].gather(
                        -1, shifted_teacher_indices
                    ).sum(-1).mean()
                wandb.log({
                    "loss/total": loss.item(),
                    "loss/ce": ce_loss.item(),
                    "loss/kl": kl_loss.item(),
                    "loss/alpha": self.alpha,
                    "kd/teacher_entropy": teacher_entropy.item(),
                    "kd/student_entropy": student_entropy.item(),
                    "kd/agreement_top1": agreement_top1.item(),
                    "kd/student_topk_prob_mass": student_mass_in_topk.item(),
                }, step=self.state.global_step)
        except ImportError:
            pass

        return (loss, outputs) if return_outputs else loss


def train_student_kl(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    dataset = CachedKLDataset(args.cache_path)

    import os
    os.environ.setdefault("SWANLAB_PROJECT", args.swanlab_project)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to=["swanlab"],
        run_name=args.swanlab_run_name,
    )

    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)

    p_dump = sub.add_parser("dump", help="Offline dump teacher top-k logits")
    p_dump.add_argument("--teacher_model", default="ckpts/legalgpt-8b-grpo")
    p_dump.add_argument("--dataset_jsonl", type=Path, required=True)
    p_dump.add_argument("--cache_path", type=Path, required=True)
    p_dump.add_argument("--max_length", type=int, default=2048)
    p_dump.add_argument("--batch_size", type=int, default=4)
    p_dump.add_argument("--top_k", type=int, default=50)

    p_train = sub.add_parser("train", help="Train student with cached KL targets")
    p_train.add_argument("--local_rank", type=int, default=-1,
                         help="Injected by deepspeed launcher; not user-set.")
    p_train.add_argument("--student_model", default="ckpts/student-warmup")
    p_train.add_argument("--cache_path", type=Path, required=True)
    p_train.add_argument("--output_dir", type=Path, required=True)
    p_train.add_argument("--learning_rate", type=float, default=2e-5)
    p_train.add_argument("--num_train_epochs", type=float, default=2)
    p_train.add_argument("--per_device_train_batch_size", type=int, default=4)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p_train.add_argument("--temperature", type=float, default=2.0)
    p_train.add_argument("--alpha", type=float, default=0.3)
    p_train.add_argument("--logging_steps", type=int, default=10)
    p_train.add_argument("--swanlab_project", default="legalgpt-2026")
    p_train.add_argument("--swanlab_run_name", default="stage4b-logits-kl")

    args = parser.parse_args()
    if args.step == "dump":
        dump_teacher_topk(args)
    else:
        train_student_kl(args)


if __name__ == "__main__":
    main()
