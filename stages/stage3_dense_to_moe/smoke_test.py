"""Smoke test for the freshly upcycled Qwen3-1.7B-MoE.

Loads the model, runs a few legal prompts, and dumps router activation stats
per expert so you can confirm the random router is not stuck on a single
expert. Pushes the same numbers to swanlab so you have a screenshottable
post-upcycle baseline before Stage 3 fine-tuning starts moving them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "请分析下列合同条款是否存在风险：甲方有权随时单方解除合同。",
    "被告人在凌晨潜入仓库盗走价值五万元的电脑，应认定何罪？",
    "民事侵权赔偿中，精神损害的计算依据是什么？",
    "行政复议与行政诉讼的衔接关系如何？",
    "公司股东查阅会计账簿被拒，能否提起诉讼？",
    "他人在同类商品上使用近似商标是否构成侵权？",
]


def _swanlab_init(args: argparse.Namespace):
    try:
        import swanlab  # noqa: F401
    except ImportError:
        return None
    from stages._swanlab_shim import wandb
    wandb.init(
        project=args.swanlab_project,
        name=args.swanlab_run_name,
        tags=["stage3", "smoke_test", "upcycle"],
        config=vars(args),
    )
    return wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage3-smoke-test")
    args = parser.parse_args()

    wandb = _swanlab_init(args)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_router_logits=True,
    ).eval()

    expert_counts: Counter[int] = Counter()
    per_layer: dict[int, Counter[int]] = {}
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)
        for layer_idx, router_logit in enumerate(outputs.router_logits):
            if router_logit is None:
                continue
            topk = router_logit.topk(args.top_k, dim=-1).indices.flatten().tolist()
            for expert_id in topk:
                expert_counts[expert_id] += 1
                per_layer.setdefault(layer_idx, Counter())[expert_id] += 1

    total = sum(expert_counts.values()) or 1
    print(f"Total token-expert assignments: {total}")
    activated_experts = len(expert_counts)
    for expert_id in range(args.num_experts):
        count = expert_counts.get(expert_id, 0)
        share = count / total
        print(f"  expert {expert_id}: {count} ({share:.2%})")
        if wandb is not None:
            wandb.log({f"moe/smoke/expert_{expert_id}_share": share})

    # Aggregate signals
    if wandb is not None:
        max_share = max(expert_counts.values()) / total
        # Std across all experts (including zero-token ones)
        shares = [expert_counts.get(i, 0) / total for i in range(args.num_experts)]
        mean = sum(shares) / len(shares)
        std = (sum((s - mean) ** 2 for s in shares) / len(shares)) ** 0.5
        dead = sum(1 for s in shares if s == 0)
        wandb.log({
            "moe/smoke/active_experts": activated_experts,
            "moe/smoke/max_share": max_share,
            "moe/smoke/load_std": std,
            "moe/smoke/dead_experts": dead,
        })

    if activated_experts < max(2, args.num_experts // 2):
        print(f"[warn] only {activated_experts}/{args.num_experts} experts saw any tokens — "
              "router may be collapsed; revisit mergekit positive_prompts.")
    else:
        print("[ok] router exercising multiple experts; ready for continued training.")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
