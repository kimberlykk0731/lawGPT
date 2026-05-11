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
    parser.add_argument("--dump_json", type=Path, default=None,
                        help="Optional path to dump per-expert shares + per-layer "
                             "counts; pair of these JSONs are what "
                             "compare_stage3_smoke.py needs for the before/after bar chart.")
    parser.add_argument("--compare_to", type=Path, default=None,
                        help="If set, after this smoke run, render a (before vs after) "
                             "bar chart from --compare_to (before JSON) and this run's "
                             "shares, log it to swanlab as a single Image.")
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
    shares = [expert_counts.get(i, 0) / total for i in range(args.num_experts)]
    for expert_id, share in enumerate(shares):
        print(f"  expert {expert_id}: {expert_counts.get(expert_id, 0)} ({share:.2%})")

    # One key with N steps (expert_id) so swanlab renders ONE chart with
    # `num_experts` points instead of `num_experts` charts with one point.
    if wandb is not None:
        for expert_id, share in enumerate(shares):
            wandb.log({"moe/smoke/expert_share": share}, step=expert_id)

    # Aggregate signals (single-point but at least one per run)
    if wandb is not None:
        max_share = max(expert_counts.values()) / total if expert_counts else 0.0
        mean = sum(shares) / len(shares)
        std = (sum((s - mean) ** 2 for s in shares) / len(shares)) ** 0.5
        dead = sum(1 for s in shares if s == 0)
        wandb.log({
            "moe/smoke/active_experts": activated_experts,
            "moe/smoke/max_share": max_share,
            "moe/smoke/load_std": std,
            "moe/smoke/dead_experts": dead,
        })

    # Always dump JSON if requested — paired before/after JSONs feed the
    # comparison bar chart that's the actual screenshot for the resume.
    if args.dump_json is not None:
        import json
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps({
            "model": str(args.model),
            "run_name": args.swanlab_run_name,
            "num_experts": args.num_experts,
            "total_assignments": total,
            "expert_shares": shares,
            "per_layer": {str(l): dict(c) for l, c in per_layer.items()},
        }, ensure_ascii=False, indent=2))
        print(f"[smoke] dumped shares -> {args.dump_json}")

    # Before/after comparison bar chart, all in one swanlab Image.
    if args.compare_to is not None and args.compare_to.exists():
        try:
            _log_compare_figure(args.compare_to, shares, args, wandb)
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] compare figure skipped: {exc!r}")

    if activated_experts < max(2, args.num_experts // 2):
        print(f"[warn] only {activated_experts}/{args.num_experts} experts saw any tokens — "
              "router may be collapsed; revisit mergekit positive_prompts.")
    else:
        print("[ok] router exercising multiple experts; ready for continued training.")
    if wandb is not None:
        wandb.finish()


def _log_compare_figure(before_path: Path, after_shares: list, args, wandb) -> None:
    """Render a side-by-side bar chart from the before/after shares + push to swanlab."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import swanlab

    before = json.loads(before_path.read_text())
    bshares = before["expert_shares"]
    n = max(len(bshares), len(after_shares), args.num_experts)
    bshares = (bshares + [0.0] * n)[:n]
    ashares = (list(after_shares) + [0.0] * n)[:n]
    x = np.arange(n)
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
    ax.bar(x - w / 2, bshares, width=w, label="before (post-upcycle)", color="#888")
    ax.bar(x + w / 2, ashares, width=w, label="after (domain SFT)", color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_xlabel("expert id")
    ax.set_ylabel("activation share")
    ax.set_title("Stage 3 — expert activation share, upcycle vs after domain SFT")
    ax.legend()
    fig.tight_layout()

    out_png = Path("outputs/stage3_smoke_compare.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"[smoke] compare figure -> {out_png}")
    try:
        swanlab.log({"moe/smoke_compare": swanlab.Image(fig)}, step=0)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] swanlab.Image compare failed: {exc!r}")
    plt.close(fig)


if __name__ == "__main__":
    main()
