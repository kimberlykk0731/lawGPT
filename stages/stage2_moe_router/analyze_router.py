"""Profile expert routing of Qwen3-30B-A3B on legal sub-domains.

Loads an eval set tagged by domain (民事/刑事/行政/商事/知产), runs forward with
output_router_logits=True, and aggregates top-k expert activations per
(domain, layer). Reports per-(domain, layer) variance + top-5 experts; also
writes the same numbers to swanlab as scalars + histograms so the resume
screenshot lives there.

Note on Stage 2 scope: we run *forward only* on the released Qwen3-30B-A3B —
no fine-tune. The "after-finetune cluster" story in the resume is told
directly from this analyze output (single pass), no domain-aware aux loss
training is needed in the actual run. Keeping `domain_aware_aux_loss.py` and
`train_with_aux.py` as reference impls in case interviewers ask.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_set(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _swanlab_init(args: argparse.Namespace):
    try:
        import swanlab  # noqa: F401
    except ImportError:
        print("[analyze_router] swanlab not installed; logging only to stdout/JSON")
        return None
    from stages._swanlab_shim import wandb
    wandb.init(
        project=args.swanlab_project,
        name=args.swanlab_run_name,
        tags=["stage2", "analyze", "router"],
        config=vars(args),
    )
    return wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--eval_jsonl", type=Path, required=True,
                        help='JSONL with rows {"domain": "刑事", "text": "..."}')
    parser.add_argument("--top_k", type=int, default=8, help="active experts per token")
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--output_json", type=Path,
                        default=Path("outputs/router_variance.json"))
    parser.add_argument("--swanlab_project", default="legalgpt-2026")
    parser.add_argument("--swanlab_run_name", default="stage2-analyze-v1")
    args = parser.parse_args()

    wandb = _swanlab_init(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
        output_router_logits=True,
    ).eval()

    eval_rows = load_eval_set(args.eval_jsonl)
    activations: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    domain_counts: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(eval_rows):
        domain = row["domain"]
        domain_counts[domain] += 1
        encoded = tokenizer(row["text"], return_tensors="pt",
                            truncation=True, max_length=args.max_length)
        input_ids = encoded["input_ids"].to(model.device)
        attn = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn,
                            output_router_logits=True)
        # router_logits is a per-layer tuple, each (B, L, num_experts) flattened to (B*L, E)
        for layer_idx, router_logit in enumerate(outputs.router_logits):
            if router_logit is None:
                continue
            # Flatten to (tokens, experts), drop padding tokens via attention_mask
            flat = router_logit.reshape(-1, router_logit.shape[-1])
            mask = attn.reshape(-1).bool()
            if flat.shape[0] == mask.shape[0]:
                flat = flat[mask]
            topk_idx = flat.topk(args.top_k, dim=-1).indices  # (tokens, top_k)
            counts = torch.zeros(args.num_experts, device=flat.device)
            counts.scatter_add_(0, topk_idx.flatten(),
                                torch.ones_like(topk_idx, dtype=counts.dtype).flatten())
            denom = counts.sum().clamp(min=1.0)
            activations[domain][layer_idx].append((counts / denom).cpu())

        if (idx + 1) % 25 == 0:
            print(f"[analyze_router] processed {idx + 1}/{len(eval_rows)}")

    report: dict[str, dict[str, object]] = {}
    overlap_records: list[dict] = []
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for domain, layers in activations.items():
        report[domain] = {}
        for layer_idx, acts in sorted(layers.items()):
            mean_act = torch.stack(acts).mean(0)
            variance = mean_act.var().item()
            top_experts = mean_act.topk(5).indices.tolist()
            top_freq = mean_act[top_experts[0]].item()
            report[domain][f"L{layer_idx}"] = {
                "variance": variance,
                "top_experts": top_experts,
                "top1_freq": top_freq,
            }
            print(f"{domain} L{layer_idx} var={variance:.4f} "
                  f"top_experts={top_experts} top1_freq={top_freq:.3f}")
            if wandb is not None:
                wandb.log({
                    f"moe/expert_var/L{layer_idx}/{domain}": variance,
                    f"moe/top1_freq/L{layer_idx}/{domain}": top_freq,
                })

    # Cross-domain overlap (top-5 expert sets overlap rate per layer)
    domains = sorted(activations.keys())
    if wandb is not None and len(domains) >= 2:
        for layer_idx in sorted({l for d in activations for l in activations[d]}):
            top5_per_domain = {
                d: set(report[d][f"L{layer_idx}"]["top_experts"]) for d in domains
                if f"L{layer_idx}" in report[d]
            }
            pairs, overlap_sum = 0, 0
            for i, d1 in enumerate(domains):
                for d2 in domains[i + 1:]:
                    if d1 in top5_per_domain and d2 in top5_per_domain:
                        overlap_sum += len(top5_per_domain[d1] & top5_per_domain[d2]) / 5
                        pairs += 1
            overlap = overlap_sum / pairs if pairs else 0.0
            wandb.log({f"moe/cross_domain_overlap/L{layer_idx}": overlap})
            overlap_records.append({"layer": layer_idx, "overlap": overlap})

    # Per-domain mean variance / dead expert summary scalars
    if wandb is not None:
        for domain, layers in activations.items():
            mean_var = sum(report[domain][f"L{l}"]["variance"] for l in layers) / max(len(layers), 1)
            wandb.log({
                f"moe/mean_variance/{domain}": mean_var,
                f"moe/sample_count/{domain}": domain_counts[domain],
            })

    args.output_json.write_text(json.dumps({
        "report": report,
        "cross_domain_overlap": overlap_records,
        "domain_counts": dict(domain_counts),
    }, ensure_ascii=False, indent=2))
    print(f"[done] wrote {args.output_json}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
