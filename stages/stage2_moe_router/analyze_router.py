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
    # mean_acts[layer_idx][domain] = 128-dim tensor of per-expert activation share.
    # Kept around so we can render the (domain × expert) heat-map per layer
    # without recomputing — and also dumped into the JSON for downstream
    # plotting (e.g. replay_stage2_swanlab.py).
    mean_acts: dict[int, dict[str, "torch.Tensor"]] = defaultdict(dict)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # First pass: compute every (domain, layer) record into the JSON report,
    # but DON'T log to swanlab yet — we need to batch per-layer to get one
    # multi-line chart per metric instead of N×D single-point charts.
    for domain, layers in activations.items():
        report[domain] = {}
        for layer_idx, acts in sorted(layers.items()):
            mean_act = torch.stack(acts).mean(0)
            mean_acts[layer_idx][domain] = mean_act
            variance = mean_act.var().item()
            top_experts = mean_act.topk(5).indices.tolist()
            top_freq = mean_act[top_experts[0]].item()
            report[domain][f"L{layer_idx}"] = {
                "variance": variance,
                "top_experts": top_experts,
                "top1_freq": top_freq,
                # Full per-expert activation share (128 floats). Needed for the
                # heat-map; cheap relative to the rest of the JSON.
                "mean_activation": [float(x) for x in mean_act.tolist()],
            }
            print(f"{domain} L{layer_idx} var={variance:.4f} "
                  f"top_experts={top_experts} top1_freq={top_freq:.3f}")

    domains = sorted(activations.keys())
    yticklabels = [_DOMAIN_LATIN.get(d, d) for d in domains]

    # Second pass: one swanlab.log per layer, with all per-domain values in a
    # single payload — produces one multi-line chart per metric, x=layer.
    if wandb is not None:
        layer_ids = sorted({l for d in activations for l in activations[d]})
        for layer_idx in layer_ids:
            payload = {}
            for d in domains:
                rec = report[d].get(f"L{layer_idx}")
                if rec is None:
                    continue
                latin = _DOMAIN_LATIN.get(d, d)
                payload[f"moe/expert_var/{latin}"] = rec["variance"]
                payload[f"moe/top1_freq/{latin}"] = rec["top1_freq"]
            # Cross-domain overlap (top-5 expert set overlap, scalar per layer)
            top5_per_domain = {
                d: set(report[d][f"L{layer_idx}"]["top_experts"]) for d in domains
                if f"L{layer_idx}" in report[d]
            }
            pairs, overlap_sum = 0, 0.0
            for i, d1 in enumerate(domains):
                for d2 in domains[i + 1:]:
                    if d1 in top5_per_domain and d2 in top5_per_domain:
                        overlap_sum += len(top5_per_domain[d1] & top5_per_domain[d2]) / 5
                        pairs += 1
            overlap = overlap_sum / pairs if pairs else 0.0
            payload["moe/cross_domain_overlap"] = overlap
            overlap_records.append({"layer": layer_idx, "overlap": overlap})
            wandb.log(payload, step=layer_idx)

    # Per-domain summary scalars (one chart per metric, one bar per domain).
    if wandb is not None:
        mean_var_payload, sample_count_payload = {}, {}
        for domain, layers in activations.items():
            mean_var = sum(report[domain][f"L{l}"]["variance"] for l in layers) / max(len(layers), 1)
            latin = _DOMAIN_LATIN.get(domain, domain)
            mean_var_payload[f"moe/mean_variance/{latin}"] = mean_var
            sample_count_payload[f"moe/sample_count/{latin}"] = domain_counts[domain]
        if mean_var_payload:
            wandb.log(mean_var_payload, step=0)
        if sample_count_payload:
            wandb.log(sample_count_payload, step=0)

    # Heat-map (domain × expert) per layer. Renders matplotlib figures and
    # logs them as swanlab Images so the resume screenshot is one click away.
    if wandb is not None and mean_acts:
        try:
            _log_heatmaps(mean_acts, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[analyze_router] heat-map logging skipped: {exc!r}")

    args.output_json.write_text(json.dumps({
        "report": report,
        "cross_domain_overlap": overlap_records,
        "domain_counts": dict(domain_counts),
    }, ensure_ascii=False, indent=2))
    print(f"[done] wrote {args.output_json}")
    if wandb is not None:
        wandb.finish()


_DOMAIN_LATIN = {
    "刑事": "criminal",
    "民事": "civil",
    "商事": "commercial",
    "行政": "administrative",
    "知产": "IP",
}


def _log_heatmaps(mean_acts, args) -> None:
    """Render one (domain × expert) heat-map per layer and push to swanlab.

    Domain labels are rendered in latin (criminal / civil / …) because
    the box has no CJK fonts installed; matplotlib falls back to tofu
    boxes otherwise. The underlying data is unchanged.

    A few hand-picked deep layers are saved as PNGs under outputs/heatmaps/
    so the resume screenshot can come straight from disk if swanlab is offline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import swanlab

    heatmap_dir = args.output_json.parent / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    domains = sorted({d for layer in mean_acts.values() for d in layer})
    yticklabels = [_DOMAIN_LATIN.get(d, d) for d in domains]
    sorted_layers = sorted(mean_acts.keys())
    # Highlight the last quarter of layers (deepest, where domain clustering
    # is typically strongest) plus a couple of shallow ones for contrast.
    deepest = sorted_layers[-min(4, len(sorted_layers)):]
    sample_shallow = sorted_layers[: min(2, len(sorted_layers))]
    highlight_layers = sorted(set(deepest + sample_shallow))

    for layer_idx in sorted_layers:
        rows = []
        for d in domains:
            t = mean_acts[layer_idx].get(d)
            rows.append(t.cpu().numpy() if t is not None else np.zeros(args.num_experts))
        mat = np.stack(rows, axis=0)  # (n_domains, num_experts)

        fig, ax = plt.subplots(figsize=(max(8, args.num_experts / 16), 0.6 * len(domains) + 1.2))
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("expert id")
        ax.set_title(f"Layer {layer_idx} — activation share by domain (top-{args.top_k} per token)")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()

        if layer_idx in highlight_layers:
            png = heatmap_dir / f"L{layer_idx:02d}.png"
            fig.savefig(png, dpi=140)
        try:
            # One key with N steps so swanlab renders this as a single image
            # panel with a layer-index slider, not N panels of one image each.
            swanlab.log({"moe/heatmap": swanlab.Image(fig)}, step=layer_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"[analyze_router] swanlab.Image(layer={layer_idx}) failed: {exc!r}")
        plt.close(fig)
    print(f"[analyze_router] heat-maps saved to {heatmap_dir} (highlights: {highlight_layers})")


if __name__ == "__main__":
    main()
