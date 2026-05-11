"""Backfill the Stage-2 swanlab dashboard from outputs/router_variance.json.

Stage 2's analyze_router.py logs each (layer, domain) metric to swanlab
during the forward pass. If that run ran on a swanlab version that lacked
the wandb integration shim (silently skipped logging), the JSON output is
still complete — replay it here to get the dashboard without re-running
the 30B-A3B forward.

Usage:
    python scripts/replay_stage2_swanlab.py [--json outputs/router_variance.json]
                                            [--run_name stage2-analyze-v1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stages._swanlab_shim import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=Path("outputs/router_variance.json"))
    parser.add_argument("--project", default="legalgpt-2026")
    parser.add_argument("--run_name", default="stage2-analyze-v1")
    args = parser.parse_args()

    data = json.loads(args.json.read_text())
    report = data["report"]
    overlap_records = data.get("cross_domain_overlap", [])
    domain_counts = data.get("domain_counts", {})

    wandb.init(
        project=args.project,
        name=args.run_name,
        tags=["stage2", "analyze", "router", "replay"],
        config={"source": str(args.json), "replayed": True},
    )

    # Per-(domain, layer) variance and top-1 expert frequency.
    # swanlab needs a monotonic step; we assign each layer-index as the step
    # so the dashboard renders the cross-layer trend cleanly.
    for domain, layers in report.items():
        for layer_key, vals in sorted(layers.items(), key=lambda kv: int(kv[0][1:])):
            layer_idx = int(layer_key[1:])
            wandb.log({
                f"moe/expert_var/L{layer_idx}/{domain}": vals["variance"],
                f"moe/top1_freq/L{layer_idx}/{domain}": vals["top1_freq"],
            }, step=layer_idx)

    # Cross-domain overlap per layer.
    for rec in overlap_records:
        wandb.log({f"moe/cross_domain_overlap/L{rec['layer']}": rec["overlap"]},
                  step=int(rec["layer"]))

    # Per-domain summary scalars.
    for domain, layers in report.items():
        mean_var = sum(v["variance"] for v in layers.values()) / max(len(layers), 1)
        wandb.log({
            f"moe/mean_variance/{domain}": mean_var,
            f"moe/sample_count/{domain}": domain_counts.get(domain, 0),
        })

    wandb.finish()
    print(f"[replay] backfilled {len(report)} domains × {sum(len(l) for l in report.values())} layer scalars to swanlab")


if __name__ == "__main__":
    main()
