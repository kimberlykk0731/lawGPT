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

    # Heat-maps if the JSON has `mean_activation` (new schema). Old JSONs
    # only store top-5 indices, no full distribution — re-run analyze_router
    # to get them.
    has_mean_act = any(
        "mean_activation" in v for layers in report.values() for v in layers.values()
    )
    if has_mean_act:
        _log_heatmaps_from_json(report)
    else:
        print("[replay] JSON has no `mean_activation` — heat-maps skipped "
              "(re-run analyze_router.py with the new code to capture them)")

    wandb.finish()
    print(f"[replay] backfilled {len(report)} domains × {sum(len(l) for l in report.values())} layer scalars to swanlab")


def _log_heatmaps_from_json(report) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import swanlab

    domains = sorted(report.keys())
    layer_keys = sorted(
        {k for layers in report.values() for k in layers},
        key=lambda k: int(k[1:]),
    )
    for lk in layer_keys:
        rows = []
        for d in domains:
            v = report[d].get(lk, {}).get("mean_activation")
            if v is None:
                v = np.zeros(128)
            rows.append(np.asarray(v, dtype=np.float32))
        mat = np.stack(rows, axis=0)

        fig, ax = plt.subplots(figsize=(max(8, mat.shape[1] / 16), 0.6 * len(domains) + 1.2))
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains)
        ax.set_xlabel("expert id")
        ax.set_title(f"Layer {lk[1:]} — activation share by domain")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        try:
            swanlab.log({f"moe/heatmap/{lk}": swanlab.Image(fig)})
        except Exception as exc:  # noqa: BLE001
            print(f"[replay] swanlab.Image({lk}) failed: {exc!r}")
        plt.close(fig)


if __name__ == "__main__":
    main()
