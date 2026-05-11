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

    # One chart per metric × domain (each domain = one line), x-axis = layer.
    # Use latin domain names as part of the key so the dashboard legend reads
    # cleanly even on font-tofu setups.
    domains = sorted(report.keys())
    layer_keys = sorted(
        {k for layers in report.values() for k in layers},
        key=lambda k: int(k[1:]),
    )
    for lk in layer_keys:
        layer_idx = int(lk[1:])
        payload = {}
        for d in domains:
            v = report[d].get(lk)
            if v is None:
                continue
            latin = _DOMAIN_LATIN.get(d, d)
            payload[f"moe/expert_var/{latin}"] = v["variance"]
            payload[f"moe/top1_freq/{latin}"] = v["top1_freq"]
        if payload:
            wandb.log(payload, step=layer_idx)

    # Cross-domain overlap: one line, x = layer.
    for rec in overlap_records:
        wandb.log({"moe/cross_domain_overlap": rec["overlap"]},
                  step=int(rec["layer"]))

    # Per-domain summary scalars (latin keys for clean legend).
    mean_var_payload = {}
    sample_count_payload = {}
    for domain, layers in report.items():
        mean_var = sum(v["variance"] for v in layers.values()) / max(len(layers), 1)
        latin = _DOMAIN_LATIN.get(domain, domain)
        mean_var_payload[f"moe/mean_variance/{latin}"] = mean_var
        sample_count_payload[f"moe/sample_count/{latin}"] = domain_counts.get(domain, 0)
    # Single step for the scalars so they're one bar / point per chart.
    if mean_var_payload:
        wandb.log(mean_var_payload, step=0)
    if sample_count_payload:
        wandb.log(sample_count_payload, step=0)

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


_DOMAIN_LATIN = {
    "刑事": "criminal",
    "民事": "civil",
    "商事": "commercial",
    "行政": "administrative",
    "知产": "IP",
}


def _log_heatmaps_from_json(report) -> None:
    """Render heat-maps from the JSON's mean_activation tensors. Domain labels
    are rendered in latin to avoid CJK-tofu when no Chinese font is installed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import swanlab

    domains = sorted(report.keys())
    yticklabels = [_DOMAIN_LATIN.get(d, d) for d in domains]
    layer_keys = sorted(
        {k for layers in report.values() for k in layers},
        key=lambda k: int(k[1:]),
    )
    # ONE key `moe/heatmap` with N steps (one image per layer). swanlab
    # renders this as a slider you can scrub through layers.
    for lk in layer_keys:
        layer_idx = int(lk[1:])
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
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("expert id")
        ax.set_title(f"Layer {layer_idx} — activation share by domain")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        try:
            swanlab.log({"moe/heatmap": swanlab.Image(fig)}, step=layer_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"[replay] swanlab.Image(layer={layer_idx}) failed: {exc!r}")
        plt.close(fig)


if __name__ == "__main__":
    main()
