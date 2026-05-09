"""Domain-aware auxiliary loss for MoE training.

Rationale (from project goal): on a legal-only fine-tune, criminal tokens cluster
on experts {17, 42, 89} but the standard load-balancing aux loss fights against
that — it tries to spread tokens uniformly across all 128 experts. We replace
that with a *domain-conditional* aux loss: within each domain we still penalize
imbalance (so we don't collapse to a single expert), but across domains we
*reward* divergence (so different domains learn distinct experts).

Hook this loss in alongside the standard cross-entropy during continued
pre-training of Qwen3-30B-A3B on domain-tagged legal corpora.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def domain_aware_aux_loss(
    router_logits_per_layer: list[torch.Tensor],
    domain_ids: torch.Tensor,
    num_experts: int,
    num_domains: int,
    top_k: int = 8,
    intra_weight: float = 1.0,
    inter_weight: float = 0.5,
) -> torch.Tensor:
    """
    Args:
        router_logits_per_layer: list of (batch, seq, num_experts) tensors, one per MoE layer
        domain_ids:              (batch,) int tensor in [0, num_domains)
        num_experts / num_domains
        top_k:                   active experts per token (Qwen3-30B-A3B uses 8)
        intra_weight:            penalty on within-domain expert imbalance
        inter_weight:            reward (subtracted) for cross-domain expert divergence

    Returns:
        Scalar loss to add to the language-modeling objective.
    """
    total = router_logits_per_layer[0].new_zeros(())
    for router_logits in router_logits_per_layer:
        # routing weights (softmax over experts) - (batch, seq, num_experts)
        routing = F.softmax(router_logits, dim=-1)
        # mean activation per (domain, expert)
        domain_expert = router_logits.new_zeros(num_domains, num_experts)
        domain_token_counts = router_logits.new_zeros(num_domains)
        for d in range(num_domains):
            mask = (domain_ids == d).view(-1, 1, 1)
            if not mask.any():
                continue
            picked = routing * mask
            domain_expert[d] = picked.sum(dim=(0, 1))
            domain_token_counts[d] = mask.sum().clamp(min=1)
        domain_expert = domain_expert / domain_token_counts.unsqueeze(-1)

        # Intra-domain: KL(domain_expert || uniform) keeps load balanced WITHIN a domain.
        uniform = router_logits.new_full((num_experts,), 1.0 / num_experts)
        intra = sum(
            F.kl_div((domain_expert[d] + 1e-9).log(), uniform, reduction="sum")
            for d in range(num_domains)
            if domain_token_counts[d] > 0
        )

        # Inter-domain: average pairwise JS divergence — *reward* divergence, so subtract.
        inter = router_logits.new_zeros(())
        pairs = 0
        for d1 in range(num_domains):
            if domain_token_counts[d1] == 0:
                continue
            for d2 in range(d1 + 1, num_domains):
                if domain_token_counts[d2] == 0:
                    continue
                p, q = domain_expert[d1], domain_expert[d2]
                m = 0.5 * (p + q)
                js = 0.5 * (
                    F.kl_div((p + 1e-9).log(), m, reduction="sum")
                    + F.kl_div((q + 1e-9).log(), m, reduction="sum")
                )
                inter = inter + js
                pairs += 1
        if pairs > 0:
            inter = inter / pairs

        total = total + intra_weight * intra - inter_weight * inter

    return total / max(len(router_logits_per_layer), 1)


__all__ = ["domain_aware_aux_loss"]
