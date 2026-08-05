"""Multi-Similarity loss (Wang et al. 2019, "Multi-Similarity Loss for Deep
Metric Learning"), the shared primitive behind both of AstroLoc's losses (see
astroloc/losses/pairwise.py and astroloc/losses/mum.py). Self-contained
(no pytorch_metric_learning dependency) since only this one loss is needed.

alpha=1, beta=50 are AstroLoc's own reported hyperparameters for both
L_pairs and L_MUM (arXiv:2502.07003).
"""

import torch


def multi_similarity_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    base: float = 0.5,
    mining_margin: float = 0.1,
) -> torch.Tensor:
    """embeddings: (N,D) L2-normalized. labels: (N,) int, same label = positive pair.

    For each anchor: positives are hard-mined to similarities below the
    (hardest negative + margin) and vice versa for negatives, then the
    standard MS-loss log-sum-exp is applied to the surviving pairs. Anchors
    with no positives or no negatives contribute zero (this happens if a
    label is unique in the batch, or if every item shares one label).
    """
    sim = embeddings @ embeddings.t()  # (N,N) cosine similarity (inputs are L2-normed)
    n = sim.shape[0]
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    pos_mask = same_label & ~self_mask
    neg_mask = ~same_label

    neg_inf = torch.finfo(sim.dtype).min
    pos_sim_for_min = sim.masked_fill(~pos_mask, float("inf"))
    neg_sim_for_max = sim.masked_fill(~neg_mask, neg_inf)
    hardest_neg = neg_sim_for_max.max(dim=1, keepdim=True).values  # (N,1)
    hardest_pos = pos_sim_for_min.min(dim=1, keepdim=True).values  # (N,1)

    mined_pos = pos_mask & (sim < hardest_neg + mining_margin)
    mined_neg = neg_mask & (sim > hardest_pos - mining_margin)

    has_pos = mined_pos.any(dim=1)
    has_neg = mined_neg.any(dim=1)
    valid = has_pos & has_neg
    if not valid.any():
        return embeddings.new_tensor(0.0, requires_grad=True)

    pos_term = torch.where(mined_pos, torch.exp(-alpha * (sim - base)), torch.zeros_like(sim))
    neg_term = torch.where(mined_neg, torch.exp(beta * (sim - base)), torch.zeros_like(sim))

    pos_loss = torch.log1p(pos_term.sum(dim=1)) / alpha
    neg_loss = torch.log1p(neg_term.sum(dim=1)) / beta

    return (pos_loss[valid] + neg_loss[valid]).mean()
