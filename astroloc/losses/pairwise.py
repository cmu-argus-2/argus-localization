"""L_pairs: attraction/repulsion between a query (astronaut photo) embedding
and its matched satellite-tile embedding, everything else in the batch as an
in-batch negative. See astroloc/losses/multi_similarity.py for the shared
loss primitive this builds on.
"""

import torch

from astroloc.losses.multi_similarity import multi_similarity_loss


def pairwise_loss(
    query_embeddings: torch.Tensor, tile_embeddings: torch.Tensor, alpha: float = 1.0, beta: float = 50.0
) -> torch.Tensor:
    """query_embeddings[i] and tile_embeddings[i] are a matched positive pair."""
    n = query_embeddings.shape[0]
    combined = torch.cat([query_embeddings, tile_embeddings], dim=0)
    pair_ids = torch.arange(n, device=query_embeddings.device)
    labels = torch.cat([pair_ids, pair_ids], dim=0)
    return multi_similarity_loss(combined, labels, alpha=alpha, beta=beta)
