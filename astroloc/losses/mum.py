"""L_MUM: multi-similarity loss over k-means-clustered satellite tiles, where
a query's cluster label is its matched tile's cluster (from astroloc/training/
cluster.py), so same-cluster items in a batch are treated as positives and
different-cluster items as negatives. See astroloc/losses/multi_similarity.py.
"""

import torch

from astroloc.losses.multi_similarity import multi_similarity_loss


def mum_loss(
    query_embeddings: torch.Tensor,
    tile_embeddings: torch.Tensor,
    cluster_ids: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
) -> torch.Tensor:
    """cluster_ids[i] is the shared cluster id of pair i (query i and tile i
    both get this same label, since a query's cluster is defined as its
    matched tile's cluster).
    """
    combined = torch.cat([query_embeddings, tile_embeddings], dim=0)
    labels = torch.cat([cluster_ids, cluster_ids], dim=0)
    return multi_similarity_loss(combined, labels, alpha=alpha, beta=beta)
