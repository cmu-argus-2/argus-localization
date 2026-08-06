"""L_MUM: multi-similarity loss over k-means-clustered satellite tiles.
Tiles define the cluster space (k-means over tile embeddings); a query's
cluster label comes from nearest-centroid assignment of its own embedding
(astroloc/training/cluster.py::recluster), which can differ from its matched
tile's cluster. Same-cluster items in a batch are positives, different-
cluster items are negatives. See astroloc/losses/multi_similarity.py.
"""

import torch

from astroloc.losses.multi_similarity import multi_similarity_loss


def mum_loss(
    query_embeddings: torch.Tensor,
    tile_embeddings: torch.Tensor,
    query_cluster_ids: torch.Tensor,
    tile_cluster_ids: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
) -> torch.Tensor:
    combined = torch.cat([query_embeddings, tile_embeddings], dim=0)
    labels = torch.cat([query_cluster_ids, tile_cluster_ids], dim=0)
    return multi_similarity_loss(combined, labels, alpha=alpha, beta=beta)
