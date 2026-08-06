"""K-means clustering of reference-tile embeddings for the MuM loss.

Per the AstroLoc paper: satellite images are clustered (k=50), queries are
assigned to clusters by nearest centroid, and clusters are sampled during
training weighted by how many queries land in each one
(astroloc/training/sampler.py::ClusterBatchSampler). `recluster()` below
redoes this with the model's current (updating) weights, for periodic
reclustering during training (train.py's --recluster-every-steps); the
one-time version at training start (train.py::build_training_data) instead
takes the shortcut of using each pair's matched tile's cluster directly.
"""

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.types import GeoTile

from astroloc.models.dinov2_salad import DinoV2SaladRetriever


def embed_tiles(
    retriever: DinoV2SaladRetriever, tiles: list[GeoTile], batch_size: int = 128
) -> np.ndarray:
    embeddings = np.empty((len(tiles), retriever.descriptor_dim), dtype=np.float32)
    for start in tqdm(range(0, len(tiles), batch_size), desc="Embedding tiles"):
        batch = tiles[start : start + batch_size]
        images = [np.array(Image.open(t.image_path).convert("RGB")) for t in batch]
        embeddings[start : start + len(batch)] = retriever.embed_batch(images)
    return embeddings


def kmeans_cluster(
    embeddings: np.ndarray, k: int = 50, niter: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centroids (k,D) f32, assignments (N,) int64)."""
    import faiss

    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=niter, seed=seed, verbose=True, gpu=False)
    kmeans.train(embeddings)
    _, assignments = kmeans.index.search(embeddings, 1)
    return kmeans.centroids.reshape(k, d), assignments.reshape(-1)


def assign_nearest_cluster(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment (L2) for embeddings not part of the original fit."""
    import faiss

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    _, assignments = index.search(embeddings, 1)
    return assignments.reshape(-1)


def recluster(
    retriever: DinoV2SaladRetriever, tiles: list[GeoTile], queries: list[GeoTile], k: int = 50
) -> tuple[dict[str, int], dict[str, int]]:
    """Re-embeds tiles and queries with the retriever's CURRENT weights,
    re-fits k-means on the tiles (clusters are of satellite images), and
    assigns queries to their nearest centroid -- the dynamic-batching
    counterpart to the one-time clustering `training/train.py::build_training_data`
    does before training starts. Returns (tile_id -> cluster_id, query_id ->
    cluster_id) dicts. Caller is responsible for retriever.model being in
    eval() mode and switching back to train() after.
    """
    tile_embeddings = embed_tiles(retriever, tiles)
    centroids, tile_cluster_ids = kmeans_cluster(tile_embeddings, k=k)
    tile_id_to_cluster = {t.tile_id: int(c) for t, c in zip(tiles, tile_cluster_ids)}

    query_embeddings = embed_tiles(retriever, queries)
    query_cluster_ids = assign_nearest_cluster(query_embeddings, centroids)
    query_id_to_cluster = {q.tile_id: int(c) for q, c in zip(queries, query_cluster_ids)}

    return tile_id_to_cluster, query_id_to_cluster
