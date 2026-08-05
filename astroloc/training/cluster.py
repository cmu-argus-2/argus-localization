"""K-means clustering of reference-tile embeddings for the MuM loss.

Per the AstroLoc paper: satellite images are clustered (k=50), queries are
assigned to clusters (here: via their matched tile's cluster, since we
already have IoU-based positive pairs -- see astroloc/data/pairing.py -- so
there's no need for the paper's own embedding-similarity assignment), and
clusters are sampled during training weighted by how many queries land in
each one.
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
