"""Flat cosine-similarity index over descriptor vectors.

IndexFlatIP on L2-normalized descriptors gives exact cosine similarity search.
Fine for the current database size; swap to IVF-PQ only if the database grows
large enough that RAM becomes the limit (see docs/argus_localization_design.md,
Nearest-neighbor search and database section).
"""

import json
import os

import faiss
import numpy as np

from core.interfaces import DescriptorIndex


class FaissFlatIndex(DescriptorIndex):
    def __init__(self, descriptor_dim: int):
        self.descriptor_dim = descriptor_dim
        self.tile_ids: list[str] = []
        self._index = faiss.IndexFlatIP(descriptor_dim)

    def add(self, tile_ids: list[str], descriptors: np.ndarray) -> None:
        descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        if descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"expected descriptors of dim {self.descriptor_dim}, got {descriptors.shape[1]}"
            )
        self._index.add(descriptors)
        self.tile_ids.extend(tile_ids)

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1)
        k = min(k, self._index.ntotal)
        similarities, indices = self._index.search(q, k)
        return [
            (self.tile_ids[idx], float(sim))
            for idx, sim in zip(indices[0], similarities[0])
            if idx != -1
        ]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self._index, f"{path}.faiss")
        with open(f"{path}.tile_ids.json", "w") as f:
            json.dump(self.tile_ids, f)

    def load(self, path: str) -> None:
        self._index = faiss.read_index(f"{path}.faiss")
        self.descriptor_dim = self._index.d
        with open(f"{path}.tile_ids.json") as f:
            self.tile_ids = json.load(f)
