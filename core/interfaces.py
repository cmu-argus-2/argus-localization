"""Swappable-stage protocols.

Retriever, DescriptorIndex, and Matcher are typing.Protocol so any concrete
implementation (released EarthLoc model, distilled small model, faiss index,
sift-lightglue matcher, ...) can stand in without touching the pipeline.
See docs/argus_localization_spec.md section 2 and 4.
"""

from typing import Optional, Protocol

import numpy as np

from core.types import MatchResult


class Retriever(Protocol):
    descriptor_dim: int

    def embed(self, image: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 -> (D,) f32, L2-normed."""
        ...

    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """List of (H,W,3) uint8 -> (N,D) f32, L2-normed."""
        ...


class DescriptorIndex(Protocol):
    def add(self, tile_ids: list[str], descriptors: np.ndarray) -> None: ...

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Returns (tile_id, similarity), best first."""
        ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


class Matcher(Protocol):
    def match(
        self,
        query_image: np.ndarray,
        tile_image: np.ndarray,
        tile_id: Optional[str] = None,
    ) -> MatchResult:
        """tile_id lets an implementation use precomputed database-side keypoints."""
        ...
