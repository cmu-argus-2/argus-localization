"""Reference tile database: embeds tiles, populates the index, and answers
retrieval queries. See docs/argus_localization_spec.md section 4.

Database tiles are embedded at 4 rotations (0/90/180/270) by default, not
just north-up. Astronaut/camera frames arrive at an arbitrary roll, and a
single north-up descriptor per tile makes rotated frames retrieve poorly:
diagnosed on Alps, 78% of no-fix frames (25/32) never had their true-match
tile appear anywhere in the top-15 shortlist at all, vs. only 22% that
reached the matcher and fell short on inliers. Rotation TTA is what
EarthLoc's own eval code does (see third_party/EarthLoc/test.py) and was
deliberately skipped in the initial Phase 0 build for baseline plumbing
only; this closes that gap. See docs/argus_localization_spec.md section 4.
"""

import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.interfaces import DescriptorIndex, Retriever
from core.types import GeoTile

DEFAULT_ROTATIONS = (0, 90, 180, 270)
_ROTATION_SEP = "::rot"


def _rotation_id(tile_id: str, degrees: int) -> str:
    return f"{tile_id}{_ROTATION_SEP}{degrees}"


def _base_tile_id(rotation_id: str) -> str:
    # Tolerates ids from caches built before rotation TTA existed (no suffix).
    return rotation_id.split(_ROTATION_SEP, 1)[0]


def dedup_search(index: DescriptorIndex, descriptor: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Top-k unique base tile_ids by best similarity across their rotation variants.

    Over-fetches from the index and dedupes rotation ids back to their base
    tile_id (keeping each tile's best-scoring rotation), doubling the fetch
    size until k unique tiles are found or the index is exhausted. This
    doesn't need to know how many rotations a tile was embedded at, so it
    also works unchanged against pre-rotation-TTA caches (no suffix -> no-op
    dedup). Shared by ReferenceDatabase.retrieve() and scripts/evaluate.py,
    which both search the same index but the latter already has descriptors
    on hand from a batched embed_batch call.
    """
    fetch_k = k * len(DEFAULT_ROTATIONS)
    best_per_tile: dict[str, float] = {}
    while True:
        hits = index.search(descriptor, fetch_k)
        best_per_tile = {}
        for rotation_id, similarity in hits:
            tile_id = _base_tile_id(rotation_id)
            if tile_id not in best_per_tile or similarity > best_per_tile[tile_id]:
                best_per_tile[tile_id] = similarity
        if len(best_per_tile) >= k or len(hits) < fetch_k:
            break
        fetch_k *= 2
    return sorted(best_per_tile.items(), key=lambda kv: -kv[1])[:k]


class ReferenceDatabase:
    def __init__(self, retriever: Retriever, index: DescriptorIndex):
        self.retriever = retriever
        self.index = index
        self.tiles: dict[str, GeoTile] = {}

    def build(
        self, tiles: list[GeoTile], batch_size: int = 64, rotations: tuple[int, ...] = DEFAULT_ROTATIONS
    ) -> None:
        self.tiles = {tile.tile_id: tile for tile in tiles}
        for start in tqdm(range(0, len(tiles), batch_size), desc="Embedding reference tiles"):
            batch = tiles[start : start + batch_size]
            images = [np.array(Image.open(tile.image_path).convert("RGB")) for tile in batch]
            rot_ids = [
                _rotation_id(tile.tile_id, degrees) for tile in batch for degrees in rotations
            ]
            rot_images = [np.rot90(image, k=degrees // 90) for image in images for degrees in rotations]
            descriptors = self.retriever.embed_batch(rot_images)
            self.index.add(rot_ids, descriptors)

    def get_tile(self, tile_id: str) -> GeoTile:
        return self.tiles[tile_id]

    def retrieve(self, frame: np.ndarray, k: int) -> list[tuple[GeoTile, float]]:
        descriptor = self.retriever.embed(frame)
        ranked = dedup_search(self.index, descriptor, k)
        return [(self.tiles[tile_id], similarity) for tile_id, similarity in ranked]

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.index.save(os.path.join(dir_path, "index"))
        tiles_payload = {
            tile_id: {
                "tile_id": tile.tile_id,
                "image_path": tile.image_path,
                "corners_latlon": tile.corners_latlon.tolist(),
                "timestamp": tile.timestamp,
                "meta": tile.meta,
            }
            for tile_id, tile in self.tiles.items()
        }
        with open(os.path.join(dir_path, "tiles.json"), "w") as f:
            json.dump(tiles_payload, f)

    @classmethod
    def load(cls, dir_path: str, retriever: Retriever, index: DescriptorIndex) -> "ReferenceDatabase":
        db = cls(retriever, index)
        db.index.load(os.path.join(dir_path, "index"))
        with open(os.path.join(dir_path, "tiles.json")) as f:
            tiles_payload = json.load(f)
        db.tiles = {
            tile_id: GeoTile(
                tile_id=payload["tile_id"],
                image_path=payload["image_path"],
                corners_latlon=np.array(payload["corners_latlon"], dtype=np.float64),
                timestamp=payload["timestamp"],
                meta=payload["meta"],
            )
            for tile_id, payload in tiles_payload.items()
        }
        return db
