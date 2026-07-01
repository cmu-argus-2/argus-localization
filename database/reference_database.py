"""Reference tile database: embeds tiles, populates the index, and answers
retrieval queries. See docs/argus_localization_spec.md section 4.
"""

import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.interfaces import DescriptorIndex, Retriever
from core.types import GeoTile


class ReferenceDatabase:
    def __init__(self, retriever: Retriever, index: DescriptorIndex):
        self.retriever = retriever
        self.index = index
        self.tiles: dict[str, GeoTile] = {}

    def build(self, tiles: list[GeoTile], batch_size: int = 64) -> None:
        self.tiles = {tile.tile_id: tile for tile in tiles}
        for start in tqdm(range(0, len(tiles), batch_size), desc="Embedding reference tiles"):
            batch = tiles[start : start + batch_size]
            images = [np.array(Image.open(tile.image_path).convert("RGB")) for tile in batch]
            descriptors = self.retriever.embed_batch(images)
            self.index.add([tile.tile_id for tile in batch], descriptors)

    def get_tile(self, tile_id: str) -> GeoTile:
        return self.tiles[tile_id]

    def retrieve(self, frame: np.ndarray, k: int) -> list[tuple[GeoTile, float]]:
        descriptor = self.retriever.embed(frame)
        hits = self.index.search(descriptor, k)
        return [(self.tiles[tile_id], similarity) for tile_id, similarity in hits]

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
