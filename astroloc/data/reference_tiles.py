"""Reference (satellite) tile set for AstroLoc-style training.

Reuses the existing rsynced EarthLoc database mirror at
user_config.yaml::database_dir (real Sentinel-2-derived cloud-free composite
tiles, already on disk, no new downloads needed -- see repo memory
astroloc_target_architecture for why this replaces AstroLoc's own from-scratch
EOxCloudless pull). Scoped to astroloc/data/regions.py::TRAIN_REGIONS (not the
6 held-out eval regions) at TRAIN_DB_RADIUS_KM, single year TRAIN_DB_YEAR,
then subsampled per region since each region alone has ~50-60k tiles at that
radius (confirmed by direct measurement) -- more than needed for a demo-scale
reference/cluster set and too many to embed with the heavier DINOv2 backbone
in a reasonable time budget.
"""

import random

from core.types import GeoTile
from scripts.evaluate import load_scoped_db_tiles

from astroloc.data.regions import TRAIN_DB_RADIUS_KM, TRAIN_DB_YEAR, TRAIN_REGIONS


def build_reference_tiles(
    database_dir: str, per_region_cap: int = 10000, seed: int = 0
) -> list[GeoTile]:
    rng = random.Random(seed)
    by_id: dict[str, GeoTile] = {}
    for name, (clat, clon) in TRAIN_REGIONS.items():
        tiles = load_scoped_db_tiles(database_dir, clat, clon, TRAIN_DB_YEAR, TRAIN_DB_RADIUS_KM)
        rng.shuffle(tiles)
        chosen = tiles[:per_region_cap]
        print(f"  {name}: {len(chosen)}/{len(tiles)} reference tiles selected")
        for t in chosen:
            by_id[t.tile_id] = t  # dedup tiles shared across nearby region scopes
    combined = list(by_id.values())
    print(f"Combined reference tile set: {len(combined)} unique tiles")
    return combined
