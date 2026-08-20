"""Nano training data: ALL available queries and reference tiles (not scoped
to astroloc/data/regions.py::TRAIN_REGIONS' hand-picked 6-region subset),
restricted to reference-tile zoom level 9 only.

Zoom is a directory-level split in the EarthLoc mirror, not a filename field:
database_dir/{year}_{zoom:02d}/**/*.jpg (scripts/evaluate.py's own
`load_scoped_db_tiles` calls this same directory "month_dir", which is wrong --
verified directly: every tile under database_dir/2021_09/ has image_id prefix
"09_...", database_dir/2021_11/ has "11_...", etc. -- these are zoom dirs).
Zoom 9 alone has 7,602 tiles for all of 2021 (checked directly), vs the
current astroloc/ multi-zoom, rotation-augmented Alps index alone at 862k
vectors -- this is the whole point of the nano variant: a small enough index
to be Jetson/nanosat-plausible.

Still excludes the 6 held-out eval regions (astroloc/data/regions.py::EVAL_REGIONS),
same TRAIN_QUERY_RADIUS_KM convention astroloc/data/regions.py already uses for
TRAIN_REGIONS -- "train on everything" means "everything except what we evaluate
on", not literally including the test regions, or the eval numbers would be
meaningless (train-on-test) and not comparable to the other astroloc/ checkpoints.
"""

import glob
import math
import os

from core.types import GeoTile
from data_loading.earthloc_loader import load_query_set, parse_geotile_filename

from astroloc.data.regions import EVAL_REGIONS, TRAIN_QUERY_RADIUS_KM

ZOOM = "09"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def _near_eval_region(lat: float, lon: float) -> bool:
    return any(
        _haversine_km(lat, lon, clat, clon) < TRAIN_QUERY_RADIUS_KM
        for clat, clon in EVAL_REGIONS.values()
    )


TRAIN_ZOOMS = ("09", "10")


def _zoom_tiles(database_dir: str, year: int, zoom: str) -> list[GeoTile]:
    zoom_dir = os.path.join(database_dir, f"{year}_{zoom}")
    paths = glob.glob(os.path.join(zoom_dir, "**", "*.jpg"), recursive=True)
    return [parse_geotile_filename(p) for p in paths]


def _zoom9_tiles(database_dir: str, year: int) -> list[GeoTile]:
    return _zoom_tiles(database_dir, year, ZOOM)


def build_reference_tiles_zoom9(database_dir: str, year: int = 2021) -> list[GeoTile]:
    """All zoom-9 tiles worldwide (for one year), minus the 6 held-out eval regions.
    Used for training -- eval uses build_eval_reference_tiles_zoom9 instead, which
    scopes to one region at a time (recall@k needs a bounded, comparable db per region,
    same as astroloc/eval/evaluate.py's convention)."""
    tiles = _zoom9_tiles(database_dir, year)
    kept = [t for t in tiles if not _near_eval_region(t.meta["nadir_lat"], t.meta["nadir_lon"])]
    print(f"zoom-{ZOOM} reference tiles: {len(kept)}/{len(tiles)} (excluding held-out eval regions)")
    return kept


def build_reference_tiles_multi_zoom(
    database_dir: str, year: int = 2021, zooms: tuple[str, ...] = TRAIN_ZOOMS
) -> list[GeoTile]:
    """Training-only tile pool spanning multiple zoom levels (default 9+10),
    minus the 6 held-out eval regions. NOT used for eval/deployment -- the
    deployed reference DB stays zoom-9-only (build_eval_reference_tiles_zoom9),
    so RAM/index size at inference is unaffected by this.

    Why multi-zoom for training: IoU-based positive pairing needs the query's
    own footprint area to roughly match the candidate tile's fixed footprint
    area, not just overlap geographically. A zoom-9 tile is ~97,000 sq km; a
    typical EarthLoc query is ~25,000 sq km -- even a query fully contained
    in a zoom-9 tile only reaches IoU~0.256 (25000/97474), a razor-thin margin
    over the 0.2 threshold that any real-world grid misalignment wipes out.
    Confirmed directly: zoom-9-only pairing kept only 4,104/54,389 (7.5%) of
    candidate queries. Zoom-10 tiles are ~4x smaller in area (29,370 tiles/year
    vs zoom-9's 7,602), giving queries with tighter footprints a scale-matched
    candidate too.
    """
    tiles: list[GeoTile] = []
    for zoom in zooms:
        zoom_tiles = _zoom_tiles(database_dir, year, zoom)
        kept = [t for t in zoom_tiles if not _near_eval_region(t.meta["nadir_lat"], t.meta["nadir_lon"])]
        print(f"zoom-{zoom} reference tiles: {len(kept)}/{len(zoom_tiles)} (excluding held-out eval regions)")
        tiles.extend(kept)
    return tiles


def build_eval_reference_tiles_zoom9(
    database_dir: str, center_lat: float, center_lon: float, year: int, dist_km: float
) -> list[GeoTile]:
    """Zoom-9-only reference tiles within dist_km of one eval region center --
    the nano equivalent of scripts/evaluate.py::load_scoped_db_tiles, restricted
    to a single zoom level instead of all of them."""
    tiles = _zoom9_tiles(database_dir, year)
    return [
        t for t in tiles
        if _haversine_km(center_lat, center_lon, t.meta["nadir_lat"], t.meta["nadir_lon"]) < dist_km
    ]


def build_eval_reference_tiles_multi_zoom(
    database_dir: str, center_lat: float, center_lon: float, year: int, dist_km: float,
    zooms: tuple[str, ...] = TRAIN_ZOOMS,
) -> list[GeoTile]:
    """Zoom-9+10 reference tiles within dist_km of one eval region center --
    both zooms in the SAME index, so retrieval can pick whichever scale
    actually matches a given query best (dedup_search already handles a
    mixed-scale index fine, same mechanism as rotation-TTA's multi-variant
    dedup). Raises the deployed index size vs zoom-9-only: zoom-10 has ~3.9x
    more tiles/year than zoom-9 (29,370 vs 7,602), so combining both roughly
    multiplies RAM at inference by ~4-5x over the zoom-9-only numbers already
    measured -- a real, deliberate tradeoff (better scale coverage vs a much
    smaller index), not a bug.
    """
    tiles: list[GeoTile] = []
    for zoom in zooms:
        zoom_tiles = _zoom_tiles(database_dir, year, zoom)
        tiles.extend(
            t for t in zoom_tiles
            if _haversine_km(center_lat, center_lon, t.meta["nadir_lat"], t.meta["nadir_lon"]) < dist_km
        )
    return tiles


def build_query_set_all(earthloc_queries_dir: str, gape_queries_dir: str) -> list[GeoTile]:
    by_id: dict[str, GeoTile] = {}
    total = 0
    for source_dir in (earthloc_queries_dir, gape_queries_dir):
        for q in load_query_set(source_dir):
            total += 1
            if not _near_eval_region(q.meta["nadir_lat"], q.meta["nadir_lon"]):
                by_id[q.tile_id] = q
    print(f"query pool: {len(by_id)}/{total} (excluding held-out eval regions, deduped)")
    return list(by_id.values())
