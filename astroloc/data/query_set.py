"""Combined training query set: the original rsynced EarthLoc queries plus the
newly downloaded GAPE mlcoord queries (astroloc/data/gape_download.py), both
scoped to the training regions (astroloc/data/regions.py) so they're
geographically co-located with the reference tile set (astroloc/data/
reference_tiles.py) and reasonably disjoint from the 6 held-out eval regions.
"""

import math

from core.types import GeoTile
from data_loading.earthloc_loader import load_query_set

from astroloc.data.regions import TRAIN_QUERY_RADIUS_KM, TRAIN_REGIONS


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def _in_train_region(query: GeoTile) -> bool:
    nlat, nlon = query.meta["nadir_lat"], query.meta["nadir_lon"]
    return any(
        _haversine_km(nlat, nlon, clat, clon) < TRAIN_QUERY_RADIUS_KM
        for clat, clon in TRAIN_REGIONS.values()
    )


def build_query_set(earthloc_queries_dir: str, gape_queries_dir: str) -> list[GeoTile]:
    by_id: dict[str, GeoTile] = {}
    for source_dir in (earthloc_queries_dir, gape_queries_dir):
        for q in load_query_set(source_dir):
            if _in_train_region(q):
                by_id[q.tile_id] = q
    return list(by_id.values())
