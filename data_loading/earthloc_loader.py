"""Loader for EarthLoc-format reference tiles and queries.

Filename schema (both database/YYYY_MM/*.jpg and queries/*.jpg):

    @lat1@lon1@lat2@lon2@lat3@lon3@lat4@lon4@image_id@timestamp@nadir_lat@nadir_lon@sq_km_area@orientation@.jpg

The four lat/lon pairs are the footprint corners. Queries carry footprints
that double as ground truth for eval. See docs/argus_localization_spec.md
section 6.
"""

import glob
import os

import numpy as np

from core.types import GeoTile

_NUM_FIELDS = 14  # 8 corner coords, image_id, timestamp, nadir_lat, nadir_lon, sq_km_area, orientation


def parse_geotile_filename(path: str) -> GeoTile:
    stem = os.path.splitext(os.path.basename(path))[0]
    fields = [f for f in stem.split("@") if f != ""]
    if len(fields) != _NUM_FIELDS:
        raise ValueError(f"expected {_NUM_FIELDS} @-delimited fields, got {len(fields)}: {path}")

    lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4 = (float(x) for x in fields[0:8])
    image_id = fields[8]
    timestamp = fields[9]
    nadir_lat, nadir_lon, sq_km_area, orientation = (float(x) for x in fields[10:14])

    corners_latlon = np.array(
        [[lat1, lon1], [lat2, lon2], [lat3, lon3], [lat4, lon4]], dtype=np.float64
    )

    # image_id alone repeats across years for the same grid cell in the multi-temporal
    # database (same cell captured in 2018, 2019, 2020, ...), so tile_id must include
    # the timestamp to stay unique.
    tile_id = f"{image_id}@{timestamp}"

    return GeoTile(
        tile_id=tile_id,
        image_path=path,
        corners_latlon=corners_latlon,
        timestamp=timestamp,
        meta={
            "nadir_lat": nadir_lat,
            "nadir_lon": nadir_lon,
            "sq_km_area": sq_km_area,
            "orientation": orientation,
        },
    )


def load_reference_tiles(database_dir: str) -> list[GeoTile]:
    paths = glob.glob(os.path.join(database_dir, "**", "*.jpg"), recursive=True)
    return [parse_geotile_filename(p) for p in paths]


def load_query_set(queries_dir: str) -> list[GeoTile]:
    paths = glob.glob(os.path.join(queries_dir, "*.jpg"))
    return [parse_geotile_filename(p) for p in paths]
