"""Build (query, reference tile) positive training pairs by geometric
footprint overlap (IoU), reusing scripts/evaluate.py's IoU machinery.

This is the whole reason AstroLoc's own weak-to-full labeling pipeline
(SuperPoint+LightGlue+EarthMatch over 300k photos) is not needed here: every
query already carries a full 4-corner footprint (either from the original
EarthLoc queries or from GAPE's mlcoord table, see astroloc/data/gape_download.py),
so finding a positive reference tile is a geometric IoU computation, not a
feature-matching problem. See repo memory astroloc_target_architecture.
"""

import numpy as np

from core.types import GeoTile
from scripts.evaluate import find_positive_tile_ids, footprint_bbox, footprint_iou


def build_positive_pairs(
    queries: list[GeoTile], tiles: list[GeoTile], iou_threshold: float = 0.1
) -> list[tuple[GeoTile, GeoTile]]:
    """One (query, tile) pair per query that has at least one positive, picking
    the highest-IoU tile among its positives. Queries with zero positives
    (e.g. photo falls in a training region but no reference tile happens to
    overlap it at this reference-tile density) are dropped.

    Correctness note: this docstring always claimed "highest-IoU", but the
    code used to just take positive_ids[0] -- find_positive_tile_ids returns
    hits in tile-list order, not IoU order, so this was silently picking an
    arbitrary (list-order-dependent) positive, not the best one. Mattered
    little when `tiles` was single-zoom (any positive was already the only
    game in town), but matters a lot once `tiles` spans multiple zoom levels
    (see nano/data.py::build_reference_tiles_multi_zoom): a query could have
    positives at two different scales, and only the tighter-matching one is
    actually the good training signal.
    """
    tile_bboxes = np.array([footprint_bbox(t) for t in tiles])
    tiles_by_id = {t.tile_id: t for t in tiles}
    pairs = []
    for query in queries:
        positive_ids = find_positive_tile_ids(query, tiles, tile_bboxes, iou_threshold)
        if not positive_ids:
            continue
        best_id = max(positive_ids, key=lambda tid: footprint_iou(query, tiles_by_id[tid]))
        pairs.append((query, tiles_by_id[best_id]))
    return pairs
