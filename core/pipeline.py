"""Retrieve-then-match orchestration.

Stateless per call: frame in, LocalizationResult out. See
docs/argus_localization_spec.md section 1: retrieve top_k candidates, match
each one, keep the max-inlier candidate, threshold on min_inliers.
"""

import numpy as np
from PIL import Image

from core.interfaces import Matcher
from core.types import LocalizationResult, PipelineConfig
from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer


class LocalizationPipeline:
    def __init__(
        self,
        db: ReferenceDatabase,
        matcher: Matcher,
        georef: Georeferencer,
        config: PipelineConfig,
    ) -> None:
        self.db = db
        self.matcher = matcher
        self.georef = georef
        self.config = config

    def localize(self, frame: np.ndarray) -> LocalizationResult:
        candidates = self.db.retrieve(frame, self.config.top_k)

        best_tile = None
        best_match = None
        best_tile_image = None
        debug_candidates = []
        for tile, similarity in candidates:
            tile_image = np.array(Image.open(tile.image_path).convert("RGB"))
            match = self.matcher.match(frame, tile_image, tile_id=tile.tile_id)
            debug_candidates.append(
                {
                    "tile_id": tile.tile_id,
                    "retrieval_similarity": similarity,
                    "num_inliers": match.num_inliers,
                }
            )
            if best_match is None or match.num_inliers > best_match.num_inliers:
                best_tile, best_match, best_tile_image = tile, match, tile_image

        # best_match is kept in full (not just the scalar fields above) so callers like
        # demo.py can render the actual correspondences without rerunning the matcher.
        debug = {"candidates": debug_candidates, "best_match": best_match}

        if best_match is None or best_match.num_inliers < self.config.min_inliers:
            return LocalizationResult(
                status="no_fix",
                confidence=float(best_match.num_inliers) if best_match is not None else 0.0,
                tie_points=[],
                matched_tile_id=best_tile.tile_id if best_tile is not None else None,
                query_footprint_latlon=None,
                debug=debug,
            )

        tile_shape = best_tile_image.shape
        tie_points = self.georef.make_tie_points(best_match, best_tile, tile_shape)
        query_footprint = self.georef.estimate_query_footprint(
            best_match, best_tile, frame.shape, tile_shape
        )
        return LocalizationResult(
            status="fix",
            confidence=float(best_match.num_inliers),
            tie_points=tie_points,
            matched_tile_id=best_tile.tile_id,
            query_footprint_latlon=query_footprint,
            debug=debug,
        )
