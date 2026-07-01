"""Core data types shared by every pipeline stage.

See docs/argus_localization_spec.md section 3 and 4 for the source spec.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GeoTile:
    tile_id: str
    image_path: str
    corners_latlon: np.ndarray  # (4,2) lat,lon; order is BL,TL,TR,BR (verified against
    # EarthLoc data: corner 0 is always the lowest lat and lowest lon of the four, corner
    # 1 shares its lon but has the highest lat, and so on counterclockwise). This differs
    # from the TL,TR,BR,BL order assumed in the original spec draft.
    timestamp: Optional[str] = None
    meta: Optional[dict] = None


@dataclass(frozen=True)
class TiePoint:
    u: float
    v: float  # query pixel
    lat: float
    lon: float  # world


@dataclass(frozen=True)
class LocalizationResult:
    status: str  # "fix" | "no_fix"
    confidence: float  # inlier count (or normalized)
    tie_points: list[TiePoint]
    matched_tile_id: Optional[str]
    query_footprint_latlon: Optional[np.ndarray]  # (4,2) or None
    debug: Optional[dict] = None  # per-candidate scores, timings


@dataclass(frozen=True)
class MatchResult:
    query_pts: np.ndarray  # (M,2)
    tile_pts: np.ndarray  # (M,2)
    inlier_mask: np.ndarray  # (M,) bool
    homography: Optional[np.ndarray]  # (3,3) query->tile or None
    num_inliers: int


@dataclass
class PipelineConfig:
    top_k: int = 15  # start high, measure recall@k, trade down
    min_inliers: int = 30
    query_size: int = 512
    max_ransac_iters: int = 3
