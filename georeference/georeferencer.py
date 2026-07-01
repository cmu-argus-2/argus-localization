"""Pixel to lat/lon mapping over a reference tile's four known corners.

Database tiles are nadir, so bilinear interpolation over the corner
footprint is treated as a regular grid. Pure numpy, no model dependency.
See docs/argus_localization_spec.md section 4 and 5.

Corner order (see core/types.py GeoTile.corners_latlon): bottom-left,
top-left, top-right, bottom-right. This was verified empirically against
EarthLoc reference tiles (corner 0 is always the lowest lat and lowest lon
of the four), since the EarthLoc filename schema does not document
pixel-to-corner order explicitly. It does NOT match the TL,TR,BR,BL order
in the original spec draft. demo.py draws the interpolated grid over the
tile image as a visual sanity check, since a wrong order shows up as a
flipped or rotated footprint.
"""

import cv2
import numpy as np

from core.types import GeoTile, MatchResult, TiePoint


class Georeferencer:
    def tile_pixel_to_latlon(self, tile: GeoTile, tile_shape, pixels: np.ndarray) -> np.ndarray:
        height, width = tile_shape[:2]
        u_norm = pixels[:, 0] / max(width - 1, 1)
        v_norm = pixels[:, 1] / max(height - 1, 1)
        bottom_left, top_left, top_right, bottom_right = tile.corners_latlon
        top = top_left[None, :] + u_norm[:, None] * (top_right - top_left)[None, :]
        bottom = bottom_left[None, :] + u_norm[:, None] * (bottom_right - bottom_left)[None, :]
        return top + v_norm[:, None] * (bottom - top)

    def make_tie_points(self, match: MatchResult, tile: GeoTile, tile_shape) -> list[TiePoint]:
        inlier_query_pts = match.query_pts[match.inlier_mask]
        inlier_tile_pts = match.tile_pts[match.inlier_mask]
        latlon = self.tile_pixel_to_latlon(tile, tile_shape, inlier_tile_pts)
        return [
            TiePoint(u=float(u), v=float(v), lat=float(lat), lon=float(lon))
            for (u, v), (lat, lon) in zip(inlier_query_pts, latlon)
        ]

    def estimate_query_footprint(
        self, match: MatchResult, tile: GeoTile, query_shape, tile_shape
    ) -> np.ndarray | None:
        if match.homography is None or match.num_inliers < 4:
            return None
        height_query, width_query = query_shape[:2]
        query_corners = np.array(
            [[0, 0], [width_query - 1, 0], [width_query - 1, height_query - 1], [0, height_query - 1]],
            dtype=np.float32,
        )
        tile_pixels = cv2.perspectiveTransform(query_corners[None, :, :], match.homography)[0]
        return self.tile_pixel_to_latlon(tile, tile_shape, tile_pixels)
