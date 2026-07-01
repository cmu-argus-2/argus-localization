"""SIFT-LightGlue matcher (mirrors EarthMatch).

Lean config for near-nadir Argus frames against scale-matched tiles:
max_num_keypoints=1024, img_size=512, hard cap max_ransac_iters=3,
fail fast if inliers < min_inliers. Homography fit with cv2.USAC_MAGSAC
or cv2.RANSAC. Database-side keypoints are precomputed and keyed by
tile_id so runtime only extracts keypoints for the query. See
docs/argus_localization_design.md, Matcher section, and
docs/argus_localization_spec.md section 5.

min_inliers is stored here for config completeness but is not enforced by
this class: the matcher's job is to report num_inliers, the accept/reject
policy on that count lives in LocalizationPipeline (docs/argus_localization_spec.md
section 1), so a candidate that fails the threshold is still reported with
its real match result rather than a nulled-out one.
"""

from typing import Optional

import cv2
import numpy as np
import torch
from lightglue import SIFT, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd

from core.interfaces import Matcher
from core.types import MatchResult


class SiftLightGlueMatcher(Matcher):
    def __init__(
        self,
        max_num_keypoints: int = 1024,
        img_size: int = 512,
        max_ransac_iters: int = 3,
        min_inliers: int = 30,
        device: str = "cuda",
    ):
        self.max_num_keypoints = max_num_keypoints
        self.img_size = img_size
        self.max_ransac_iters = max_ransac_iters
        self.min_inliers = min_inliers
        self.device = device if torch.cuda.is_available() else "cpu"

        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="sift").eval().to(self.device)
        self._tile_feats_cache: dict[str, dict] = {}

    def _extract(self, image: np.ndarray) -> dict:
        tensor = numpy_image_to_torch(image).to(self.device)
        with torch.no_grad():
            return self.extractor.extract(tensor, resize=self.img_size)

    def _extract_tile(self, tile_image: np.ndarray, tile_id: Optional[str]) -> dict:
        if tile_id is not None and tile_id in self._tile_feats_cache:
            return self._tile_feats_cache[tile_id]
        feats = self._extract(tile_image)
        if tile_id is not None:
            self._tile_feats_cache[tile_id] = feats
        return feats

    def match(
        self,
        query_image: np.ndarray,
        tile_image: np.ndarray,
        tile_id: Optional[str] = None,
    ) -> MatchResult:
        query_feats = self._extract(query_image)
        tile_feats = self._extract_tile(tile_image, tile_id)

        with torch.no_grad():
            matches01 = self.matcher({"image0": query_feats, "image1": tile_feats})
        query_feats_r, tile_feats_r, matches01 = (
            rbd(query_feats),
            rbd(tile_feats),
            rbd(matches01),
        )
        matches = matches01["matches"]  # (M, 2) indices into keypoints0 / keypoints1

        if matches.shape[0] < 4:
            return MatchResult(
                query_pts=np.zeros((0, 2), dtype=np.float32),
                tile_pts=np.zeros((0, 2), dtype=np.float32),
                inlier_mask=np.zeros((0,), dtype=bool),
                homography=None,
                num_inliers=0,
            )

        query_pts = query_feats_r["keypoints"][matches[:, 0]].cpu().numpy()
        tile_pts = tile_feats_r["keypoints"][matches[:, 1]].cpu().numpy()

        homography, mask = cv2.findHomography(
            query_pts,
            tile_pts,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=3.0,
            maxIters=self.max_ransac_iters,
            confidence=0.999,
        )
        inlier_mask = (
            np.zeros((query_pts.shape[0],), dtype=bool)
            if mask is None
            else mask.astype(bool).reshape(-1)
        )
        num_inliers = int(inlier_mask.sum())

        return MatchResult(
            query_pts=query_pts,
            tile_pts=tile_pts,
            inlier_mask=inlier_mask,
            homography=homography,
            num_inliers=num_inliers,
        )
