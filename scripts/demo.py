"""One-frame demo: retrieve the matching tile, draw inlier correspondences,
plot the estimated footprint. See docs/argus_localization_spec.md section 7.

Requires a cached reference database (built by scripts/evaluate.py, which
saves it under user_config['cache_dir']).
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import LocalizationPipeline
from core.types import PipelineConfig
from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from retrievers.earthloc_retriever import EarthLocRetriever


def draw_corner_sanity_check(tile_image: np.ndarray, tile, georef: Georeferencer) -> np.ndarray:
    """Overlay lat/lon labels at the tile's pixel corners and center.

    This is the sanity check the corner-order assumption in
    georeference/georeferencer.py needs: if the BL,TL,TR,BR order were wrong,
    the printed corner coordinates would be visibly transposed or flipped
    relative to the tile image instead of failing with an exception.
    """
    vis = tile_image.copy()
    h, w = vis.shape[:2]
    pixel_pts = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [w // 2, h // 2]], dtype=np.float32
    )
    labels = ["px(0,0)", "px(w,0)", "px(w,h)", "px(0,h)", "center"]
    latlon = georef.tile_pixel_to_latlon(tile, vis.shape, pixel_pts)
    for (x, y), (lat, lon), label in zip(pixel_pts, latlon, labels):
        cv2.circle(vis, (int(x), int(y)), 6, (255, 0, 0), -1)
        text = f"{label} {lat:.2f},{lon:.2f}"
        tx = int(min(max(x - 60, 0), w - 165))
        ty = int(min(max(y + 20, 15), h - 5))
        cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    return vis


def draw_matches(query_image: np.ndarray, tile_image: np.ndarray, match) -> np.ndarray:
    height = max(query_image.shape[0], tile_image.shape[0])
    width = query_image.shape[1] + tile_image.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: query_image.shape[0], : query_image.shape[1]] = query_image
    canvas[: tile_image.shape[0], query_image.shape[1] :] = tile_image
    offset = query_image.shape[1]
    for (qx, qy), (tx, ty), inlier in zip(match.query_pts, match.tile_pts, match.inlier_mask):
        color = (0, 255, 0) if inlier else (0, 0, 255)
        cv2.line(canvas, (int(qx), int(qy)), (int(tx) + offset, int(ty)), color, 1, cv2.LINE_AA)
    return canvas


def run_demo(image_path: str, config_path: str, user_config_path: str, output_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(user_config_path) as f:
        user_config = yaml.safe_load(f)

    device = config.get("device", "cuda")
    retriever = EarthLocRetriever(user_config["earthloc_checkpoint"], device=device)
    index = FaissFlatIndex(retriever.descriptor_dim)

    cache_dir = os.path.join(user_config["cache_dir"], "db_Alps")
    if not os.path.exists(os.path.join(cache_dir, "tiles.json")):
        raise FileNotFoundError(
            f"No cached reference database at {cache_dir}. Run scripts/evaluate.py first, "
            "it builds and caches the database."
        )
    db = ReferenceDatabase.load(cache_dir, retriever, index)

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=config["pipeline"]["min_inliers"],
        device=device,
    )
    georef = Georeferencer()
    pipeline = LocalizationPipeline(db, matcher, georef, PipelineConfig(**config["pipeline"]))

    query_image = np.array(Image.open(image_path).convert("RGB"))
    result = pipeline.localize(query_image)

    print(f"status: {result.status}")
    print(f"matched_tile_id: {result.matched_tile_id}")
    print(f"confidence (num_inliers): {result.confidence}")
    if result.query_footprint_latlon is not None:
        print(f"estimated query footprint, lat/lon, order BL,TL,TR,BR:\n{result.query_footprint_latlon}")

    if result.matched_tile_id is None:
        print("No candidate tile at all (empty database or empty shortlist), nothing to render.")
        return

    tile = db.get_tile(result.matched_tile_id)
    tile_image = np.array(Image.open(tile.image_path).convert("RGB"))
    match = result.debug["best_match"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, ext = os.path.splitext(output_path)

    matches_path = f"{base}_matches{ext}"
    cv2.imwrite(matches_path, cv2.cvtColor(draw_matches(query_image, tile_image, match), cv2.COLOR_RGB2BGR))

    sanity_path = f"{base}_corner_sanity{ext}"
    cv2.imwrite(sanity_path, cv2.cvtColor(draw_corner_sanity_check(tile_image, tile, georef), cv2.COLOR_RGB2BGR))

    print(f"Saved {matches_path} and {sanity_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the localizer on one frame and visualize it.")
    parser.add_argument("image_path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--user-config", default="user_config.yaml")
    parser.add_argument("--output", default="output/demo.png")
    args = parser.parse_args()
    run_demo(args.image_path, args.config, args.user_config, args.output)


if __name__ == "__main__":
    main()
