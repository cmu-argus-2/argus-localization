"""Batch export of localized coordinates, ready for OD to consume later.

Runs the existing retrieve-then-match pipeline (EarthLocRetriever +
FaissFlatIndex retrieval, SiftLightGlueMatcher matching, Georeferencer)
over a set of frames and writes one record per frame: matched tile,
confidence, tie points (image pixel <-> lat/lon), and the estimated query
footprint. This is upstream of integration/batchopt_adapter.py on purpose,
see docs/argus_localization_spec.md section 8: OD integration is a later
phase, this script only produces the coordinates that adapter will
eventually consume.

Frames default to the region-scoped EarthLoc query set (for testing against
data already on this machine). Pass --frames-dir to point at any directory
of images instead, since the pipeline only needs a plain (H,W,3) uint8
array per frame, not the EarthLoc filename schema, so future Argus-captured
frames drop in the same way.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import LocalizationPipeline
from core.types import LocalizationResult, PipelineConfig
from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from retrievers.earthloc_retriever import EarthLocRetriever
from scripts.evaluate import REGIONS, load_scoped_db_tiles, load_scoped_queries

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_image_array(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def gather_frames(args, user_config: dict, center_lat: float, center_lon: float) -> list[tuple[str, str]]:
    """Returns a list of (frame_id, image_path)."""
    if args.frames_dir is not None:
        paths = sorted(
            p for p in glob.glob(os.path.join(args.frames_dir, "*")) if p.lower().endswith(_IMAGE_EXTENSIONS)
        )
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]

    with open(args.config) as f:
        config = yaml.safe_load(f)
    queries = load_scoped_queries(
        user_config["queries_dir"], center_lat, center_lon, config["eval"]["query_dist_km"]
    )
    return [(tile.tile_id, tile.image_path) for tile in queries]


def result_to_record(frame_id: str, frame_path: str, result: LocalizationResult) -> dict:
    return {
        "frame_id": frame_id,
        "frame_path": frame_path,
        "status": result.status,
        "confidence_num_inliers": result.confidence,
        "matched_tile_id": result.matched_tile_id,
        "tie_points": [
            {"u": tp.u, "v": tp.v, "lat": tp.lat, "lon": tp.lon} for tp in result.tie_points
        ],
        "query_footprint_latlon": (
            result.query_footprint_latlon.tolist() if result.query_footprint_latlon is not None else None
        ),
    }


def build_or_load_db(args, config: dict, user_config: dict, center_lat: float, center_lon: float) -> ReferenceDatabase:
    device = config.get("device", "cuda")
    retriever = EarthLocRetriever(user_config["earthloc_checkpoint"], device=device)
    index = FaissFlatIndex(retriever.descriptor_dim)

    cache_dir = os.path.join(user_config["cache_dir"], f"db_{args.region.replace(' ', '_')}")
    if not args.rebuild_db and os.path.exists(os.path.join(cache_dir, "tiles.json")):
        logging.info(f"Loading cached reference database from {cache_dir}")
        return ReferenceDatabase.load(cache_dir, retriever, index)

    logging.info(f"Scoping and building reference database for region {args.region}")
    db_tiles = load_scoped_db_tiles(
        user_config["database_dir"], center_lat, center_lon,
        config["eval"]["db_year"], config["eval"]["db_dist_km"],
    )
    logging.info(f"{len(db_tiles)} reference tiles within {config['eval']['db_dist_km']} km of {args.region}")
    db = ReferenceDatabase(retriever, index)
    t0 = time.time()
    db.build(db_tiles)
    logging.info(f"Built and embedded reference database in {time.time() - t0:.1f}s")
    db.save(cache_dir)
    return db


def main():
    parser = argparse.ArgumentParser(
        description="Run the retrieve-then-match pipeline over a set of frames and export coordinates."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--user-config", default="user_config.yaml")
    parser.add_argument("--region", default="Alps", choices=list(REGIONS.keys()))
    parser.add_argument("--rebuild-db", action="store_true", help="ignore any cached reference database")
    parser.add_argument("--frames-dir", default=None, help="directory of images to localize; defaults to the region-scoped EarthLoc query set")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of frames processed")
    parser.add_argument("--output", default=None, help="defaults to output/coordinates_<region>.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)

    center_lat, center_lon = REGIONS[args.region]
    device = config.get("device", "cuda")

    db = build_or_load_db(args, config, user_config, center_lat, center_lon)

    frames = gather_frames(args, user_config, center_lat, center_lon)
    if args.limit is not None:
        frames = frames[: args.limit]
    logging.info(f"Localizing {len(frames)} frames")

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=config["pipeline"]["min_inliers"],
        device=device,
    )
    georef = Georeferencer()
    pipeline = LocalizationPipeline(db, matcher, georef, PipelineConfig(**config["pipeline"]))

    records = []
    t0 = time.time()
    for frame_id, frame_path in frames:
        frame_image = load_image_array(frame_path)
        result = pipeline.localize(frame_image)
        records.append(result_to_record(frame_id, frame_path, result))
    elapsed = time.time() - t0

    num_fix = sum(1 for r in records if r["status"] == "fix")
    fix_rate = 100.0 * num_fix / len(records) if records else 0.0
    logging.info(f"Localized {len(records)} frames in {elapsed:.1f}s, fix rate {fix_rate:.1f}%")

    output_path = args.output or os.path.join(user_config.get("output_dir", "output"), f"coordinates_{args.region.replace(' ', '_')}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "region": args.region,
        "num_frames": len(records),
        "fix_rate_percent": fix_rate,
        "pipeline_config": {
            "top_k": pipeline.config.top_k,
            "min_inliers": pipeline.config.min_inliers,
            "query_size": pipeline.config.query_size,
            "max_ransac_iters": pipeline.config.max_ransac_iters,
        },
        "records": records,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Wrote coordinates dataset to {output_path}")
    print(f"\n=== Coordinates dataset ({args.region}, {len(records)} frames) ===")
    print(f"fix rate: {fix_rate:.1f}%")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
