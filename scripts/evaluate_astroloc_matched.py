"""Full retrieve-then-match evaluation of the fine-tuned astroloc/ retriever
(DINOv2+SALAD, LoRA), reusing the existing SiftLightGlueMatcher/Georeferencer/
LocalizationPipeline unchanged -- only the retriever is swapped in, exactly
like scripts/evaluate.py does for EarthLocRetriever. Directly comparable to
that script's own numbers (e.g. Alps: 44% fix rate, 3.0km median error),
since it reuses the same evaluate_matching() function.

This is the number astroloc/eval/evaluate.py's coordinate estimate does NOT
measure: that script's "coord eval" is retrieval-only (top-1 tile centroid),
not matched (SIFT-LightGlue + homography + per-pixel interpolation).
"""

import argparse
import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from core.pipeline import LocalizationPipeline
from core.types import PipelineConfig
from database.reference_database import ReferenceDatabase
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from georeference.georeferencer import Georeferencer
from scripts.evaluate import REGIONS, evaluate_matching, load_scoped_queries

from astroloc.models.dinov2_salad import DinoV2SaladRetriever


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="Alps", choices=list(REGIONS.keys()))
    ap.add_argument("--checkpoint", default="/mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_lora/final.pt")
    ap.add_argument("--db-cache", default=None, help="defaults to the eval_cache dir this checkpoint's own eval run already built")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--user-config", default="user_config.yaml")
    ap.add_argument("--num-queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)

    device = config.get("device", "cuda")
    center_lat, center_lon = REGIONS[args.region]

    retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device=device)

    db_cache = args.db_cache or (
        f"/mnt/sdc1/astroloc/reference_db/astroloc_train/eval_cache/lora_finetuned_a/{args.region.replace(' ', '_')}"
    )
    index = FaissFlatIndex(retriever.descriptor_dim)
    logging.info(f"Loading cached reference database from {db_cache}")
    db = ReferenceDatabase.load(db_cache, retriever, index)
    logging.info(f"{len(db.tiles)} reference tiles loaded")

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=config["pipeline"]["min_inliers"],
        device=device,
    )
    georef = Georeferencer()
    pipeline_config = PipelineConfig(**config["pipeline"])
    pipeline = LocalizationPipeline(db, matcher, georef, pipeline_config)

    queries = load_scoped_queries(
        user_config["queries_dir"], center_lat, center_lon, config["eval"]["query_dist_km"]
    )
    matching_queries = random.sample(queries, min(args.num_queries, len(queries)))
    logging.info(f"Evaluating matching on {len(matching_queries)} of {len(queries)} queries...")

    stats = evaluate_matching(pipeline, matching_queries)
    print(f"\n=== Matched pipeline ({args.region}, astroloc DINOv2+SALAD LoRA retriever, {stats['num_queries']} queries) ===")
    print(f"mean best-candidate num_inliers: {stats['mean_num_inliers']:.1f}")
    print(f"fix rate (num_inliers >= {config['pipeline']['min_inliers']}): {stats['fix_rate']:.1f}%")
    median_error = stats["median_localization_error_km"]
    print(f"median localization error: {median_error:.2f} km" if median_error is not None else "median localization error: n/a")


if __name__ == "__main__":
    main()
