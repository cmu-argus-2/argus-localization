"""Collects a per-frame FULL-PIPELINE (retrieve -> SIFT+LightGlue match ->
homography -> georeference) localization-error dataset across a fixed set of
trained retrievers and all 6 benchmark regions, for later least-squares
error/noise modeling (error_km ~ num_inliers, retrieval_similarity, rank,
region) -- the intended input to OD's per-measurement noise weighting,
replacing the single fixed constant integration/od_integration_test.py
currently derives from one README number.

Deliberately NOT astroloc/eval/evaluate.py's "coord eval": that measures only
the top-1 retrieved tile's centroid vs ground truth (retrieval-only). This
script runs the real LocalizationPipeline (same as scripts/evaluate.py /
scripts/evaluate_astroloc_matched.py), so error_km reflects actual matched
tie-point georeferencing -- what OD will actually receive.

Each of the 3 models below already has a cached ReferenceDatabase (built by
an earlier eval run) at its own eval_cache dir, in the same on-disk format
(index.faiss/index.tile_ids.json/tiles.json) regardless of astroloc vs nano
-- so no reference-tile re-embedding is needed here, only the matching stage.

Writes one row per query to --out as JSONL, flushed immediately after each
query, so a partial/interrupted run still leaves usable data.
"""

import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from core.pipeline import LocalizationPipeline
from core.types import PipelineConfig
from database.reference_database import ReferenceDatabase
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from georeference.georeferencer import Georeferencer
from scripts.evaluate import REGIONS, haversine_km, load_image_array, load_scoped_queries

from astroloc.models.dinov2_salad import DinoV2SaladRetriever

ASTROLOC_ROOT = "/mnt/sdc1/astroloc/reference_db/astroloc_train"
NANO_ROOT = "/mnt/sdc1/astroloc/reference_db/nano_train"

MODELS = {
    "faithful_lora_v2": {
        "checkpoint": f"{ASTROLOC_ROOT}/checkpoints_faithful_lora_v2/final.pt",
        "db_cache_root": f"{ASTROLOC_ROOT}/eval_cache/faithful_lora_v2",
    },
    "faithful_dynamic_v2": {
        "checkpoint": f"{ASTROLOC_ROOT}/checkpoints_faithful_dynamic_v2/final.pt",
        "db_cache_root": f"{ASTROLOC_ROOT}/eval_cache/faithful_dynamic_v2",
    },
    "nano_v2": {
        "checkpoint": f"{NANO_ROOT}/checkpoints_v2/final.pt",
        "db_cache_root": f"{NANO_ROOT}/eval_cache/nano-v2-multizoom-lora-dynamic",
    },
}


def collect_region(pipeline, model_name, region, queries, out_f):
    for query in queries:
        t0 = time.time()
        image = load_image_array(query.image_path)
        try:
            result = pipeline.localize(image)
        except Exception as e:
            # The vendored LightGlue SIFT extractor can throw on a degenerate
            # candidate tile image (confirmed directly: "ValueError: array is
            # not broadcastable to correct shape" in filter_dog_point, hit by
            # both faithful_lora_v2 and faithful_dynamic_v2 partway through
            # Amazon -- core/pipeline.py has no try/except around the matcher
            # call, so this is a real production fragility, not just a data-
            # collection nuisance). Logged as its own status instead of
            # crashing the whole collection run over one bad candidate.
            elapsed_s = time.time() - t0
            out_f.write(json.dumps({
                "model": model_name, "region": region, "query_id": query.tile_id,
                "status": "error", "num_inliers": None, "retrieval_similarity": None,
                "retrieval_rank": None, "error_km": None, "elapsed_s": elapsed_s,
                "error_msg": f"{type(e).__name__}: {e}",
            }) + "\n")
            out_f.flush()
            yield {"status": "error"}
            continue
        elapsed_s = time.time() - t0

        retrieval_similarity = None
        retrieval_rank = None
        if result.debug is not None and result.matched_tile_id is not None:
            for rank, cand in enumerate(result.debug["candidates"]):
                if cand["tile_id"] == result.matched_tile_id:
                    retrieval_similarity = cand["retrieval_similarity"]
                    retrieval_rank = rank
                    break

        error_km = None
        if result.status == "fix":
            gt_center = query.corners_latlon.mean(axis=0)
            if result.query_footprint_latlon is not None:
                est_center = result.query_footprint_latlon.mean(axis=0)
            else:
                est_center = None
            if est_center is not None:
                error_km = haversine_km(gt_center[0], gt_center[1], est_center[0], est_center[1])

        row = {
            "model": model_name,
            "region": region,
            "query_id": query.tile_id,
            "status": result.status,
            "num_inliers": result.confidence,
            "retrieval_similarity": retrieval_similarity,
            "retrieval_rank": retrieval_rank,
            "error_km": error_km,
            "elapsed_s": elapsed_s,
        }
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument("--regions", nargs="+", default=list(REGIONS.keys()))
    ap.add_argument("--n-per-region", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--user-config", default="user_config.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=config["pipeline"]["min_inliers"],
        device=args.device,
    )
    georef = Georeferencer()
    pipeline_config = PipelineConfig(**config["pipeline"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as out_f:
        for model_name in args.models:
            spec = MODELS[model_name]
            logging.info(f"=== {model_name}: loading checkpoint {spec['checkpoint']} ===")
            retriever = DinoV2SaladRetriever.from_checkpoint(spec["checkpoint"], device=args.device)

            for region in args.regions:
                center_lat, center_lon = REGIONS[region]
                cache_dir = os.path.join(spec["db_cache_root"], region.replace(" ", "_"))
                index = FaissFlatIndex(retriever.descriptor_dim)
                logging.info(f"[{model_name}/{region}] loading cached reference db from {cache_dir}")
                db = ReferenceDatabase.load(cache_dir, retriever, index)
                logging.info(f"[{model_name}/{region}] {len(db.tiles)} reference tiles loaded")

                pipeline = LocalizationPipeline(db, matcher, georef, pipeline_config)

                queries = load_scoped_queries(
                    user_config["queries_dir"], center_lat, center_lon, config["eval"]["query_dist_km"]
                )
                rng = random.Random(args.seed)
                sampled = rng.sample(queries, min(args.n_per_region, len(queries)))
                logging.info(f"[{model_name}/{region}] collecting {len(sampled)} of {len(queries)} queries...")

                t0 = time.time()
                n_fix = 0
                for i, row in enumerate(collect_region(pipeline, model_name, region, sampled, out_f)):
                    if row["status"] == "fix":
                        n_fix += 1
                    if (i + 1) % 50 == 0:
                        logging.info(
                            f"[{model_name}/{region}] {i + 1}/{len(sampled)} done, "
                            f"fix_rate={100.0 * n_fix / (i + 1):.1f}%, elapsed={time.time() - t0:.0f}s"
                        )
                logging.info(
                    f"[{model_name}/{region}] DONE: {len(sampled)} queries in {time.time() - t0:.0f}s, "
                    f"fix_rate={100.0 * n_fix / len(sampled) if sampled else 0:.1f}%"
                )

    logging.info(f"All done, dataset at {args.out}")


if __name__ == "__main__":
    main()
