"""Nano eval: recall@k + retrieval-only coordinate/distance error, same shape
as astroloc/eval/evaluate.py. Reference tiles come from
nano/data.py::build_eval_reference_tiles_multi_zoom (zoom 9+10 by default,
per the RAM-vs-recall tradeoff decision -- see that function's docstring;
pass --zooms 09 to fall back to the original zoom-9-only DB).

The RAM/resource measurement is only meaningful on CPU (this variant's whole
point is nanosatellite-plausible resource use), but building a multi-zoom
index on CPU is slow (~90min/region). The realistic workflow, and the one
this script is built for: run once with --device cuda:0 --rebuild-db to
build+cache the index fast (index files are device-agnostic once saved --
a real deployment would build the index on ground hardware anyway, not on
the satellite), then run again with --device cpu (no --rebuild-db) to load
that cache and do the actual retrieval+coord eval with CPU RAM sampling via
astroloc/eval/evaluate_cpu_profiled.py::ResourceSampler -- "around the evals",
per the request that created this script.
"""

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from database.reference_database import ReferenceDatabase, dedup_search
from index.faiss_index import FaissFlatIndex
from scripts.evaluate import (
    REGIONS,
    evaluate_retrieval,
    find_positive_tile_ids,
    footprint_bbox,
    haversine_km,
    load_image_array,
    load_scoped_queries,
)

from astroloc.eval.evaluate_cpu_profiled import ResourceSampler
from astroloc.models.dinov2_salad import DinoV2SaladRetriever

from nano.data import TRAIN_ZOOMS, build_eval_reference_tiles_multi_zoom

TRAIN_DIR = "/mnt/sdc1/astroloc/reference_db/nano_train"


def evaluate_coords(db: ReferenceDatabase, queries: list, iou_threshold: float = 0.2) -> dict:
    db_tiles = list(db.tiles.values())
    db_bboxes = np.array([footprint_bbox(t) for t in db_tiles])
    errors_km = []
    top1_hit = 0
    n_scored = 0
    for query in queries:
        positives = find_positive_tile_ids(query, db_tiles, db_bboxes, iou_threshold)
        if not positives:
            continue
        n_scored += 1
        image = load_image_array(query.image_path)
        descriptor = db.retriever.embed(image)
        top1_id, _ = dedup_search(db.index, descriptor, 1)[0]
        if top1_id in positives:
            top1_hit += 1
        top1_tile = db.tiles[top1_id]
        pred_center = top1_tile.corners_latlon.mean(axis=0)
        gt_center = query.corners_latlon.mean(axis=0)
        errors_km.append(haversine_km(gt_center[0], gt_center[1], pred_center[0], pred_center[1]))

    return {
        "num_scored": n_scored,
        "top1_correct_tile_rate": 100.0 * top1_hit / n_scored if n_scored else 0.0,
        "median_coord_error_km": float(np.median(errors_km)) if errors_km else None,
        "mean_coord_error_km": float(np.mean(errors_km)) if errors_km else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--db-year", type=int, default=2021)
    ap.add_argument("--db-dist-km", type=float, default=5000)
    ap.add_argument("--query-dist-km", type=float, default=2500)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 100])
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--coord-eval-limit", type=int, default=200)
    ap.add_argument("--device", default="cpu", help="cpu by default -- the deployment-relevant regime for nano")
    ap.add_argument("--cache-root", default=os.path.join(TRAIN_DIR, "eval_cache"))
    ap.add_argument("--rebuild-db", action="store_true")
    ap.add_argument("--wandb-project", default="astroloc-demo")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS.keys()))
    ap.add_argument("--sample-interval-s", type=float, default=1.0)
    ap.add_argument("--zooms", nargs="+", default=list(TRAIN_ZOOMS))
    args = ap.parse_args()

    retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device=args.device)
    print(f"Loaded retriever: backbone={retriever.model.backbone_name}, descriptor_dim={retriever.descriptor_dim}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=f"eval-{args.run_name}", config=vars(args))

    all_results = {}
    for region in args.regions:
        center_lat, center_lon = REGIONS[region]
        cache_dir = os.path.join(args.cache_root, args.run_name, region.replace(" ", "_"))

        index = FaissFlatIndex(retriever.descriptor_dim)
        if not args.rebuild_db and os.path.exists(os.path.join(cache_dir, "tiles.json")):
            print(f"[{region}] loading cached reference db from {cache_dir}")
            db = ReferenceDatabase.load(cache_dir, retriever, index)
        else:
            print(f"[{region}] scoping + building zoom-{'+'.join(args.zooms)} reference db...")
            db_tiles = build_eval_reference_tiles_multi_zoom(
                args.database_dir, center_lat, center_lon, args.db_year, args.db_dist_km,
                zooms=tuple(args.zooms),
            )
            print(f"[{region}] {len(db_tiles)} reference tiles within {args.db_dist_km}km")
            db = ReferenceDatabase(retriever, index)
            t0 = time.time()
            db.build(db_tiles)
            print(f"[{region}] built in {time.time() - t0:.0f}s")
            db.save(cache_dir)

        queries = load_scoped_queries(args.queries_dir, center_lat, center_lon, args.query_dist_km)
        print(f"[{region}] {len(queries)} queries within {args.query_dist_km}km")

        sampler = ResourceSampler(interval_s=args.sample_interval_s)
        with sampler:
            t0 = time.time()
            recalls, num_evaluated = evaluate_retrieval(db, queries, args.k_values, args.iou_threshold)
            retrieval_eval_s = time.time() - t0
            print(f"[{region}] retrieval eval ({num_evaluated} queries) in {retrieval_eval_s:.0f}s: {recalls}")

            coord_queries = queries[: args.coord_eval_limit] if args.coord_eval_limit else queries
            t0 = time.time()
            coord_stats = evaluate_coords(db, coord_queries, args.iou_threshold)
            coord_eval_s = time.time() - t0
            print(f"[{region}] coord eval ({coord_stats['num_scored']} queries) in {coord_eval_s:.0f}s: {coord_stats}")
        resource_stats = sampler.summary()
        print(f"[{region}] resource use during eval: {resource_stats}")

        region_result = {
            "recalls": recalls,
            "num_evaluated": num_evaluated,
            "coords": coord_stats,
            "resource_use_during_eval": resource_stats,
            "num_db_tiles": len(db.tiles),
        }
        all_results[region] = region_result

        if use_wandb:
            import wandb

            wandb.log(
                {f"{region}/R@{k}": v for k, v in recalls.items()}
                | {
                    f"{region}/top1_correct_tile_rate": coord_stats["top1_correct_tile_rate"],
                    f"{region}/median_coord_error_km": coord_stats["median_coord_error_km"] or float("nan"),
                    f"{region}/peak_rss_mb": resource_stats["peak_rss_mb"] or float("nan"),
                    f"{region}/mean_rss_mb": resource_stats["mean_rss_mb"] or float("nan"),
                    f"{region}/peak_power_w": resource_stats["peak_power_w"] or float("nan"),
                    f"{region}/num_db_tiles": len(db.tiles),
                }
            )

    out_path = os.path.join(args.cache_root, f"results_{args.run_name}.json")
    os.makedirs(args.cache_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")
    print(json.dumps(all_results, indent=2))

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
