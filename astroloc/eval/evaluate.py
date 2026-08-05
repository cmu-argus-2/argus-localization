"""Evaluate a DINOv2+SALAD retriever (baseline pretrained or fine-tuned) on
the same 6 EarthLoc/AstroLoc benchmark regions as scripts/evaluate.py and
AstroLoc's own Table 2, reusing that script's region scoping, IoU ground
truth, and ReferenceDatabase/FaissFlatIndex machinery unchanged (only the
retriever is swapped in).

Reports recall@1/5/10/100 (AstroLoc's own k-values) per region, plus a
retrieval-only "predicted coordinate" (the top-1 retrieved tile's footprint
centroid) and its localization error in km against the query's own ground
truth footprint centroid -- this is "getting the coords of the images" that
was asked for: how far AstroLoc-style retrieval alone gets, no SIFT-LightGlue
matching/homography stage (that's the existing pipeline's job, see
core/pipeline.py, and is a separate, later concern).
"""

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    load_scoped_db_tiles,
    load_scoped_queries,
)

from astroloc.models.dinov2_salad import DinoV2SaladModel, DinoV2SaladRetriever


def evaluate_coords(db: ReferenceDatabase, queries: list, iou_threshold: float = 0.2) -> dict:
    """Retrieval-only coordinate estimate: top-1 tile's footprint centroid vs
    the query's own ground-truth footprint centroid. Only counted for queries
    that have at least one true positive in the database (same convention as
    evaluate_retrieval), so this measures localization error conditioned on
    the region actually being coverable, not retrieval failures.
    """
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
    ap.add_argument("--checkpoint", default=None, help="path to a fine-tuned checkpoint; omit for the off-the-shelf pretrained SALAD baseline")
    ap.add_argument("--run-name", required=True, help="namespaces the reference-db cache and output file, e.g. 'baseline' or 'finetuned'")
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--db-year", type=int, default=2021)
    ap.add_argument("--db-dist-km", type=float, default=5000)
    ap.add_argument("--query-dist-km", type=float, default=2500)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 100])
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--coord-eval-limit", type=int, default=200, help="cap queries for the slower per-query coordinate eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-root", default="/mnt/sdc1/astroloc/reference_db/astroloc_train/eval_cache")
    ap.add_argument("--rebuild-db", action="store_true")
    ap.add_argument("--wandb-project", default="astroloc-demo")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS.keys()))
    args = ap.parse_args()

    if args.checkpoint:
        retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device=args.device)
    else:
        model = DinoV2SaladModel(pretrained=True)
        retriever = DinoV2SaladRetriever(model, device=args.device)

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
            print(f"[{region}] scoping + building reference db...")
            db_tiles = load_scoped_db_tiles(args.database_dir, center_lat, center_lon, args.db_year, args.db_dist_km)
            print(f"[{region}] {len(db_tiles)} reference tiles within {args.db_dist_km}km")
            db = ReferenceDatabase(retriever, index)
            t0 = time.time()
            db.build(db_tiles)
            print(f"[{region}] built in {time.time() - t0:.0f}s")
            db.save(cache_dir)

        queries = load_scoped_queries(args.queries_dir, center_lat, center_lon, args.query_dist_km)
        print(f"[{region}] {len(queries)} queries within {args.query_dist_km}km")

        t0 = time.time()
        recalls, num_evaluated = evaluate_retrieval(db, queries, args.k_values, args.iou_threshold)
        print(f"[{region}] retrieval eval ({num_evaluated} queries) in {time.time() - t0:.0f}s: {recalls}")

        coord_queries = queries[: args.coord_eval_limit] if args.coord_eval_limit else queries
        t0 = time.time()
        coord_stats = evaluate_coords(db, coord_queries, args.iou_threshold)
        print(f"[{region}] coord eval ({coord_stats['num_scored']} queries) in {time.time() - t0:.0f}s: {coord_stats}")

        region_result = {"recalls": recalls, "num_evaluated": num_evaluated, "coords": coord_stats}
        all_results[region] = region_result

        if use_wandb:
            import wandb

            wandb.log(
                {
                    f"{region}/R@{k}": v for k, v in recalls.items()
                } | {
                    f"{region}/top1_correct_tile_rate": coord_stats["top1_correct_tile_rate"],
                    f"{region}/median_coord_error_km": coord_stats["median_coord_error_km"] or float("nan"),
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
