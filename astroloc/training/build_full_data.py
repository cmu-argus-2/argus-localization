"""Builds the shared, paper-scale training data for the controlled AstroLoc
comparison (LoRA+static vs full-FT+dynamic, see repo memory
astroloc_target_architecture). Both runs load this exact same cache, so any
difference between them is only their fine-tune method and clustering mode,
not the underlying data.

Query pool: whole GAPE+EarthLoc pool minus the 6 held-out eval regions
(331,710 eligible, measured 2026-08-26 after the full 773k GAPE download).
Reference tiles: zoom 8-12, worldwide, minus eval regions (391,890 eligible,
measured) -- the paper's own zoom range. Ours is smaller than the paper's
5.3M tiles because that's everything locally available, not a scoping choice.
Positive pairs: IoU >= 0.2, reusing scripts/evaluate.py's IoU machinery via
astroloc/data/pairing.py. Split 90/5/5 train/val/test, seeded. Initial k=50
clustering on the train split's unique tiles only (embedding with the
pretrained checkpoint), matching build_training_data()'s existing convention.
"""

import argparse
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from astroloc.data.pairing import build_positive_pairs
from astroloc.models.dinov2_salad import DinoV2SaladModel, DinoV2SaladRetriever
from astroloc.training.cluster import embed_tiles, kmeans_cluster
from nano.data import build_query_set_all, build_reference_tiles_multi_zoom

CACHE_PATH = "/mnt/sdc1/astroloc/reference_db/astroloc_train/cache/full_pairs_cache.pkl"
ZOOMS_FAITHFUL = ("08", "09", "10", "11", "12")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--earthloc-queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument(
        "--gape-queries-dir",
        default="/mnt/sdc1/astroloc/reference_db/astroloc_train/gape_queries",
    )
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--num-clusters", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=CACHE_PATH)
    ap.add_argument("--max-queries", type=int, default=None, help="debug: cap query pool for a fast smoke test")
    ap.add_argument("--max-tiles", type=int, default=None, help="debug: cap tile pool for a fast smoke test")
    args = ap.parse_args()

    print("Building query pool (whole pool minus eval regions)...", flush=True)
    t0 = time.time()
    queries = build_query_set_all(args.earthloc_queries_dir, args.gape_queries_dir)
    if args.max_queries:
        random.Random(args.seed).shuffle(queries)
        queries = queries[: args.max_queries]
    print(f"{len(queries)} eligible queries in {time.time() - t0:.0f}s", flush=True)

    print("Building reference tile pool (zoom 8-12, worldwide, minus eval regions)...", flush=True)
    t0 = time.time()
    tiles = build_reference_tiles_multi_zoom(args.database_dir, zooms=ZOOMS_FAITHFUL)
    if args.max_tiles:
        random.Random(args.seed).shuffle(tiles)
        tiles = tiles[: args.max_tiles]
    print(f"{len(tiles)} eligible tiles in {time.time() - t0:.0f}s", flush=True)

    print(f"Building positive pairs (IoU >= {args.iou_threshold})... this is the long step", flush=True)
    t0 = time.time()
    pairs = build_positive_pairs(queries, tiles, iou_threshold=args.iou_threshold)
    print(f"{len(pairs)}/{len(queries)} queries paired in {time.time() - t0:.0f}s", flush=True)

    rng = random.Random(args.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    val_pairs = shuffled[:n_val]
    test_pairs = shuffled[n_val : n_val + n_test]
    train_pairs = shuffled[n_val + n_test :]
    print(f"split: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test", flush=True)

    # Cluster over the FULL eligible tile pool, not just the tiles that happen
    # to have won a query match (only ~52,844/391,890, ~13.5%, in the prior
    # run -- most reference tiles never get photographed by an astronaut, but
    # L_MUM's job is embedding-space geometry over the reference DB itself,
    # same as the paper's 5.3M-tile clustering pool, so it should see that
    # full breadth rather than only query-adjacent tiles.
    print(f"Embedding ALL {len(tiles)} eligible reference tiles with the pretrained checkpoint for clustering...", flush=True)
    pretrained_model = DinoV2SaladModel(pretrained=True)
    retriever = DinoV2SaladRetriever(pretrained_model, device=args.device)
    tile_embeddings = embed_tiles(retriever, tiles)
    _, tile_cluster_ids = kmeans_cluster(tile_embeddings, k=args.num_clusters)
    tile_id_to_cluster = {t.tile_id: int(c) for t, c in zip(tiles, tile_cluster_ids)}
    cluster_ids = [tile_id_to_cluster[tile.tile_id] for _, tile in train_pairs]
    del pretrained_model, retriever
    torch.cuda.empty_cache()

    data = {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "cluster_ids": cluster_ids,  # per TRAIN pair, matched tile's cluster id
        "all_tiles": tiles,  # full eligible tile pool, for L_MUM quadruplets/clustering
        "all_tile_cluster_ids": [tile_id_to_cluster[t.tile_id] for t in tiles],
        "num_clusters": args.num_clusters,
        "num_eligible_queries": len(queries),
        "num_eligible_tiles": len(tiles),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
