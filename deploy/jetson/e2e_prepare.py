"""End-to-end localization test, WORKSTATION half.

Builds the deployment-style reference database (whole Earth, zoom 9, one year,
4 rotations per tile, embedded with the original PyTorch nano model), picks
test photos from one held-out region that have a correct tile in that database,
and records the workstation's own localization results for them.

Everything the Jetson needs is written to --out, so it can be copied over and
the Jetson (e2e_jetson.py) can repeat the exact same test with the TensorRT
engine.

Run from the argus-localization repo root:
    ~/venv-export/bin/python deploy/jetson/e2e_prepare.py --out ~/e2e_alps

Output (~200 MB):
    db_vecs.npy        (N_tiles, 4, 512) float32 fingerprints, 4 rotations each
    db_tiles.json      tile id + centre lat/lon, same order as db_vecs
    queries/           the selected test photos (copied)
    queries.json       per photo: filename, true centre, indices of correct tiles
    ws_query_vecs.npy  workstation (PyTorch) fingerprint of each photo
    ws_results.json    workstation metrics + per-photo top-1 tile
"""

import argparse
import glob
import json
import math
import os
import resource
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, os.getcwd())
from astroloc.data.regions import EVAL_REGIONS  # noqa: E402
from astroloc.models.dinov2_salad import DinoV2SaladRetriever  # noqa: E402
from data_loading.earthloc_loader import load_query_set, parse_geotile_filename  # noqa: E402

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
K_VALUES = (1, 5, 10, 100)


# ---- preprocessing: must stay byte-identical to e2e_jetson.py ----
def preprocess(img: Image.Image, rot_k: int = 0) -> np.ndarray:
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    if rot_k:
        x = np.rot90(x, rot_k, axes=(0, 1))
    x = x.transpose(2, 0, 1)
    return ((x - MEAN) / STD).astype(np.float32)


# ---- geometry ----
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def polygon(tile):
    # corners are (lat, lon); shapely wants (x=lon, y=lat). convex_hull tolerates corner order.
    return Polygon([(lon, lat) for lat, lon in tile.corners_latlon]).convex_hull


def bbox(tile):
    lat, lon = tile.corners_latlon[:, 0], tile.corners_latlon[:, 1]
    return lat.min(), lat.max(), lon.min(), lon.max()


def positive_indices(query, db_polys, db_bboxes, iou_threshold):
    qa, qb, qc, qd = bbox(query)
    mask = (db_bboxes[:, 2] <= qd) & (db_bboxes[:, 3] >= qc) & (db_bboxes[:, 0] <= qb) & (db_bboxes[:, 1] >= qa)
    qp = polygon(query)
    out = []
    for i in np.nonzero(mask)[0]:
        inter = qp.intersection(db_polys[i]).area
        union = qp.union(db_polys[i]).area
        if union > 0 and inter / union >= iou_threshold:
            out.append(int(i))
    return out


# ---- search + scoring: must stay identical to e2e_jetson.py ----
def rank_tiles(db_vecs: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.argsort(-(db_vecs @ q).max(axis=1))


def score(rankings, queries_meta, db_centres):
    hits = {k: 0 for k in K_VALUES}
    errors = []
    for ranking, q in zip(rankings, queries_meta):
        pos = set(q["positives"])
        for k in K_VALUES:
            if pos.intersection(ranking[:k].tolist()):
                hits[k] += 1
        top1 = db_centres[ranking[0]]
        errors.append(haversine_km(q["centre"][0], q["centre"][1], top1[0], top1[1]))
    n = len(queries_meta)
    return {
        "num_queries": n,
        **{f"R@{k}": round(100.0 * hits[k] / n, 2) for k in K_VALUES},
        "median_error_km": round(float(np.median(errors)), 1),
        "mean_error_km": round(float(np.mean(errors)), 1),
    }


def load_retriever(checkpoint_path, device="cuda"):
    """DinoV2SaladRetriever.from_checkpoint, plus a fix for LoRA checkpoints saved
    before commit cc75c2b (Sept 8). That commit wraps each DINOv2 block in
    _CheckpointedBlock, which adds a ".block." level to the weight names, so older
    checkpoints (e.g. nano_train/checkpoints_v2, Aug 20) no longer load. The weights
    are unchanged; only the names are remapped. Loading stays strict."""
    import re
    import torch
    from astroloc.models.dinov2_salad import REDUCED_DIM, DinoV2SaladModel, DinoV2SaladRetriever

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DinoV2SaladModel(
        pretrained=False,
        use_lora=state.get("use_lora", False),
        lora_r=state.get("lora_r", 8),
        lora_alpha=state.get("lora_alpha", 16),
        backbone_name=state.get("backbone_name", "dinov2_vitb14"),
        reduced_dim=state.get("reduced_dim", REDUCED_DIM),
    )
    weights = state["model"] if "model" in state else state
    expected = set(model.state_dict())
    if set(weights) != expected:
        remapped = {re.sub(r"^(backbone\.model\.blocks\.\d+\.)(?!block\.)", r"\1block.", k): v
                    for k, v in weights.items()}
        if set(remapped) == expected:
            print("  note: remapped pre-Sept-8 checkpoint weight names (added '.block.')")
            weights = remapped
    model.load_state_dict(weights)
    return DinoV2SaladRetriever(model, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/sdc1/astroloc/reference_db/nano_train/checkpoints/final.pt")
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--zoom", default="09")
    ap.add_argument("--region", default="Alps", choices=list(EVAL_REGIONS))
    ap.add_argument("--query-dist-km", type=float, default=2500)
    ap.add_argument("--iou", type=float, default=0.2)
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32, help="tiles per GPU batch (x4 rotations)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "queries"), exist_ok=True)

    # 1. reference tiles: whole Earth, one zoom level, one year
    tile_dir = os.path.join(args.database_dir, f"{args.year}_{args.zoom}")
    paths = sorted(glob.glob(os.path.join(tile_dir, "**", "*.jpg"), recursive=True))
    tiles = [parse_geotile_filename(p) for p in paths]
    print(f"reference tiles in {tile_dir}: {len(tiles)}")
    if not tiles:
        print("ERROR: no tiles found")
        return 1

    # 2. test photos: near the held-out region, with at least one correct tile in this database
    c_lat, c_lon = EVAL_REGIONS[args.region]
    all_q = load_query_set(args.queries_dir)
    near = sorted(
        (q for q in all_q if haversine_km(c_lat, c_lon, q.meta["nadir_lat"], q.meta["nadir_lon"]) < args.query_dist_km),
        key=lambda q: os.path.basename(q.image_path),
    )
    db_polys = [polygon(t) for t in tiles]
    db_bboxes = np.array([bbox(t) for t in tiles])
    selected = []
    for q in near:
        pos = positive_indices(q, db_polys, db_bboxes, args.iou)
        if pos:
            selected.append((q, pos))
        if len(selected) >= args.max_queries:
            break
    print(f"photos within {args.query_dist_km:.0f} km of {args.region}: {len(near)}; "
          f"with a correct zoom-{args.zoom} tile: using {len(selected)}")
    if not selected:
        print("ERROR: no usable test photos")
        return 1

    # 3. model
    retriever = load_retriever(args.checkpoint, device="cuda")
    model = retriever.model.eval()
    dim = retriever.descriptor_dim
    print(f"model: descriptor dim {dim}")

    def embed(batch_arrays):
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch_arrays)).cuda()
            return model(x).float().cpu().numpy()

    # 4. embed the database, 4 rotations per tile
    t0 = time.time()
    db_vecs = np.empty((len(tiles), 4, dim), dtype=np.float32)
    for start in range(0, len(tiles), args.batch):
        chunk = tiles[start:start + args.batch]
        arrays = []
        for t in chunk:
            img = Image.open(t.image_path).convert("RGB")
            arrays += [preprocess(img, k) for k in range(4)]
        db_vecs[start:start + len(chunk)] = embed(arrays).reshape(len(chunk), 4, dim)
        if (start // args.batch) % 50 == 0:
            print(f"  embedded {start + len(chunk)}/{len(tiles)} tiles")
    print(f"database embedded in {time.time() - t0:.0f}s -> {db_vecs.nbytes / 1e6:.1f} MB")

    db_centres = np.array([t.corners_latlon.mean(axis=0) for t in tiles])
    np.save(os.path.join(out, "db_vecs.npy"), db_vecs)
    with open(os.path.join(out, "db_tiles.json"), "w") as f:
        json.dump([{"tile_id": t.tile_id, "centre": c.tolist()} for t, c in zip(tiles, db_centres)], f)

    # 5. copy photos and record ground truth
    queries_meta = []
    for q, pos in selected:
        name = os.path.basename(q.image_path)
        shutil.copy(q.image_path, os.path.join(out, "queries", name))
        queries_meta.append({"file": name, "centre": q.corners_latlon.mean(axis=0).tolist(), "positives": pos})
    with open(os.path.join(out, "queries.json"), "w") as f:
        json.dump(queries_meta, f)

    # 6. workstation localization, same preprocessing and search as the Jetson
    q_vecs, rankings, embed_ms, search_ms = [], [], [], []
    for qm in queries_meta:
        img = Image.open(os.path.join(out, "queries", qm["file"])).convert("RGB")
        t0 = time.perf_counter()
        v = embed([preprocess(img)])[0]
        t1 = time.perf_counter()
        r = rank_tiles(db_vecs, v)
        t2 = time.perf_counter()
        q_vecs.append(v)
        rankings.append(r)
        embed_ms.append((t1 - t0) * 1e3)
        search_ms.append((t2 - t1) * 1e3)
    np.save(os.path.join(out, "ws_query_vecs.npy"), np.stack(q_vecs))

    metrics = score(rankings, queries_meta, db_centres)
    metrics.update({
        "machine": "workstation (PyTorch, RTX 4090)",
        "num_db_tiles": len(tiles),
        "db_mb": round(db_vecs.nbytes / 1e6, 1),
        "embed_ms_median": round(float(np.median(embed_ms)), 2),
        "search_ms_median": round(float(np.median(search_ms)), 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
    })
    with open(os.path.join(out, "ws_results.json"), "w") as f:
        json.dump({"metrics": metrics, "top1": [int(r[0]) for r in rankings]}, f, indent=2)

    print("\n=== WORKSTATION RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v}")
    print(f"\nwrote everything to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
