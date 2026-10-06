"""Retrieval validation on Argus-format Landsat-8 GeoTIFF frames
(/mnt/sda2/geotiffs/<MGRS zone>/l8_<zone>_NNNNN.tif), instead of the ISS
astronaut-photo benchmark astroloc/eval/evaluate.py uses.

Each GeoTIFF is one simulated Argus frame: ~4609x2593 px at 175 m/px
(~806x454 km), UTM-projected, bands B4/B3/B2. The 500 files per zone are
500 DIFFERENT shifted footprints within the zone (checked directly -- e.g.
l8_10S_00000 vs l8_10S_00001 bounds differ by ~5 deg lat), so every file is
an independent query with an exact ground-truth footprint from its own
geotransform -- no IoU-labeling heuristics on the query side.

Reference DB is GLOBAL (all 2021 EarthLoc tiles at the given zooms, no
region scoping), the honest onboard case: the satellite has no position
prior. Zoom 8 tiles are ~5.6 deg (~500 km) and zoom 9 ~250 km, so an Argus
frame is already at reference-tile scale; zoom 10 (~125 km) is included
mostly as realistic distractors, it can rarely reach IoU 0.2 with a frame.

Two query modes, since the retriever squashes any input to 224x224:
- "full": the whole 16:9 frame (aspect distorted by the resize)
- "center": the central square crop (frame height x frame height, ~454 km)

Cross-sensor caveat: queries are Landsat-8 composites, the DB is Sentinel-2
cloudless (EOx) -- the model never saw Landsat in training.
"""

import argparse
import glob
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform

from core.types import GeoTile
from data_loading.earthloc_loader import parse_geotile_filename
from database.reference_database import ReferenceDatabase, dedup_search
from index.faiss_index import FaissFlatIndex
from scripts.evaluate import find_positive_tile_ids, footprint_bbox, haversine_km

from astroloc.models.dinov2_salad import DinoV2SaladModel, DinoV2SaladRetriever

GEOTIFF_ROOT = "/mnt/sda2/geotiffs"
# Read frames decimated: the retriever resizes to 224 anyway, so full 4609x2593
# reads (~36MB each) would just be wasted IO.
READ_HEIGHT = 448
MAX_NODATA_FRACTION = 0.5


def load_global_db_tiles(database_dir: str, year: int, zooms: list[str]) -> list[GeoTile]:
    tiles = []
    for zoom in zooms:
        paths = glob.glob(os.path.join(database_dir, f"{year}_{zoom}", "**", "*.jpg"), recursive=True)
        tiles += [parse_geotile_filename(p) for p in paths]
    return tiles


def _corners_latlon(crs, transform, col0: float, row0: float, col1: float, row1: float) -> np.ndarray:
    # BL, TL, TR, BR (core/types.py's GeoTile convention), in pixel (col, row).
    px = [(col0, row1), (col0, row0), (col1, row0), (col1, row1)]
    xs, ys = zip(*[transform * p for p in px])
    lons, lats = warp_transform(crs, "EPSG:4326", list(xs), list(ys))
    return np.array(list(zip(lats, lons)), dtype=np.float64)


def load_query(path: str, mode: str) -> tuple[GeoTile, np.ndarray, float] | None:
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        if mode == "full":
            col0, row0, col1, row1 = 0, 0, w, h
        else:
            col0 = (w - h) // 2
            row0, col1, row1 = 0, col0 + h, h
        window = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        out_w = int(round(READ_HEIGHT * (col1 - col0) / (row1 - row0)))
        arr = src.read(
            [1, 2, 3], window=window, out_shape=(3, READ_HEIGHT, out_w), resampling=Resampling.average
        )
        corners = _corners_latlon(src.crs, src.transform, col0, row0, col1, row1)
    image = np.ascontiguousarray(arr.transpose(1, 2, 0))
    nodata_frac = float((image.sum(axis=2) == 0).mean())
    tile = GeoTile(tile_id=os.path.basename(path), image_path=path, corners_latlon=corners)
    return tile, image, nodata_frac


def sample_frames(zones: list[str], per_zone: int) -> list[str]:
    paths = []
    for zone in zones:
        files = sorted(glob.glob(os.path.join(GEOTIFF_ROOT, zone, "*.tif")))
        if not files:
            continue
        idx = np.linspace(0, len(files) - 1, min(per_zone, len(files))).round().astype(int)
        paths += [files[i] for i in sorted(set(idx))]
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="omit for the off-the-shelf pretrained SALAD baseline")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--database-dir", default="/mnt/sdc1/astroloc/data/database")
    ap.add_argument("--db-year", type=int, default=2021)
    ap.add_argument("--db-zooms", nargs="+", default=["08", "09", "10"])
    ap.add_argument("--zones", nargs="+", default=None, help="default: every zone dir under /mnt/sda2/geotiffs")
    ap.add_argument("--per-zone", type=int, default=50)
    ap.add_argument("--modes", nargs="+", default=["center", "full"], choices=["center", "full"])
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 100])
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-root", default="/mnt/sdc1/astroloc/reference_db/argus_geotiff_eval")
    ap.add_argument("--rebuild-db", action="store_true")
    args = ap.parse_args()

    zones = args.zones or sorted(d for d in os.listdir(GEOTIFF_ROOT) if os.path.isdir(os.path.join(GEOTIFF_ROOT, d)))

    if args.checkpoint:
        retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device=args.device)
    else:
        retriever = DinoV2SaladRetriever(DinoV2SaladModel(pretrained=True), device=args.device)

    db_dir = os.path.join(args.cache_root, "db", f"{args.run_name}_{args.db_year}_z{'-'.join(args.db_zooms)}")
    index = FaissFlatIndex(retriever.descriptor_dim)
    if not args.rebuild_db and os.path.exists(os.path.join(db_dir, "tiles.json")):
        print(f"loading cached global reference db from {db_dir}", flush=True)
        db = ReferenceDatabase.load(db_dir, retriever, index)
    else:
        db_tiles = load_global_db_tiles(args.database_dir, args.db_year, args.db_zooms)
        print(f"building global reference db: {len(db_tiles)} tiles (zooms {args.db_zooms})", flush=True)
        db = ReferenceDatabase(retriever, index)
        t0 = time.time()
        db.build(db_tiles)
        print(f"built in {time.time() - t0:.0f}s", flush=True)
        db.save(db_dir)

    db_tiles = list(db.tiles.values())
    db_bboxes = np.array([footprint_bbox(t) for t in db_tiles])
    frame_paths = sample_frames(zones, args.per_zone)
    print(f"{len(frame_paths)} frames across {len(zones)} zones", flush=True)
    max_k = max(args.k_values)

    results = {"config": vars(args) | {"zones": zones, "num_db_tiles": len(db_tiles)}, "modes": {}}
    for mode in args.modes:
        rows = []
        t0 = time.time()
        for path in frame_paths:
            query, image, nodata_frac = load_query(path, mode)
            zone = os.path.basename(os.path.dirname(path))
            row = {"frame": query.tile_id, "zone": zone, "nodata_frac": nodata_frac}
            if nodata_frac > MAX_NODATA_FRACTION:
                row["status"] = "skipped_nodata"
                rows.append(row)
                continue
            positives = set(find_positive_tile_ids(query, db_tiles, db_bboxes, args.iou_threshold))
            descriptor = retriever.embed(image)
            ranked = dedup_search(db.index, descriptor, max_k)
            ranked_ids = [tile_id for tile_id, _ in ranked]
            top1 = db.tiles[ranked_ids[0]]
            pred = top1.corners_latlon.mean(axis=0)
            gt = query.corners_latlon.mean(axis=0)
            first_hit = next((i + 1 for i, t in enumerate(ranked_ids) if t in positives), None)
            row |= {
                "status": "scored" if positives else "no_positives",
                "num_positives": len(positives),
                "first_hit_rank": first_hit,
                "top1_tile": ranked_ids[0],
                "top1_similarity": float(ranked[0][1]),
                "top1_center_error_km": haversine_km(gt[0], gt[1], pred[0], pred[1]),
            }
            rows.append(row)
        print(f"[{mode}] {len(rows)} frames in {time.time() - t0:.0f}s", flush=True)

        def summarize(subset: list[dict]) -> dict:
            scored = [r for r in subset if r["status"] == "scored"]
            errs = np.array([r["top1_center_error_km"] for r in subset if "top1_center_error_km" in r])
            out = {
                "num_frames": len(subset),
                "num_scored": len(scored),
                "num_skipped_nodata": sum(r["status"] == "skipped_nodata" for r in subset),
                "num_no_positives": sum(r["status"] == "no_positives" for r in subset),
            }
            if scored:
                out["recalls"] = {
                    str(k): 100.0 * sum(r["first_hit_rank"] is not None and r["first_hit_rank"] <= k for r in scored) / len(scored)
                    for k in args.k_values
                }
            if len(errs):
                out["median_top1_error_km"] = float(np.median(errs))
                out["pct_top1_within_250km"] = float(100.0 * (errs < 250).mean())
                out["pct_top1_within_500km"] = float(100.0 * (errs < 500).mean())
            return out

        mode_result = {"overall": summarize(rows), "per_zone": {z: summarize([r for r in rows if r["zone"] == z]) for z in zones}}
        results["modes"][mode] = mode_result
        print(f"[{mode}] overall: {json.dumps(mode_result['overall'])}", flush=True)
        with open(os.path.join(args.cache_root, f"rows_{args.run_name}_{mode}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    out_path = os.path.join(args.cache_root, f"results_{args.run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
