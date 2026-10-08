"""Full retrieve-then-MATCH validation on the Argus-format Landsat-8 GeoTIFF
frames (/mnt/sda2/geotiffs), reusing evaluate_argus_geotiffs.py's query
loading and the reference DB it already built/cached, but running the real
LocalizationPipeline (SIFT+LightGlue+homography+Georeferencer) instead of
reporting bare top-1 retrieval centroid distance.

Why this exists: evaluate_argus_geotiffs.py (retrieval-only, matching
AstroLoc's own VINSat methodology) showed median top1-centroid error of
~100-130km even on genuine R@1 hits -- confirmed to be a reference-tile-grid
quantization floor (error scales with tile size: zoom-8 ~500km tiles give
~107km median error on exact hits, zoom-9 ~250km tiles give ~86km), not a
matching-quality problem. This script measures what the SECOND pipeline
stage (which retrieval-only eval skips entirely, same as the paper's own
VINSat eval) actually buys in coordinate precision for this cross-sensor
(Landsat query / Sentinel-2-trained retriever) case.
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

from core.pipeline import LocalizationPipeline
from core.types import PipelineConfig
from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from scripts.evaluate import haversine_km

from astroloc.eval.evaluate_argus_geotiffs import GEOTIFF_ROOT, load_query, sample_frames
from astroloc.models.dinov2_salad import DinoV2SaladModel, DinoV2SaladRetriever


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--db-dir", required=True, help="existing cached reference db dir from evaluate_argus_geotiffs.py")
    ap.add_argument("--zones", nargs="+", default=None)
    ap.add_argument("--per-zone", type=int, default=50)
    ap.add_argument("--mode", default="center", choices=["center", "full"])
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--max-num-keypoints", type=int, default=1024)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--max-ransac-iters", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="/mnt/sdc1/astroloc/reference_db/argus_geotiff_eval")
    args = ap.parse_args()

    zones = args.zones or sorted(d for d in os.listdir(GEOTIFF_ROOT) if os.path.isdir(os.path.join(GEOTIFF_ROOT, d)))

    if args.checkpoint:
        retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device=args.device)
    else:
        retriever = DinoV2SaladRetriever(DinoV2SaladModel(pretrained=True), device=args.device)

    index = FaissFlatIndex(retriever.descriptor_dim)
    print(f"loading cached reference db from {args.db_dir}", flush=True)
    db = ReferenceDatabase.load(args.db_dir, retriever, index)
    print(f"{len(db.tiles)} reference tiles loaded", flush=True)

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=args.max_num_keypoints, img_size=args.img_size,
        max_ransac_iters=args.max_ransac_iters, min_inliers=args.min_inliers, device=args.device,
    )
    georef = Georeferencer()
    pipeline = LocalizationPipeline(
        db, matcher, georef,
        PipelineConfig(top_k=args.top_k, min_inliers=args.min_inliers, query_size=512, max_ransac_iters=args.max_ransac_iters),
    )

    frame_paths = sample_frames(zones, args.per_zone)
    print(f"{len(frame_paths)} frames across {len(zones)} zones, mode={args.mode}", flush=True)

    rows = []
    t0 = time.time()
    for i, path in enumerate(frame_paths):
        loaded = load_query(path, args.mode)
        if loaded is None:
            continue
        query, image, nodata_frac = loaded
        zone = os.path.basename(os.path.dirname(path))
        row = {"frame": query.tile_id, "zone": zone, "nodata_frac": nodata_frac}
        if nodata_frac > 0.5:
            row["status"] = "skipped_nodata"
            rows.append(row)
            continue

        t_q0 = time.time()
        result = pipeline.localize(image)
        elapsed = time.time() - t_q0

        gt = query.corners_latlon.mean(axis=0)
        error_km = None
        if result.status == "fix" and result.query_footprint_latlon is not None:
            est = result.query_footprint_latlon.mean(axis=0)
            error_km = haversine_km(gt[0], gt[1], est[0], est[1])

        row |= {
            "status": result.status,
            "num_inliers": result.confidence,
            "matched_tile_id": result.matched_tile_id,
            "error_km": error_km,
            "elapsed_s": elapsed,
        }
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(frame_paths)} done, elapsed={time.time() - t0:.0f}s", flush=True)

    out_path = os.path.join(args.out_dir, f"rows_matched_{args.run_name}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    scored = [r for r in rows if r.get("status") in ("fix", "no_fix")]
    fixes = [r for r in scored if r["status"] == "fix" and r["error_km"] is not None]
    n_fix = len(fixes)
    fix_rate = 100.0 * n_fix / len(scored) if scored else 0.0
    print(f"\n=== {args.run_name} ({args.mode}): {len(scored)} queries, fix_rate={fix_rate:.1f}% ===")
    if fixes:
        errs = np.array([r["error_km"] for r in fixes])
        print(f"median_error_km={np.median(errs):.2f}  mean_error_km={np.mean(errs):.2f}  n_fix={n_fix}")

    zones_seen = sorted(set(r["zone"] for r in rows))
    print("\nper-zone:")
    for zone in zones_seen:
        zr = [r for r in scored if r["zone"] == zone]
        zf = [r for r in zr if r["status"] == "fix" and r["error_km"] is not None]
        zfix_rate = 100.0 * len(zf) / len(zr) if zr else 0.0
        zmed = np.median([r["error_km"] for r in zf]) if zf else None
        print(f"  {zone:6s} n={len(zr):3d} fix_rate={zfix_rate:5.1f}%  median_err={zmed if zmed is None else round(float(zmed), 1)}km")

    print(f"\nSaved rows to {out_path}")


if __name__ == "__main__":
    main()
