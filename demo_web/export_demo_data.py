"""Exports a static, precomputed bundle of REAL pipeline outputs for the public
web demo (demo_web/site/), so the demo can be hosted on GitHub Pages without
exposing the lab machine or shipping the ~100GB reference database.

For a seeded sample of astronaut photos in each of the 6 benchmark regions it
runs exactly what core/pipeline.py::LocalizationPipeline.localize does
(retrieve top_k -> SIFT-LightGlue match every candidate -> keep max-inlier ->
threshold min_inliers -> georeference), but keeps what the pipeline normally
throws away so the site can replay it: every candidate's inlier count and
correspondences, per-stage wall-clock timings, and the estimated footprint.

Nothing in the output is simulated: the site only animates these numbers.

No images are written. The site loads each astronaut photo straight from NASA
GAPE (EarthLoc queries are GAPE's full frame resized to a square -- verified
pixel-identical, corr 1.00 -- so the normalized correspondences below map onto
GAPE's own image unchanged), and rebuilds each reference tile in the browser
from EOx's public Sentinel-2 cloudless 2021 WMTS: tile_id "ZZ_YYYY_XXXX@2021"
is exactly a 4x4 block of web-mercator tiles at zoom ZZ starting at (x, y)
(verified against every candidate's corners, and corr 0.993 vs the on-disk
tile). The published bundle is therefore JSON only.

Run (GPU recommended, ~2s/query):
    python demo_web/export_demo_data.py --device cuda:0 --n-per-region 80
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from PIL import Image

from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from scripts.evaluate import REGIONS, haversine_km, load_scoped_queries

from astroloc.models.dinov2_salad import DinoV2SaladRetriever

ASTROLOC_ROOT = "/mnt/sdc1/astroloc/reference_db/astroloc_train"
MODEL_NAME = "faithful_lora_v2"
CHECKPOINT = f"{ASTROLOC_ROOT}/checkpoints_faithful_lora_v2/final.pt"
DB_CACHE_ROOT = f"{ASTROLOC_ROOT}/eval_cache/faithful_lora_v2"
RESULTS_JSON = f"{ASTROLOC_ROOT}/eval_cache/results_faithful_lora_v2.json"
ERROR_DATASET = "output/error_dataset/faithful_lora_v2.jsonl"

MAX_LINES = 90  # inlier correspondences kept per candidate, for drawing match lines


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _r(x, nd=4):
    return round(float(x), nd)


def _norm_pts(pts: np.ndarray, shape) -> list:
    h, w = shape[:2]
    return [[_r(u / w, 4), _r(v / h, 4)] for u, v in pts]


def _mission(image_id: str) -> str:
    # "ISS047-E-55412" -> "ISS Expedition 47"
    head = image_id.split("-")[0]
    if head.startswith("ISS") and head[3:].isdigit():
        return f"ISS Expedition {int(head[3:])}"
    return head


def localize_and_record(query, db, matcher, georef, top_k, min_inliers, device):
    frame = np.array(Image.open(query.image_path).convert("RGB"))

    _sync(device)
    t0 = time.perf_counter()
    descriptor = db.retriever.embed(frame)
    _sync(device)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    from database.reference_database import dedup_search

    ranked = dedup_search(db.index, descriptor, top_k)
    candidates = [(db.tiles[tid], sim) for tid, sim in ranked]
    t_search = time.perf_counter() - t0

    cands = []
    best = None
    for rank, (tile, sim) in enumerate(candidates):
        tile_image = np.array(Image.open(tile.image_path).convert("RGB"))
        _sync(device)
        t0 = time.perf_counter()
        m = matcher.match(frame, tile_image, tile_id=tile.tile_id)
        _sync(device)
        dt = time.perf_counter() - t0
        cands.append({"tile": tile, "sim": float(sim), "match": m, "tile_image": tile_image, "ms": dt * 1000})
        if best is None or m.num_inliers > cands[best]["match"].num_inliers:
            best = rank

    return frame, cands, best, {"embed_ms": t_embed * 1000, "search_ms": t_search * 1000}


def export_query(idx, query, region, frame, cands, best, timings, georef, min_inliers, out_dir):
    gt_corners = query.corners_latlon
    gt_center = gt_corners.mean(axis=0)
    best_c = cands[best]
    status = "fix" if best_c["match"].num_inliers >= min_inliers else "no_fix"

    pred_center, pred_corners, error_km = None, None, None
    if status == "fix":
        fp = georef.estimate_query_footprint(best_c["match"], best_c["tile"], frame.shape, best_c["tile_image"].shape)
        if fp is not None:
            pred_corners = fp
            pred_center = fp.mean(axis=0)
            error_km = haversine_km(gt_center[0], gt_center[1], pred_center[0], pred_center[1])
        else:
            status = "no_fix"

    cand_out = []
    rng = np.random.default_rng(idx)
    for i, c in enumerate(cands):
        m, tile = c["match"], c["tile"]
        lines = []
        if m.num_inliers > 0:
            ii = np.flatnonzero(m.inlier_mask)
            if len(ii) > MAX_LINES:
                ii = np.sort(rng.choice(ii, MAX_LINES, replace=False))
            q = _norm_pts(m.query_pts[ii], frame.shape)
            t = _norm_pts(m.tile_pts[ii], c["tile_image"].shape)
            lines = [qq + tt for qq, tt in zip(q, t)]
        center = tile.corners_latlon.mean(axis=0)
        zyx, year = tile.tile_id.split("@")
        z, y, x = (int(v) for v in zyx.split("_"))
        cand_out.append(
            {
                "tile_id": tile.tile_id,
                "wmts": {"z": z, "y": y, "x": x, "year": year},  # 4x4 web-mercator tiles at zoom z
                "sim": _r(c["sim"]),
                "inliers": int(m.num_inliers),
                "matches": int(len(m.query_pts)),
                "match_ms": _r(c["ms"], 1),
                "center": [_r(center[0]), _r(center[1])],
                "corners": [[_r(a), _r(b)] for a, b in tile.corners_latlon],
                "dist_to_truth_km": _r(haversine_km(gt_center[0], gt_center[1], center[0], center[1]), 1),
                "lines": lines,
            }
        )

    image_id = query.tile_id.split("@")[0]
    mission, roll, frame_no = image_id.split("-")
    ts = query.timestamp or ""
    detail = {
        "id": idx,
        "region": region,
        "photo_id": image_id,
        "mission": _mission(image_id),
        "gape_img": f"https://eol.jsc.nasa.gov/DatabaseImages/ESC/small/{mission}/{image_id}.JPG",
        "gape_page": f"https://eol.jsc.nasa.gov/SearchPhotos/photo.pl?mission={mission}&roll={roll}&frame={frame_no}",
        "date": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) == 8 else ts,
        "nadir": [_r(query.meta["nadir_lat"]), _r(query.meta["nadir_lon"])],
        "area_km2": int(query.meta["sq_km_area"]),
        "truth_center": [_r(gt_center[0]), _r(gt_center[1])],
        "truth_corners": [[_r(a), _r(b)] for a, b in gt_corners],
        "status": status,
        "best": best,
        "min_inliers": min_inliers,
        "pred_center": None if pred_center is None else [_r(pred_center[0], 5), _r(pred_center[1], 5)],
        "pred_corners": None if pred_corners is None else [[_r(a), _r(b)] for a, b in pred_corners],
        "error_km": None if error_km is None else _r(error_km, 2),
        "timings": {k: _r(v, 1) for k, v in timings.items()},
        "candidates": cand_out,
    }
    with open(os.path.join(out_dir, "q", f"{idx}.json"), "w") as f:
        json.dump(detail, f, separators=(",", ":"))
    return detail


def aggregate_stats() -> dict:
    stats = {"model": MODEL_NAME}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            res = json.load(f)
        stats["recall"] = {
            r: {k: _r(v["recalls"][k], 1) for k in ("1", "5", "10", "100")} for r, v in res.items()
        }
    if os.path.exists(ERROR_DATASET):
        rows = [json.loads(l) for l in open(ERROR_DATASET)]
        errs = [r["error_km"] for r in rows if r["status"] == "fix" and r["error_km"] is not None]
        stats["full_pipeline"] = {
            "n_queries": len(rows),
            "fix_rate": _r(100.0 * sum(r["status"] == "fix" for r in rows) / len(rows), 1),
            "median_error_km": _r(float(np.median(errs)), 2),
            "p90_error_km": _r(float(np.percentile(errs, 90)), 2),
            "median_latency_s": _r(float(np.median([r["elapsed_s"] for r in rows])), 2),
        }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="+", default=list(REGIONS.keys()))
    ap.add_argument("--n-per-region", type=int, default=90)
    ap.add_argument("--nofix-keep-frac", type=float, default=0.3,
                    help="fraction of no-fix results kept in the gallery (fixes are all kept)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--user-config", default="user_config.yaml")
    ap.add_argument("--out", default="demo_web/site/data")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)
    top_k = config["pipeline"]["top_k"]
    min_inliers = config["pipeline"]["min_inliers"]

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.join(args.out, "q"))

    retriever = DinoV2SaladRetriever.from_checkpoint(CHECKPOINT, device=args.device)
    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=min_inliers,
        device=args.device,
    )
    georef = Georeferencer()

    manifest, seen, idx = [], set(), 0
    rng = random.Random(args.seed)
    db_sizes = {}
    for region in args.regions:
        center_lat, center_lon = REGIONS[region]
        db = ReferenceDatabase.load(
            os.path.join(DB_CACHE_ROOT, region.replace(" ", "_")), retriever, FaissFlatIndex(retriever.descriptor_dim)
        )
        db_sizes[region] = len(db.tiles)
        matcher._tile_feats_cache.clear()  # per-region tiles never repeat across regions; bound GPU memory
        queries = load_scoped_queries(user_config["queries_dir"], center_lat, center_lon, config["eval"]["query_dist_km"])
        queries = [q for q in queries if q.tile_id not in seen]
        sample = rng.sample(queries, min(args.n_per_region, len(queries)))
        logging.info(f"[{region}] {len(db.tiles)} db tiles, running {len(sample)} of {len(queries)} queries")

        n_fix = n_kept = 0
        for q in sample:
            seen.add(q.tile_id)
            try:
                frame, cands, best, timings = localize_and_record(q, db, matcher, georef, top_k, min_inliers, args.device)
            except Exception as e:  # see collect_error_dataset.py: degenerate tiles can crash the SIFT extractor
                logging.warning(f"[{region}] {q.tile_id}: {type(e).__name__}: {e}")
                continue
            is_fix = cands[best]["match"].num_inliers >= min_inliers
            n_fix += is_fix
            if not is_fix and rng.random() > args.nofix_keep_frac:
                continue
            d = export_query(idx, q, region, frame, cands, best, timings, georef, min_inliers, args.out)
            manifest.append(
                {
                    "id": idx,
                    "lat": d["truth_center"][0],
                    "lon": d["truth_center"][1],
                    "region": region,
                    "status": d["status"],
                    "err": d["error_km"],
                }
            )
            idx += 1
            n_kept += 1
        logging.info(f"[{region}] fix rate {100 * n_fix / max(len(sample), 1):.1f}%, kept {n_kept} in gallery")
        del db
        torch.cuda.empty_cache()

    stats = aggregate_stats()
    stats["db_tiles"] = db_sizes
    stats["top_k"] = top_k
    stats["min_inliers"] = min_inliers
    stats["regions"] = {r: list(REGIONS[r]) for r in args.regions}
    stats["region_radius_km"] = config["eval"]["query_dist_km"]
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"stats": stats, "photos": manifest}, f, separators=(",", ":"))
    logging.info(f"Wrote {len(manifest)} photos to {args.out}")


if __name__ == "__main__":
    main()
