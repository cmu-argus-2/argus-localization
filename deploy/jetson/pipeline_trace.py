"""Record every intermediate result of the localization flow for a few photos, so it can be drawn.

Runs the same flow as e2e_match.py (TensorRT retriever, search, repo matcher with stored
tile features, georeferencer) and saves what each stage produced. visualize_trace.py
turns the output into figures.

Run on the Jetson from ~/Desktop/astroloc/repo:
    PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../pipeline_trace.py --out ../trace

Picks two photos from match_results_pre.json: a typical success (correct tile, error
closest to the median) and the worst confidently wrong fix.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_match import haversine_km, preprocess, trt_embedder  # noqa: E402
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from georeference.georeferencer import Georeferencer  # noqa: E402
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher  # noqa: E402


def pick_photos(records):
    fixes = [r for r in records if r["fix"]]
    median = float(np.median([r["match_error_km"] for r in fixes]))
    good = [r for r in fixes if r["matched_tile_correct"] and r["retrieval_top1_correct"] and r["excluded_from_training"]]
    success = min(good, key=lambda r: abs(r["match_error_km"] - median))
    failure = max(fixes, key=lambda r: r["match_error_km"])
    return [("success", success["file"]), ("failure", failure["file"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../e2e_alps_full")
    ap.add_argument("--tiles-dir", default="../tiles/2021_09")
    ap.add_argument("--tile-feats", default="../tile_feats_2021_09")
    ap.add_argument("--engine", default="../astroloc_full_fp32.trt")
    ap.add_argument("--results", default="match_results_pre.json")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = np.load(os.path.join(args.data, "db_vecs.npy"))
    meta = json.load(open(os.path.join(args.data, "db_tiles.json")))
    queries = {q["file"]: q for q in json.load(open(os.path.join(args.data, "queries.json")))}
    records = json.load(open(os.path.join(args.data, args.results)))["records"]
    by_id = {t.tile_id: t for t in map(parse_geotile_filename,
                                       glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))}
    tiles = [by_id[m["tile_id"]] for m in meta]

    embed = trt_embedder(args.engine)
    matcher = SiftLightGlueMatcher(min_inliers=args.min_inliers)
    georef = Georeferencer()

    for label, fname in pick_photos(records):
        out = os.path.join(args.out, label)
        os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
        qm = queries[fname]
        path = os.path.join(args.data, "queries", fname)
        truth = parse_geotile_filename(path)
        shutil.copy(path, os.path.join(out, "photo.jpg"))
        img = Image.open(path).convert("RGB")
        photo = np.array(img)
        ms = {}

        # 1-2. preprocess + retriever
        t = time.perf_counter(); x = preprocess(img); ms["preprocess"] = time.perf_counter() - t
        t = time.perf_counter(); v = embed(x); ms["retriever (TensorRT)"] = time.perf_counter() - t
        # 3. search: best of 4 rotations per tile, keep top-k
        t = time.perf_counter()
        sims = db @ v
        best_rot, score = sims.argmax(axis=1), sims.max(axis=1)
        ranking = np.argsort(-score)
        cand = ranking[: args.top_k]
        ms["search"] = time.perf_counter() - t

        # 4. photo SIFT
        t = time.perf_counter(); pf = matcher._extract(photo); torch.cuda.synchronize(); ms["SIFT on photo"] = time.perf_counter() - t
        extract = matcher._extract
        matcher._extract = lambda im, _p=photo, _f=pf: _f if im is _p else extract(im)

        # 5-6. LightGlue + MAGSAC per candidate
        cands, best, best_i, t_match = [], None, None, 0.0
        for rank, idx in enumerate(cand):
            tile = tiles[idx]
            stored = np.load(os.path.join(args.tile_feats, tile.tile_id + ".npz"))
            shape = tuple(int(s) for s in stored["tile_shape"])
            matcher._tile_feats_cache[tile.tile_id] = {
                k: torch.from_numpy(stored[k]).to(matcher.device) for k in stored.files if k != "tile_shape"}
            t = time.perf_counter()
            m = matcher.match(photo, None, tile_id=tile.tile_id)
            torch.cuda.synchronize(); t_match += time.perf_counter() - t
            matcher._tile_feats_cache.clear()
            shutil.copy(tile.image_path, os.path.join(out, "tiles", f"{rank:02d}.jpg"))
            centre = tile.corners_latlon.mean(axis=0)
            cands.append({
                "rank": rank, "tile_id": tile.tile_id, "score": float(score[idx]), "rotation_deg": int(best_rot[idx]) * 90,
                "correct": int(idx) in set(qm["positives"]), "km_from_truth": haversine_km(*qm["centre"], *centre),
                "raw_matches": int(m.query_pts.shape[0]), "inliers": int(m.num_inliers), "tile_shape": shape,
                "tile_keypoints": int(stored["keypoints"].shape[1]),
            })
            if best is None or m.num_inliers > best.num_inliers:
                best, best_i, best_stored = m, rank, stored
        matcher._extract = extract
        ms["LightGlue + MAGSAC, 15 tiles"] = t_match

        # 7-8. fix decision, footprint, tie points
        btile = tiles[cand[best_i]]
        bshape = cands[best_i]["tile_shape"]
        fix = best.num_inliers >= args.min_inliers
        footprint = georef.estimate_query_footprint(best, btile, photo.shape, bshape) if fix else None
        ties = georef.make_tie_points(best, btile, bshape) if fix else []
        with open(os.path.join(out, "tie_points.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["photo_u_px", "photo_v_px", "lat_deg", "lon_deg"])
            for tp in ties:
                w.writerow([round(tp.u, 2), round(tp.v, 2), round(tp.lat, 6), round(tp.lon, 6)])

        np.savez(os.path.join(out, "trace.npz"),
                 preprocessed=x, fingerprint=v, all_scores=score,
                 photo_kp=pf["keypoints"][0].cpu().numpy(), photo_scales=pf["scales"][0].cpu().numpy(),
                 tile_kp=best_stored["keypoints"][0], tile_scales=best_stored["scales"][0],
                 query_pts=best.query_pts, tile_pts=best.tile_pts, inlier_mask=best.inlier_mask,
                 homography=best.homography if best.homography is not None else np.zeros((3, 3)),
                 footprint=footprint if footprint is not None else np.zeros((4, 2)),
                 truth_corners=truth.corners_latlon, tile_corners=btile.corners_latlon)
        est_centre = footprint.mean(axis=0) if footprint is not None else None
        summary = {
            "label": label, "photo": fname, "photo_shape": list(photo.shape),
            "truth_centre": qm["centre"], "nadir": [truth.meta.get("nadir_lat"), truth.meta.get("nadir_lon")] if hasattr(truth, "meta") else None,
            "fingerprint_dim": int(v.shape[0]), "fingerprint_norm": float(np.linalg.norm(v)),
            "db_tiles": int(db.shape[0]), "candidates": cands, "best_rank": best_i,
            "photo_keypoints": int(pf["keypoints"].shape[1]), "fix": bool(fix), "min_inliers": args.min_inliers,
            "estimated_centre": est_centre.tolist() if est_centre is not None else None,
            "error_km": haversine_km(*qm["centre"], *est_centre) if est_centre is not None else None,
            "tie_points": len(ties), "stage_seconds": {k: round(s, 4) for k, s in ms.items()},
        }
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        print(f"{label}: {fname[:60]}... fix={fix} best rank {best_i} inliers {best.num_inliers} "
              f"error {summary['error_km']} km, {len(ties)} tie points | " +
              ", ".join(f"{k} {1e3 * s:.0f} ms" for k, s in ms.items()))


if __name__ == "__main__":
    main()
