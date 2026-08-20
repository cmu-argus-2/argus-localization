"""Two-frame simulated OD, end to end: localize a real query image (frame 1),
simulate a plausible next ground point via two-body dynamics (frame 2),
snap it to a real reference-DB tile, then solve for the connecting orbital
velocity. See integration/orbit_simulator.py's docstring for why this is a
round-trip self-consistency check (not independent-measurement recovery) and
what it does and doesn't validate.

Model-agnostic on purpose ("easy to call on the other models"): takes a
retriever + reference-db cache dir, exactly like scripts/evaluate_astroloc_matched.py,
so the same function runs against any of astroloc/'s checkpoints or nano's.
"""

import argparse
import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core.path_setup import ensure_repo_root_first

ensure_repo_root_first(_REPO_ROOT)
os.environ.setdefault("TORCH_HOME", "/mnt/sdc1/astroloc/reference_db/astroloc_train/cache/torch")

import numpy as np
import yaml
from PIL import Image

from brahe.epoch import Epoch

from core.types import GeoTile
from database.reference_database import ReferenceDatabase
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from scripts.evaluate import REGIONS, load_scoped_queries

from integration.orbit_simulator import (
    ISS_ALTITUDE_KM,
    nearest_reference_tile,
    simulate_next_frame,
    solve_two_frame_od,
)


def localize_frame1(db: ReferenceDatabase, matcher, georef, query: GeoTile, top_k: int = 10) -> dict | None:
    """Real retrieve -> match -> georeference. Returns None if no candidate
    clears min_inliers (caller should try a different query)."""
    frame = np.array(Image.open(query.image_path).convert("RGB"))
    candidates = db.retrieve(frame, top_k)

    best_tile, best_match, best_tile_image = None, None, None
    for tile, _sim in candidates:
        tile_image = np.array(Image.open(tile.image_path).convert("RGB"))
        m = matcher.match(frame, tile_image, tile_id=tile.tile_id)
        if best_match is None or m.num_inliers > best_match.num_inliers:
            best_tile, best_match, best_tile_image = tile, m, tile_image

    if best_match is None or best_match.num_inliers < matcher.min_inliers:
        return None

    tie_points = georef.make_tie_points(best_match, best_tile, best_tile_image.shape)
    if not tie_points:
        return None
    pred_center = np.array([[tp.lat, tp.lon] for tp in tie_points]).mean(axis=0)

    ts = query.timestamp
    epoch1 = Epoch(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), 0, 0, 0.0)

    return {
        "query_id": query.tile_id,
        "num_inliers": best_match.num_inliers,
        "matched_tile_id": best_tile.tile_id,
        "lat1": float(pred_center[0]),
        "lon1": float(pred_center[1]),
        "epoch1": epoch1,
    }


def run_two_frame_od(
    db: ReferenceDatabase,
    matcher,
    georef,
    queries: list[GeoTile],
    reference_tiles_for_snap: list[GeoTile],
    altitude_km: float = ISS_ALTITUDE_KM,
    dt_s: float = 120.0,
    max_attempts: int = 15,
    seed: int = 0,
) -> dict:
    """Tries queries until one localizes (frame 1), then runs the full
    simulate -> snap -> solve chain. reference_tiles_for_snap is whatever
    tile set frame 2 should be snapped against (e.g. nano's zoom-9-only
    tiles, or astroloc's full multi-zoom set) -- deliberately separate from
    db's own (possibly rotation-augmented) index, since snapping is a plain
    geometric nearest-neighbor lookup, not a retrieval/embedding operation.
    """
    rng = random.Random(seed)
    shuffled = queries[:]
    rng.shuffle(shuffled)

    frame1 = None
    for query in shuffled[:max_attempts]:
        frame1 = localize_frame1(db, matcher, georef, query)
        if frame1 is not None:
            break
    if frame1 is None:
        return {"success": False, "reason": f"no query localized in {max_attempts} attempts"}

    sim = simulate_next_frame(
        frame1["lat1"], frame1["lon1"], frame1["epoch1"], altitude_km=altitude_km, dt_s=dt_s
    )
    snapped_tile = nearest_reference_tile(reference_tiles_for_snap, sim["lat2_deg"], sim["lon2_deg"])
    snapped_center = snapped_tile.corners_latlon.mean(axis=0)

    od = solve_two_frame_od(
        frame1["lat1"], frame1["lon1"], frame1["epoch1"],
        float(snapped_center[0]), float(snapped_center[1]), sim["epoch2"],
        altitude_km=altitude_km,
    )

    true_speed_kms = float(np.linalg.norm(sim["v1_true"]) / 1e3)
    snap_distance_km = float(
        np.linalg.norm(np.array([sim["lat2_deg"], sim["lon2_deg"]]) - snapped_center) * 111.0
    )  # rough deg->km, just for a human-readable sense of snap distance

    return {
        "success": od["success"],
        "frame1": frame1,
        "simulated_frame2": {"lat2": sim["lat2_deg"], "lon2": sim["lon2_deg"], "epoch2": str(sim["epoch2"])},
        "snapped_tile_id": snapped_tile.tile_id,
        "snapped_center": snapped_center.tolist(),
        "approx_snap_distance_km": snap_distance_km,
        "input_altitude_km": altitude_km,
        "dt_s": dt_s,
        "true_speed_kms": true_speed_kms,
        "solved_speed_kms": od["speed_solved_kms"],
        "solved_altitude_km": od["altitude_r1_km"],
        "speed_error_vs_true_ms": abs(od["speed_solved_kms"] - true_speed_kms) * 1e3,
        "direction_residual_rad": od["direction_residual_rad"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--db-cache", required=True)
    ap.add_argument("--retriever", choices=["astroloc", "nano"], default="astroloc")
    ap.add_argument("--region", default="Alps", choices=list(REGIONS.keys()))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--user-config", default="user_config.yaml")
    ap.add_argument("--altitude-km", type=float, default=ISS_ALTITUDE_KM)
    ap.add_argument("--dt-s", type=float, default=120.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.retriever == "nano":
        from astroloc.models.dinov2_salad import DinoV2SaladRetriever
        from nano.data import build_eval_reference_tiles_zoom9
    else:
        from astroloc.models.dinov2_salad import DinoV2SaladRetriever

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.user_config) as f:
        user_config = yaml.safe_load(f)

    retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device=args.device)
    index = FaissFlatIndex(retriever.descriptor_dim)
    db = ReferenceDatabase.load(args.db_cache, retriever, index)
    print(f"Loaded db: {len(db.tiles)} tiles, backbone={retriever.model.backbone_name}, dim={retriever.descriptor_dim}")

    matcher = SiftLightGlueMatcher(
        max_num_keypoints=config["matcher"]["max_num_keypoints"],
        img_size=config["matcher"]["img_size"],
        max_ransac_iters=config["matcher"]["max_ransac_iters"],
        min_inliers=config["pipeline"]["min_inliers"],
        device=args.device,
    )
    georef = Georeferencer()

    center_lat, center_lon = REGIONS[args.region]
    queries = load_scoped_queries(user_config["queries_dir"], center_lat, center_lon, config["eval"]["query_dist_km"])
    reference_tiles_for_snap = list(db.tiles.values())

    result = run_two_frame_od(
        db, matcher, georef, queries, reference_tiles_for_snap,
        altitude_km=args.altitude_km, dt_s=args.dt_s,
    )
    import json

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
