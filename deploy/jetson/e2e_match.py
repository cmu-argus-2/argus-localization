"""End-to-end localization WITH the matcher: retrieval -> SIFT + LightGlue -> location.

Takes the folder written by e2e_prepare.py (fingerprint database, test photos,
ground truth, photo fingerprints) and adds the repo's matcher on top, following
core/pipeline.py: match the top-k retrieved tiles, keep the one with the most
inliers, call it a fix at >= min_inliers, and georeference the photo through the
fitted homography. The matcher and georeferencer are the repo's own classes.

Run from the argus-localization repo root:
    python deploy/jetson/e2e_match.py --data ~/e2e_alps_full \
        --tiles-dir /mnt/sdc1/astroloc/data/database/2021_09

With --engine (Jetson) the photo fingerprints come from the TensorRT engine in this
same process instead of from --query-vecs, so one process holds the whole flow:
retriever + search + matcher.

With --tile-feats the tiles' SIFT features are read from the folder written by
precompute_tile_feats.py instead of being recomputed from the tile images.

Writes <data>/match_results.json (summary + one record per photo).
"""

import argparse
import glob
import json
import math
import os
import resource
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from georeference.georeferencer import Georeferencer  # noqa: E402
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher  # noqa: E402

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Held-out region centres and the training exclusion radius (astroloc/data/regions.py).
EVAL_CENTRES = [(45, 10), (30, -95), (23, 30), (-3, -60), (38, -122), (40, 105)]
TRAIN_EXCLUSION_KM = 2000


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _gnomonic_km(lat, lon, lat0, lon0):
    """Points on the sphere projected onto the tangent plane at (lat0, lon0), in km."""
    la, lo, la0, lo0 = (np.radians(np.asarray(x, dtype=float)) for x in (lat, lon, lat0, lon0))
    p = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    e = np.array([-np.sin(lo0), np.cos(lo0), 0.0])
    n = np.array([-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)])
    u = np.array([np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)])
    return 6371.0 * np.stack([p @ e, p @ n], -1) / (p @ u)[:, None]


def tie_point_error_km(filename, shape, tie_points):
    """Median distance between a fix's tie points and the true ground points of their pixels.

    Truth per pixel comes from the photo's four true corners (in its filename: BL TL TR BR lat/lon),
    through a homography from the photo onto the ground's tangent plane. Against exact truth this is
    within ~0.2 km on the Argus-like GeoTIFF frames and ~1 km on perspective renders. This is the
    error OD sees; match_error_km (footprint centre) extrapolates to the photo corners and reads higher.
    """
    corners = np.array([f for f in os.path.splitext(os.path.basename(filename))[0].split("@") if f][:8],
                       dtype=float).reshape(4, 2)
    h, w = shape
    lat0, lon0 = corners.mean(axis=0)
    src = np.array([[0, h], [0, 0], [w, 0], [w, h]], dtype=float)
    dst = _gnomonic_km(corners[:, 0], corners[:, 1], lat0, lon0)
    rows = []
    for (x, y), (a, b) in zip(src, dst):
        rows += [[x, y, 1, 0, 0, 0, -a * x, -a * y, -a], [0, 0, 0, x, y, 1, -b * x, -b * y, -b]]
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    tp = np.asarray(tie_points, dtype=float)
    q = np.column_stack([tp[:, :2] + 0.5, np.ones(len(tp))]) @ H.T
    truth = q[:, :2] / q[:, 2:]
    return float(np.median(np.linalg.norm(truth - _gnomonic_km(tp[:, 2], tp[:, 3], lat0, lon0), axis=1)))


# ---- preprocessing: must stay byte-identical to e2e_prepare.py / e2e_jetson.py ----
def preprocess(img):
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((x.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)


def trt_embedder(engine_path):
    """Fingerprint function backed by a TensorRT engine. The buffers are PyTorch CUDA
    tensors, so the engine and the matcher share one CUDA context."""
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    d_in = torch.empty(tuple(ctx.get_tensor_shape(in_name)), dtype=torch.float32, device="cuda")
    d_out = torch.empty(tuple(ctx.get_tensor_shape(out_name)), dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(in_name, d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    stream = torch.cuda.current_stream()

    def embed(arr):
        d_in[0].copy_(torch.from_numpy(arr))
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return d_out[0].cpu().numpy()

    embed.keep_alive = (runtime, engine, ctx, d_in, d_out)
    return embed


def stored_tile_feats(stored, device, max_keypoints):
    """Stored tile features as tensors, cut to the max_keypoints highest-scoring points.
    SIFT keeps its top-scoring points, so the top N of the stored 1,024 are exactly
    what extraction with a cap of N would have produced."""
    feats = {k: torch.from_numpy(stored[k]).to(device) for k in stored.files if k != "tile_shape"}
    n = feats["keypoints"].shape[1]
    if n > max_keypoints:
        keep = torch.topk(feats["keypoint_scores"][0], max_keypoints).indices
        feats = {k: (v[:, keep] if v.dim() > 1 and v.shape[1] == n else v) for k, v in feats.items()}
    return feats


class TilePack:
    """All tiles' stored SIFT features from pack_tile_feats.py, in database order.
    Memory-mapped from disk, or (on_gpu) loaded once into GPU memory."""

    KEYS = ("keypoints", "scales", "oris", "keypoint_scores", "descriptors")

    def __init__(self, pack_dir, device, on_gpu):
        self.device, self.on_gpu = device, on_gpu
        mm = {k: np.load(os.path.join(pack_dir, k + ".npy"), mmap_mode="r") for k in self.KEYS}
        self.arrays = {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in mm.items()} if on_gpu else mm
        self.counts = np.load(os.path.join(pack_dir, "counts.npy"))
        self.image_size = torch.from_numpy(np.load(os.path.join(pack_dir, "image_size.npy"))).to(device)
        self.tile_shape = np.load(os.path.join(pack_dir, "tile_shape.npy"))
        self.tile_ids = json.load(open(os.path.join(pack_dir, "tile_ids.json")))

    def feats(self, idx, max_keypoints=None):
        n = int(self.counts[idx])
        if self.on_gpu:
            out = {k: self.arrays[k][idx:idx + 1, :n] for k in self.KEYS}
        else:
            out = {k: torch.from_numpy(np.ascontiguousarray(self.arrays[k][idx:idx + 1, :n])).to(self.device)
                   for k in self.KEYS}
        out["image_size"] = self.image_size[idx:idx + 1]
        if max_keypoints and n > max_keypoints:  # keep the highest-scoring points, as SIFT itself would
            keep = torch.topk(out["keypoint_scores"][0].float(), max_keypoints).indices
            out = {k: (v[:, keep] if v.dim() > 1 and v.shape[1] == n else v) for k, v in out.items()}
        return out, tuple(int(x) for x in self.tile_shape[idx])


def batched_lightglue(lg, f0, f1s):
    """LightGlue on one photo against several tiles in a single batched pass.
    Tiles are padded to the same number of points and the padding is masked out.
    Per-pair early stopping is kept (each pair is scored at the layer where it would
    have stopped); point pruning is off, as LightGlue does not support it for batches.
    Returns one (matches [S x 2]) tensor per tile."""
    from lightglue.lightglue import filter_matches, normalize_keypoints

    b, dev = len(f1s), f0["keypoints"].device
    m = f0["keypoints"].shape[1]
    ns = [f["keypoints"].shape[1] for f in f1s]
    n_max = max(ns)

    def kp(f):
        k = normalize_keypoints(f["keypoints"], f["image_size"])
        if lg.conf.add_scale_ori:
            k = torch.cat([k, f["scales"].unsqueeze(-1), f["oris"].unsqueeze(-1)], -1)
        return k

    def pad(x):
        if x.shape[1] == n_max:
            return x
        extra = list(x.shape)
        extra[1] = n_max - x.shape[1]
        return torch.cat([x, x.new_zeros(extra)], 1)

    with torch.autocast(enabled=lg.conf.mp, device_type="cuda"):
        k0 = kp(f0).expand(b, -1, -1)
        k1 = torch.cat([pad(kp(f)) for f in f1s])
        desc0 = f0["descriptors"].detach().expand(b, -1, -1).contiguous()
        desc1 = torch.cat([pad(f["descriptors"].detach()) for f in f1s]).contiguous()
        if torch.is_autocast_enabled():
            desc0, desc1 = desc0.half(), desc1.half()
        # masks carry a head dimension so attention can broadcast them per batch item
        mask0 = torch.ones(b, 1, m, 1, dtype=torch.bool, device=dev)
        valid1 = torch.arange(n_max, device=dev)[None] < torch.tensor(ns, device=dev)[:, None]
        mask1 = valid1[:, None, :, None]
        desc0, desc1 = lg.input_proj(desc0), lg.input_proj(desc1)
        enc0, enc1 = lg.posenc(k0), lg.posenc(k1)

        n_vec = torch.tensor(ns, device=dev, dtype=torch.float32)
        stop = [None] * b
        frozen = [None] * b
        active = list(range(b))
        for i in range(lg.conf.n_layers):
            desc0, desc1 = lg.transformers[i](desc0, desc1, enc0, enc1, mask0=mask0, mask1=mask1)
            if i == lg.conf.n_layers - 1:
                break
            if lg.conf.depth_confidence > 0:
                t0, t1 = lg.token_confidence[i](desc0, desc1)
                th = lg.confidence_thresholds[i]
                low = (t0 < th).float().sum(1) + ((t1 < th) & valid1).float().sum(1)
                done = ((1.0 - low / (m + n_vec)) > lg.conf.depth_confidence).tolist()
                for k in [k for k in active if done[k]]:
                    stop[k], frozen[k] = i, (desc0[k:k + 1], desc1[k:k + 1, :ns[k]])
                active = [k for k in active if not done[k]]
                if not active:
                    break
        for k in active:
            stop[k], frozen[k] = i, (desc0[k:k + 1], desc1[k:k + 1, :ns[k]])

        out = []
        for k in range(b):
            d0, d1 = frozen[k]
            scores, _ = lg.log_assignment[stop[k]](d0, d1)
            m0, _, _, _ = filter_matches(scores, lg.conf.filter_threshold)
            valid = m0[0] > -1
            out.append(torch.stack([torch.where(valid)[0], m0[0][valid]], -1))
    return out


def homography_check(qp, tp, max_iters):
    """Same geometry check as the repo's SiftLightGlueMatcher.match."""
    from core.types import MatchResult
    if len(qp) < 4:
        return MatchResult(query_pts=np.zeros((0, 2), np.float32), tile_pts=np.zeros((0, 2), np.float32),
                           inlier_mask=np.zeros((0,), bool), homography=None, num_inliers=0)
    h, mask = cv2.findHomography(qp, tp, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0,
                                 maxIters=max_iters, confidence=0.999)
    inl = np.zeros((len(qp),), bool) if mask is None else mask.astype(bool).reshape(-1)
    return MatchResult(query_pts=qp, tile_pts=tp, inlier_mask=inl, homography=h, num_inliers=int(inl.sum()))


def rank_tiles(db_vecs, q):
    return np.argsort(-(db_vecs @ q).max(axis=1))


def excluded_from_training(filename):
    """True if the photo's nadir point is within the training exclusion radius of a held-out region."""
    fields = [f for f in os.path.splitext(filename)[0].split("@") if f]
    lat, lon = float(fields[10]), float(fields[11])
    return min(haversine_km(lat, lon, a, b) for a, b in EVAL_CENTRES) < TRAIN_EXCLUSION_KM


def pct(x):
    return round(100.0 * float(x), 1)


def summarize(records, label):
    n = len(records)
    fixes = [r for r in records if r["fix"]]
    errs = np.array([r["match_error_km"] for r in fixes]) if fixes else np.array([])
    out = {
        "subset": label,
        "photos": n,
        "retrieval_top1_correct_pct": pct(np.mean([r["retrieval_top1_correct"] for r in records])),
        "retrieval_median_error_km": round(float(np.median([r["retrieval_error_km"] for r in records])), 1),
        "correct_tile_in_topk_pct": pct(np.mean([r["correct_in_topk"] for r in records])),
        "fix_rate_pct": pct(len(fixes) / n),
    }
    if fixes:
        out.update({
            "fixes": len(fixes),
            "fix_on_correct_tile_pct": pct(np.mean([r["matched_tile_correct"] for r in fixes])),
            "fix_median_error_km": round(float(np.median(errs)), 1),
            "fix_p90_error_km": round(float(np.percentile(errs, 90)), 1),
            "fix_max_error_km": round(float(errs.max()), 1),
            "fixes_over_50km": int((errs > 50).sum()),
        })
        tie = np.array([r["tie_error_km"] for r in fixes if r.get("tie_error_km") is not None])
        if len(tie):
            out.update({
                "fix_tie_point_error_median_km": round(float(np.median(tie)), 2),
                "fix_tie_point_error_p90_km": round(float(np.percentile(tie, 90)), 2),
                "fix_tie_point_error_max_km": round(float(tie.max()), 2),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder written by e2e_prepare.py")
    ap.add_argument("--tiles-dir", required=True, help="folder holding the tile images")
    ap.add_argument("--query-vecs", default="ws_query_vecs.npy", help="photo fingerprints, relative to --data")
    ap.add_argument("--engine", default=None,
                    help="TensorRT engine; if given, fingerprints are computed here and "
                         "--query-vecs is only used to report agreement")
    ap.add_argument("--tile-feats", default=None,
                    help="folder of precomputed tile SIFT features (precompute_tile_feats.py); "
                         "if given, tile images are not opened")
    ap.add_argument("--max-keypoints", type=int, default=1024,
                    help="SIFT points per image (repo default 1024); stored tile features are cut to match")
    ap.add_argument("--lightglue-layers", type=int, default=None,
                    help="stop LightGlue after this many of its 9 layers and use that layer's matches "
                         "(each layer has its own match head; same weights, no retraining)")
    ap.add_argument("--lightglue-mp", action="store_true",
                    help="run LightGlue in mixed (half) precision")
    ap.add_argument("--tile-pack", default=None,
                    help="folder from pack_tile_feats.py; replaces --tile-feats (needs --lightglue-mp)")
    ap.add_argument("--tiles-on-gpu", action="store_true", help="load the whole tile pack into GPU memory")
    ap.add_argument("--search-on-gpu", action="store_true", help="run the fingerprint search on the GPU")
    ap.add_argument("--lightglue-trt", default=None,
                    help="LightGlue as a TensorRT engine (lightglue_trt.py) instead of PyTorch")
    ap.add_argument("--batch-tiles", action="store_true",
                    help="match all candidate tiles in one batched LightGlue call (needs --tile-pack or --tile-feats)")
    ap.add_argument("--stop-inliers", type=int, default=0,
                    help="stop matching a photo's candidates at the first tile with at least this many "
                         "inliers (0 = match all; TensorRT LightGlue path only)")
    ap.add_argument("--log-tiles", action="store_true",
                    help="record every candidate tile's inliers and the fix error it would give "
                         "(for testing stop rules offline; adds georeferencing time)")
    ap.add_argument("--save-tie-points", action="store_true",
                    help="store each fix's tie points [u, v, lat, lon] (photo pixels, inliers of the chosen tile), "
                         "the measurements orbit determination uses")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos (0 = all)")
    ap.add_argument("--out", default="match_results.json")
    args = ap.parse_args()
    data = os.path.expanduser(args.data)

    db_vecs = np.load(os.path.join(data, "db_vecs.npy"))
    with open(os.path.join(data, "db_tiles.json")) as f:
        db_meta = json.load(f)
    with open(os.path.join(data, "queries.json")) as f:
        queries = json.load(f)
    qv_path = os.path.join(data, args.query_vecs)
    if os.path.exists(qv_path):
        q_vecs = np.load(qv_path)
    elif args.engine:
        q_vecs = None  # fingerprints come from the engine; nothing to compare against
    else:
        sys.exit(f"ERROR: {qv_path} not found (needed without --engine)")
    if args.limit:
        queries = queries[: args.limit]
        q_vecs = q_vecs[: args.limit] if q_vecs is not None else None

    # Tile images, put in the same order as the fingerprint database (matched by tile id).
    paths = glob.glob(os.path.join(os.path.expanduser(args.tiles_dir), "**", "*.jpg"), recursive=True)
    by_id = {t.tile_id: t for t in (parse_geotile_filename(p) for p in paths)}
    missing = [m["tile_id"] for m in db_meta if m["tile_id"] not in by_id]
    if missing:
        sys.exit(f"ERROR: {len(missing)} of {len(db_meta)} database tiles have no image under {args.tiles_dir}")
    tiles = [by_id[m["tile_id"]] for m in db_meta]
    tile_centres = np.array([t.corners_latlon.mean(axis=0) for t in tiles])
    print(f"database: {len(tiles)} tiles | photos: {len(queries)} | top-k {args.top_k} | min inliers {args.min_inliers}")

    matcher = SiftLightGlueMatcher(min_inliers=args.min_inliers, max_num_keypoints=args.max_keypoints)
    matcher.matcher.conf.mp = args.lightglue_mp
    if args.lightglue_layers:
        if not 1 <= args.lightglue_layers <= len(matcher.matcher.transformers):
            sys.exit(f"ERROR: --lightglue-layers must be 1-{len(matcher.matcher.transformers)}")
        matcher.matcher.conf.n_layers = args.lightglue_layers
    georef = Georeferencer()
    print(f"matcher on: {matcher.device} | SIFT points per image: {args.max_keypoints} | "
          f"LightGlue precision: {'mixed (half)' if args.lightglue_mp else 'full (32-bit)'}")
    extract = matcher._extract
    embed = trt_embedder(args.engine) if args.engine else None
    if embed:
        print(f"retriever: {args.engine} (TensorRT, same process)")
    feats_dir = os.path.expanduser(args.tile_feats) if args.tile_feats else None
    if feats_dir:
        print(f"tile features: precomputed, from {feats_dir}")
    pack = None
    if args.tile_pack:
        if not (args.lightglue_mp or args.lightglue_trt):
            sys.exit("ERROR: --tile-pack stores descriptors in float16; use it with --lightglue-mp or --lightglue-trt")
        pack = TilePack(os.path.expanduser(args.tile_pack), matcher.device, args.tiles_on_gpu)
        if pack.tile_ids != [m["tile_id"] for m in db_meta]:
            sys.exit("ERROR: tile pack order does not match db_tiles.json")
        print(f"tile features: packed, {'in GPU memory' if args.tiles_on_gpu else 'memory-mapped'}")
    db_gpu = torch.from_numpy(db_vecs).to(matcher.device) if args.search_on_gpu else None
    if args.batch_tiles:
        print("LightGlue: all candidate tiles in one batched call")
    lg_trt = None
    if args.lightglue_trt:
        from lightglue_trt import LightGlueTRT
        lg_trt = LightGlueTRT(args.lightglue_trt, threshold=matcher.matcher.conf.filter_threshold)
        print(f"LightGlue: TensorRT engine {args.lightglue_trt} ({lg_trt.k} points)")

    records, times, cosines = [], [], []
    for i, (qm, qv) in enumerate(zip(queries, q_vecs if q_vecs is not None else [None] * len(queries))):
        img = Image.open(os.path.join(data, "queries", qm["file"])).convert("RGB")
        photo = np.array(img)
        truth, positives = qm["centre"], set(qm["positives"])
        t0 = time.perf_counter()
        if embed:
            ref, qv = qv, embed(preprocess(img))
            if ref is not None:
                cosines.append(float(ref @ qv / (np.linalg.norm(ref) * np.linalg.norm(qv))))
        if db_gpu is not None:
            q_t = torch.from_numpy(np.ascontiguousarray(qv)).to(matcher.device)
            candidates = torch.topk((db_gpu @ q_t).amax(dim=1), args.top_k).indices.cpu().numpy()
        else:
            candidates = rank_tiles(db_vecs, qv)[: args.top_k]

        # The repo's match() re-extracts the photo's SIFT points for every candidate.
        # Extract them once per photo and hand them back; the matching logic is untouched.
        photo_feats = extract(photo)
        matcher._extract = lambda im, _p=photo, _f=photo_feats: _f if im is _p else extract(im)
        best, best_idx, best_shape, best_rank = None, None, None, None
        tile_log, tiles_matched = [], 0
        try:
            if lg_trt is not None:
                lg_trt.set_photo(photo_feats)
                kp0 = photo_feats["keypoints"][0]
                for rank, idx in enumerate(candidates):
                    if pack is not None:
                        tf, tile_shape = pack.feats(idx, args.max_keypoints)
                    else:
                        stored = np.load(os.path.join(feats_dir, tiles[idx].tile_id + ".npz"))
                        tf = stored_tile_feats(stored, matcher.device, args.max_keypoints)
                        tile_shape = tuple(int(x) for x in stored["tile_shape"])
                    mt = lg_trt.match(tf)
                    qp = kp0[mt[:, 0]].float().cpu().numpy()
                    tp = tf["keypoints"][0][mt[:, 1]].float().cpu().numpy()
                    m = homography_check(qp, tp, matcher.max_ransac_iters)
                    tiles_matched = rank + 1
                    if best is None or m.num_inliers > best.num_inliers:
                        best, best_idx, best_shape, best_rank = m, int(idx), tile_shape, rank
                    if args.log_tiles:
                        err = None
                        if m.num_inliers >= args.min_inliers:
                            fp = georef.estimate_query_footprint(m, tiles[idx], photo.shape, tile_shape)
                            if fp is not None and np.isfinite(fp).all():
                                c = fp.mean(axis=0)
                                err = round(haversine_km(truth[0], truth[1], float(c[0]), float(c[1])), 2)
                        tile_log.append({"inliers": int(m.num_inliers), "correct": int(idx) in positives,
                                         "error_km": err})
                    if args.stop_inliers and m.num_inliers >= args.stop_inliers:
                        break
            elif args.batch_tiles:
                loaded = []
                for idx in candidates:
                    if pack is not None:
                        loaded.append(pack.feats(idx, args.max_keypoints))
                    else:
                        stored = np.load(os.path.join(feats_dir, tiles[idx].tile_id + ".npz"))
                        loaded.append((stored_tile_feats(stored, matcher.device, args.max_keypoints),
                                       tuple(int(x) for x in stored["tile_shape"])))
                with torch.no_grad():
                    all_matches = batched_lightglue(matcher.matcher, photo_feats, [f for f, _ in loaded])
                kp0 = photo_feats["keypoints"][0]
                for rank, (idx, (tf, tile_shape), mt) in enumerate(zip(candidates, loaded, all_matches)):
                    qp = kp0[mt[:, 0]].float().cpu().numpy()
                    tp = tf["keypoints"][0][mt[:, 1]].float().cpu().numpy()
                    m = homography_check(qp, tp, matcher.max_ransac_iters)
                    if best is None or m.num_inliers > best.num_inliers:
                        best, best_idx, best_shape, best_rank = m, int(idx), tile_shape, rank
            for rank, idx in enumerate([] if (args.batch_tiles or lg_trt is not None) else candidates):
                if pack is not None:
                    tile_id = tiles[idx].tile_id
                    tf, tile_shape = pack.feats(idx, args.max_keypoints)
                    matcher._tile_feats_cache[tile_id] = tf
                    m = matcher.match(photo, None, tile_id=tile_id)
                    matcher._tile_feats_cache.clear()
                elif feats_dir:
                    # Hand the stored features to the matcher through its own tile cache,
                    # then empty the cache so memory use reflects one photo at a time.
                    tile_id = tiles[idx].tile_id
                    stored = np.load(os.path.join(feats_dir, tile_id + ".npz"))
                    tile_shape = tuple(int(x) for x in stored["tile_shape"])
                    matcher._tile_feats_cache[tile_id] = stored_tile_feats(stored, matcher.device, args.max_keypoints)
                    m = matcher.match(photo, None, tile_id=tile_id)
                    matcher._tile_feats_cache.clear()
                else:
                    tile_img = np.array(Image.open(tiles[idx].image_path).convert("RGB"))
                    tile_shape = tile_img.shape
                    # tile_id=None: no per-tile feature cache, so memory use reflects one photo at a time
                    m = matcher.match(photo, tile_img, tile_id=None)
                if best is None or m.num_inliers > best.num_inliers:
                    best, best_idx, best_shape, best_rank = m, int(idx), tile_shape, rank
        finally:
            matcher._extract = extract

        rec = {
            "file": qm["file"],
            "excluded_from_training": excluded_from_training(qm["file"]),
            "retrieval_top1_correct": int(candidates[0]) in positives,
            "retrieval_error_km": haversine_km(truth[0], truth[1], *tile_centres[candidates[0]]),
            "correct_in_topk": bool(positives.intersection(int(c) for c in candidates)),
            "best_inliers": int(best.num_inliers),
            "best_rank": best_rank,
            "tiles_matched": tiles_matched or len(candidates),
            "fix": False,
        }
        if best.num_inliers >= args.min_inliers:
            footprint = georef.estimate_query_footprint(best, tiles[best_idx], photo.shape, best_shape)
            if footprint is not None and np.isfinite(footprint).all():
                centre = footprint.mean(axis=0)
                rec.update({
                    "fix": True,
                    "matched_tile_correct": best_idx in positives,
                    "match_error_km": haversine_km(truth[0], truth[1], float(centre[0]), float(centre[1])),
                })
        times.append(time.perf_counter() - t0)
        if rec["fix"]:  # scoring only, outside the timed part
            tps = [[round(tp.u, 2), round(tp.v, 2), round(tp.lat, 6), round(tp.lon, 6)]
                   for tp in georef.make_tie_points(best, tiles[best_idx], best_shape)]
            rec["tie_error_km"] = round(tie_point_error_km(qm["file"], photo.shape[:2], tps), 3)
            if args.save_tie_points:
                rec["photo_shape"] = list(photo.shape[:2])
                rec["tie_points"] = tps
        if args.log_tiles:
            rec["tiles"] = tile_log
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(queries)} photos, {np.median(times):.2f} s/photo")

    summary = {
        "all": summarize(records, "all photos"),
        "excluded_from_training": summarize([r for r in records if r["excluded_from_training"]],
                                            "photos the model cannot have trained on"),
        "seconds_per_photo_median": round(float(np.median(times)), 2),
        "seconds_per_photo_mean": round(float(np.mean(times)), 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
        "top_k": args.top_k,
        "min_inliers": args.min_inliers,
        "query_vecs": args.engine or args.query_vecs,
        "tile_feats": "precomputed" if feats_dir else "computed per photo",
        "max_keypoints": args.max_keypoints,
        "tile_pack": bool(args.tile_pack), "tiles_on_gpu": args.tiles_on_gpu,
        "search_on_gpu": args.search_on_gpu, "batch_tiles": args.batch_tiles, "lightglue_trt": args.lightglue_trt,
        "lightglue_mp": args.lightglue_mp, "lightglue_layers": args.lightglue_layers or 9,
        "stop_inliers": args.stop_inliers, "log_tiles": args.log_tiles,
        "tiles_matched_mean": round(float(np.mean([r["tiles_matched"] for r in records])), 2),
    }
    if cosines:
        summary["fingerprint_cosine_min_vs_" + os.path.splitext(args.query_vecs)[0]] = round(min(cosines), 6)
    if torch.cuda.is_available():
        summary["torch_gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 0)

    with open(os.path.join(data, args.out), "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)

    for key in ("all", "excluded_from_training"):
        s = summary[key]
        print(f"\n=== {s['subset']} ({s['photos']}) ===")
        for k, v in s.items():
            if k not in ("subset", "photos"):
                print(f"  {k:30s}: {v}")
    print(f"\n  seconds per photo (median): {summary['seconds_per_photo_median']}")
    print(f"  peak process memory (MB)  : {summary['peak_rss_mb']}")
    if "torch_gpu_peak_mb" in summary:
        print(f"  peak torch GPU memory (MB): {summary['torch_gpu_peak_mb']}")
    if cosines:
        print(f"  fingerprint agreement with {args.query_vecs}, cosine min: {min(cosines):.6f}")
    print(f"\nwrote {os.path.join(data, args.out)}")


if __name__ == "__main__":
    main()
