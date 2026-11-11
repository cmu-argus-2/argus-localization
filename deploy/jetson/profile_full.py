"""Step-by-step cost of the full localization flow on the Jetson.

Runs the current configuration (TensorRT retriever, stored tile features, top-10
tiles, repo matcher) on N photos and measures, for every step:
  wall time, CPU time (all threads of this process), GPU time (CUDA events),
  disk I/O, CPU<->GPU copies, process RAM, PyTorch GPU memory.
Also a cold-disk pass (tile files evicted from the page cache before reading).
GPU kernel time inside LightGlue is measured separately with nsys (lg_bench.py).

Run on the Jetson from ~/Desktop/astroloc/repo, with tegrastats logging alongside:
    PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../profile_full.py --photos 50
Writes ../e2e_alps_full/profile_full.json.
"""

import argparse
import glob
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightglue.sift as lg_sift  # noqa: E402
from e2e_match import haversine_km, preprocess  # noqa: E402
from core.types import MatchResult  # noqa: E402
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from georeference.georeferencer import Georeferencer  # noqa: E402
from lightglue.utils import numpy_image_to_torch, rbd  # noqa: E402
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher  # noqa: E402

STAGES = [
    "1 read photo from disk",
    "2 decode JPEG",
    "3 resize to 224 + normalize",
    "4a retriever: copy in",
    "4b retriever: TensorRT",
    "4c retriever: copy out",
    "5 search 7,602 tiles",
    "6a photo SIFT: copy to GPU",
    "6b photo SIFT: OpenCV SIFT",
    "6c photo SIFT: resize to 512 + other",
    "7a tiles: read features from disk",
    "7b tiles: copy to GPU",
    "7c tiles: LightGlue",
    "7d tiles: results to CPU",
    "7e tiles: MAGSAC geometry check",
    "8 decide + georeference",
]
GPU_STAGES = {"5 search 7,602 tiles", "4a retriever: copy in", "4b retriever: TensorRT", "4c retriever: copy out",
              "6a photo SIFT: copy to GPU", "7b tiles: copy to GPU", "7c tiles: LightGlue"}


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


class Timer:
    """Accumulates wall, CPU and (optionally) GPU time per stage."""

    def __init__(self):
        self.wall = {s: [] for s in STAGES}
        self.cpu = {s: [] for s in STAGES}
        self.gpu = {s: [] for s in STAGES}
        self.cur = None

    def start(self, stage, gpu=False):
        torch.cuda.synchronize()
        ev = None
        if gpu:
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        self.cur = (stage, time.perf_counter(), time.process_time(), ev)

    def stop(self, add=False):
        stage, w0, c0, ev = self.cur
        if ev:
            ev[1].record()
        torch.cuda.synchronize()
        w, c = time.perf_counter() - w0, time.process_time() - c0
        g = ev[0].elapsed_time(ev[1]) / 1e3 if ev else 0.0
        if add and self.wall[stage]:
            self.wall[stage][-1] += w; self.cpu[stage][-1] += c; self.gpu[stage][-1] += g
        else:
            self.wall[stage].append(w); self.cpu[stage].append(c); self.gpu[stage].append(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../e2e_alps_full")
    ap.add_argument("--tiles-dir", default="../tiles/2021_09")
    ap.add_argument("--tile-feats", default="../tile_feats_2021_09")
    ap.add_argument("--engine", default="../astroloc_full_fp32.trt")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--photos", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cold-photos", type=int, default=15)
    ap.add_argument("--lightglue-mp", action="store_true", help="LightGlue in mixed (half) precision")
    ap.add_argument("--out", default="profile_full.json")
    ap.add_argument("--tile-pack", default=None, help="packed tile features (pack_tile_feats.py); replaces --tile-feats")
    ap.add_argument("--search-on-gpu", action="store_true", help="fingerprint search on the GPU")
    args = ap.parse_args()
    report = {"config": vars(args)}
    mem = {"rss_start_mb": rss_mb()}

    # ---- load everything (one-time costs) ----
    t = time.perf_counter()
    db = np.load(os.path.join(args.data, "db_vecs.npy"))
    db_gpu = torch.from_numpy(db).cuda() if args.search_on_gpu else None
    pack = None
    if args.tile_pack:
        pack = {k: np.load(os.path.join(args.tile_pack, k + ".npy"), mmap_mode="r")
                for k in ("keypoints", "scales", "oris", "keypoint_scores", "descriptors")}
        pack_counts = np.load(os.path.join(args.tile_pack, "counts.npy"))
        pack_size = np.load(os.path.join(args.tile_pack, "image_size.npy"))
        pack_shape = np.load(os.path.join(args.tile_pack, "tile_shape.npy"))
    meta = json.load(open(os.path.join(args.data, "db_tiles.json")))
    queries = json.load(open(os.path.join(args.data, "queries.json")))
    by_id = {x.tile_id: x for x in map(parse_geotile_filename,
                                       glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))}
    tiles = [by_id[m["tile_id"]] for m in meta]
    load = {"fingerprint database + tile list": time.perf_counter() - t}
    mem["rss_after_db_mb"] = rss_mb()

    t = time.perf_counter()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    d_in = torch.empty(tuple(ctx.get_tensor_shape(in_name)), dtype=torch.float32, device="cuda")
    d_out = torch.empty(tuple(ctx.get_tensor_shape(out_name)), dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(in_name, d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    load["TensorRT engine"] = time.perf_counter() - t
    mem["trt_engine_file_mb"] = os.path.getsize(args.engine) / 2**20
    mem["trt_activation_memory_mb"] = engine.device_memory_size / 2**20
    mem["rss_after_engine_mb"] = rss_mb()

    t = time.perf_counter()
    matcher = SiftLightGlueMatcher(min_inliers=30)
    matcher.matcher.conf.mp = args.lightglue_mp
    georef = Georeferencer()
    torch.cuda.synchronize()
    load["SIFT + LightGlue (PyTorch, CUDA init)"] = time.perf_counter() - t
    mem["rss_after_matcher_mb"] = rss_mb()
    mem["torch_gpu_after_load_mb"] = torch.cuda.memory_allocated() / 2**20
    report["load_seconds"] = load

    # time OpenCV SIFT on its own by wrapping the function LightGlue calls
    sift_box = {"t": 0.0, "c": 0.0}
    orig_sift = lg_sift.run_opencv_sift

    def timed_sift(*a, **k):
        w0, c0 = time.perf_counter(), time.process_time()
        out = orig_sift(*a, **k)
        sift_box["t"] += time.perf_counter() - w0
        sift_box["c"] += time.process_time() - c0
        return out
    lg_sift.run_opencv_sift = timed_sift

    T = Timer()
    n_total = args.warmup + args.photos
    peak_rss = 0.0
    torch.cuda.reset_peak_memory_stats()
    results = []
    for i, qm in enumerate(queries[:n_total]):
        if i == args.warmup:
            T = Timer()
            torch.cuda.reset_peak_memory_stats()
        path = os.path.join(args.data, "queries", qm["file"])

        T.start(STAGES[0]); raw = open(path, "rb").read(); T.stop()
        T.start(STAGES[1]); img = Image.open(io.BytesIO(raw)).convert("RGB"); photo = np.array(img); T.stop()
        T.start(STAGES[2]); x = preprocess(img); T.stop()
        T.start(STAGES[3], gpu=True); d_in[0].copy_(torch.from_numpy(x)); T.stop()
        T.start(STAGES[4], gpu=True); ctx.execute_async_v3(stream.cuda_stream); T.stop()
        T.start(STAGES[5], gpu=True); v = d_out[0].cpu().numpy(); T.stop()
        if db_gpu is not None:
            T.start(STAGES[6], gpu=True)
            cand = torch.topk((db_gpu @ d_out[0]).amax(dim=1), args.top_k).indices.cpu().numpy()
            T.stop()
        else:
            T.start(STAGES[6]); cand = np.argsort(-(db @ v).max(axis=1))[: args.top_k]; T.stop()

        # photo SIFT, split into upload/resize (GPU), OpenCV SIFT (CPU), the rest
        T.start(STAGES[7], gpu=True)
        tens = numpy_image_to_torch(photo).to("cuda")
        T.stop()
        sift_box["t"] = sift_box["c"] = 0.0
        w0, c0 = time.perf_counter(), time.process_time()
        with torch.no_grad():
            pf = matcher.extractor.extract(tens, resize=matcher.img_size)
        torch.cuda.synchronize()
        w_all, c_all = time.perf_counter() - w0, time.process_time() - c0
        for s, w, c in ((STAGES[8], sift_box["t"], sift_box["c"]),
                        (STAGES[9], w_all - sift_box["t"], c_all - sift_box["c"])):
            T.wall[s].append(w); T.cpu[s].append(c); T.gpu[s].append(0.0)

        best = None
        for k, idx in enumerate(cand):
            add = k > 0
            tile = tiles[idx]
            T.start(STAGES[10])
            if pack is not None:
                n = int(pack_counts[idx])
                arrays = {key: np.ascontiguousarray(a[idx:idx + 1, :n]) for key, a in pack.items()}
                arrays["image_size"] = pack_size[idx:idx + 1]
                arrays["tile_shape"] = pack_shape[idx]
            else:
                stored = np.load(os.path.join(args.tile_feats, tile.tile_id + ".npz"))
                arrays = {key: stored[key] for key in stored.files}
            T.stop(add)
            T.start(STAGES[11], gpu=True)
            tf = {key: torch.from_numpy(a).to("cuda") for key, a in arrays.items() if key != "tile_shape"}
            T.stop(add)
            T.start(STAGES[12], gpu=True)
            with torch.no_grad():
                out = matcher.matcher({"image0": pf, "image1": tf})
            T.stop(add)
            T.start(STAGES[13])
            pr, tr, mr = rbd(pf), rbd(tf), rbd(out)
            mt = mr["matches"]
            qp = pr["keypoints"][mt[:, 0]].cpu().numpy()
            tp = tr["keypoints"][mt[:, 1]].cpu().numpy()
            T.stop(add)
            T.start(STAGES[14])
            h, mask, n_in = None, np.zeros(len(qp), bool), 0
            if len(qp) >= 4:
                h, m_ = cv2.findHomography(qp, tp, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0,
                                           maxIters=matcher.max_ransac_iters, confidence=0.999)
                if m_ is not None:
                    mask = m_.astype(bool).reshape(-1); n_in = int(mask.sum())
            T.stop(add)
            if best is None or n_in > best[0].num_inliers:
                best = (MatchResult(query_pts=qp, tile_pts=tp, inlier_mask=mask, homography=h, num_inliers=n_in),
                        tile, tuple(int(s) for s in arrays["tile_shape"]))

        T.start(STAGES[15])
        m, tile, shape = best
        fix = m.num_inliers >= 30
        if fix:
            fp = georef.estimate_query_footprint(m, tile, photo.shape, shape)
            ties = georef.make_tie_points(m, tile, shape)
        T.stop()
        if fix and fp is not None:
            c = fp.mean(axis=0)
            results.append(haversine_km(qm["centre"][0], qm["centre"][1], float(c[0]), float(c[1])))
        peak_rss = max(peak_rss, rss_mb())

    # ---- summarise ----
    n = len(T.wall[STAGES[0]])
    rows, total_w = [], sum(np.mean(T.wall[s]) for s in STAGES)
    for s in STAGES:
        w, c, g = np.mean(T.wall[s]), np.mean(T.cpu[s]), np.mean(T.gpu[s])
        rows.append({"stage": s, "wall_ms": round(1e3 * w, 2), "share_pct": round(100 * w / total_w, 1),
                     "cpu_ms": round(1e3 * c, 2), "cpu_cores_busy": round(c / w, 2) if w > 0 else 0,
                     "gpu_ms": round(1e3 * g, 2) if s in GPU_STAGES else None})
    report["photos_measured"] = n
    report["per_photo_stages"] = rows
    report["per_photo_total_ms"] = round(1e3 * total_w, 1)
    report["fixes"] = len(results)
    report["fix_median_error_km"] = round(float(np.median(results)), 1) if results else None
    mem.update({"rss_peak_mb": peak_rss, "torch_gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                "torch_gpu_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                "fingerprint_db_mb": db.nbytes / 2**20})
    report["memory"] = {k: round(v, 1) for k, v in mem.items()}

    # GPU kernel time inside LightGlue: PyTorch's profiler reports no CUDA activity on this
    # Jetson build, so that is measured separately with nsys (lg_bench.py).

    # ---- cold disk: evict tile feature files from the page cache before reading ----
    cold = []
    for qm in queries[n_total: n_total + args.cold_photos]:
        img = Image.open(os.path.join(args.data, "queries", qm["file"])).convert("RGB")
        d_in[0].copy_(torch.from_numpy(preprocess(img)))
        ctx.execute_async_v3(stream.cuda_stream)
        v = d_out[0].cpu().numpy()
        cand = np.argsort(-(db @ v).max(axis=1))[: args.top_k]
        paths = [os.path.join(args.tile_feats, tiles[i].tile_id + ".npz") for i in cand]
        for p in paths:
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        w0 = time.perf_counter()
        for p in paths:
            st = np.load(p); _ = {key: st[key] for key in st.files}
        cold.append(time.perf_counter() - w0)
    report["cold_disk_tile_features_ms_per_photo"] = round(1e3 * float(np.median(cold)), 1) if cold else None
    report["tile_feature_file_kb"] = round(np.mean([os.path.getsize(os.path.join(args.tile_feats, t.tile_id + ".npz"))
                                                    for t in tiles[:200]]) / 1024, 1)

    out = os.path.join(args.data, args.out)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
