"""End-to-end localization test, JETSON half.

Repeats the workstation test (e2e_prepare.py) on the Orin: each test photo goes
through the TensorRT engine, its fingerprint is searched against the same
reference database, and the result is scored the same way. Then it compares
against the workstation's results.

Run on the Jetson, next to the engine and the copied e2e folder:
    python3 e2e_jetson.py --engine nano_fp32.trt --data e2e_alps

Needs: tensorrt, pycuda, numpy, PIL. For total memory use, run `tegrastats`
in a second terminal while this runs.
"""

import argparse
import json
import math
import os
import resource
import time

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
K_VALUES = (1, 5, 10, 100)


# ---- preprocessing: must stay byte-identical to e2e_prepare.py ----
def preprocess(img, rot_k=0):
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    if rot_k:
        x = np.rot90(x, rot_k, axes=(0, 1))
    x = x.transpose(2, 0, 1)
    return ((x - MEAN) / STD).astype(np.float32)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ---- search + scoring: must stay identical to e2e_prepare.py ----
def rank_tiles(db_vecs, q):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="nano_fp32.trt")
    ap.add_argument("--data", default="e2e_alps")
    args = ap.parse_args()

    # load reference database + ground truth
    t0 = time.time()
    db_vecs = np.load(os.path.join(args.data, "db_vecs.npy"))
    with open(os.path.join(args.data, "db_tiles.json")) as f:
        db_centres = np.array([t["centre"] for t in json.load(f)])
    with open(os.path.join(args.data, "queries.json")) as f:
        queries_meta = json.load(f)
    ws_vecs = np.load(os.path.join(args.data, "ws_query_vecs.npy"))
    with open(os.path.join(args.data, "ws_results.json")) as f:
        ws = json.load(f)
    print(f"database: {db_vecs.shape[0]} tiles, {db_vecs.nbytes / 1e6:.1f} MB, loaded in {time.time() - t0:.1f}s")
    print(f"test photos: {len(queries_meta)}")

    # load engine, allocate buffers once
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    host_in = np.empty(tuple(ctx.get_tensor_shape(in_name)), dtype=np.float32)
    host_out = np.empty(tuple(ctx.get_tensor_shape(out_name)), dtype=np.float32)
    d_in, d_out = cuda.mem_alloc(host_in.nbytes), cuda.mem_alloc(host_out.nbytes)
    stream = cuda.Stream()
    ctx.set_tensor_address(in_name, int(d_in))
    ctx.set_tensor_address(out_name, int(d_out))

    def embed(arr):
        host_in[0] = arr
        cuda.memcpy_htod_async(d_in, host_in, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(host_out, d_out, stream)
        stream.synchronize()
        return host_out[0].copy()

    # localize each photo
    rankings, j_vecs, pre_ms, embed_ms, search_ms = [], [], [], [], []
    for qm in queries_meta:
        t0 = time.perf_counter()
        arr = preprocess(Image.open(os.path.join(args.data, "queries", qm["file"])).convert("RGB"))
        t1 = time.perf_counter()
        v = embed(arr)
        t2 = time.perf_counter()
        r = rank_tiles(db_vecs, v)
        t3 = time.perf_counter()
        j_vecs.append(v)
        rankings.append(r)
        pre_ms.append((t1 - t0) * 1e3)
        embed_ms.append((t2 - t1) * 1e3)
        search_ms.append((t3 - t2) * 1e3)

    metrics = score(rankings, queries_meta, db_centres)
    metrics.update({
        "machine": "Jetson Orin (TensorRT FP32)",
        "num_db_tiles": int(db_vecs.shape[0]),
        "db_mb": round(db_vecs.nbytes / 1e6, 1),
        "embed_ms_median": round(float(np.median(embed_ms)), 2),
        "search_ms_median": round(float(np.median(search_ms)), 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
    })

    # agreement with the workstation
    j = np.stack(j_vecs)
    cos = (j * ws_vecs).sum(1) / (np.linalg.norm(j, axis=1) * np.linalg.norm(ws_vecs, axis=1))
    same_top1 = float(np.mean([r[0] == t for r, t in zip(rankings, ws["top1"])])) * 100

    wm = ws["metrics"]
    print("\n=== JETSON vs WORKSTATION ===")
    print(f"  {'metric':18s} {'Jetson':>14s} {'workstation':>14s}")
    for k in ["num_queries", "R@1", "R@5", "R@10", "R@100", "median_error_km", "mean_error_km",
              "num_db_tiles", "db_mb", "embed_ms_median", "search_ms_median", "peak_rss_mb"]:
        print(f"  {k:18s} {str(metrics[k]):>14s} {str(wm.get(k)):>14s}")
    print(f"\n  preprocess ms (median, Jetson): {np.median(pre_ms):.1f}")
    print(f"  fingerprint agreement, cosine  : min {cos.min():.6f}  mean {cos.mean():.6f}")
    print(f"  same top-1 tile as workstation : {same_top1:.1f}%")
    print("\n  peak_rss_mb is this process only. On the Jetson, CPU and GPU share memory;")
    print("  use tegrastats for the system total.")

    with open(os.path.join(args.data, "jetson_results.json"), "w") as f:
        json.dump({"metrics": metrics, "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
                   "same_top1_pct": same_top1}, f, indent=2)

    del ctx, engine
    d_in.free()
    d_out.free()


if __name__ == "__main__":
    main()
