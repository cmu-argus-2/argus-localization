"""Where the time goes around the TensorRT LightGlue engine, per tile (Jetson).
Run from ~/Desktop/astroloc/repo:
    PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../trt_lg_bench.py --engine ../lightglue_sift_k1024_fp16.trt
"""
import argparse, json, os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_match import TilePack
from lightglue_trt import LightGlueTRT, prepare, matches_from_scores
from lightglue.utils import numpy_image_to_torch
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher

ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True); ap.add_argument("--data", default="../e2e_alps_full")
ap.add_argument("--pack", default="../tile_feats_pack"); ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()
m = SiftLightGlueMatcher(); trt_lg = LightGlueTRT(a.engine)
pack = TilePack(a.pack, "cuda", False)
q = json.load(open(os.path.join(a.data, "queries.json")))[0]
photo = np.array(Image.open(os.path.join(a.data, "queries", q["file"])).convert("RGB"))
with torch.no_grad():
    pf = m.extractor.extract(numpy_image_to_torch(photo).cuda(), resize=512)
trt_lg.set_photo(pf)
T = {"pack read + to GPU": 0, "prepare (normalize, pad)": 0, "copy into engine buffers": 0, "engine": 0, "matches from scores": 0}
def t(): torch.cuda.synchronize(); return time.perf_counter()
idxs = q["positives"][:1] * a.n
for i, idx in enumerate(idxs):
    t0 = t(); tf, _ = pack.feats(idx); t1 = t()
    kp, desc, valid, n = prepare(tf, trt_lg.k); t2 = t()
    trt_lg.buf["kpts1"].copy_(kp); trt_lg.buf["desc1"].copy_(desc); trt_lg.buf["valid1"].copy_(valid); t3 = t()
    trt_lg.ctx.execute_async_v3(trt_lg.stream.cuda_stream); t4 = t()
    mt = matches_from_scores(trt_lg.buf[trt_lg.out_name], trt_lg.m, n, trt_lg.threshold); _ = mt.cpu(); t5 = t()
    if i >= 5:
        for k, d in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)): T[k] += d
for k, v in T.items(): print("%-28s %6.2f ms" % (k, 1e3 * v / (a.n - 5)))
print("%-28s %6.2f ms" % ("total per tile", 1e3 * sum(T.values()) / (a.n - 5)))
