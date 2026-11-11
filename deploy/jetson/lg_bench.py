"""LightGlue on one photo/tile pair, repeated, for nsys: is it GPU work or CPU launch overhead?
Run from ~/Desktop/astroloc/repo:
    nsys profile -t cuda,nvtx -o ../lg_bench python3 ../lg_bench.py
"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.getcwd())
from lightglue.utils import numpy_image_to_torch
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher

D = "../e2e_alps_full"
q = json.load(open(D + "/queries.json"))[0]
meta = json.load(open(D + "/db_tiles.json"))
tile_id = meta[q["positives"][0]]["tile_id"]
m = SiftLightGlueMatcher()
photo = np.array(Image.open(D + "/queries/" + q["file"]).convert("RGB"))
st = np.load("../tile_feats_2021_09/" + tile_id + ".npz")
tf = {k: torch.from_numpy(st[k]).cuda() for k in st.files if k != "tile_shape"}
with torch.no_grad():
    pf = m.extractor.extract(numpy_image_to_torch(photo).cuda(), resize=512)
lg = m.matcher
print("LightGlue conf:", {k: getattr(lg.conf, k, None) for k in ("n_layers", "flash", "mp", "depth_confidence", "width_confidence", "filter_threshold")})
print("flash/mem-efficient SDPA enabled:", torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
print("points: photo", pf["keypoints"].shape[1], "tile", tf["keypoints"].shape[1])
with torch.no_grad():
    for _ in range(5):
        out = lg({"image0": pf, "image1": tf})
    torch.cuda.synchronize()
    n = 20
    t0 = time.perf_counter(); c0 = time.process_time()
    torch.cuda.nvtx.range_push("lightglue_x20")
    for _ in range(n):
        out = lg({"image0": pf, "image1": tf})
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    w = (time.perf_counter() - t0) / n; c = (time.process_time() - c0) / n
print("per call: wall %.1f ms, CPU %.1f ms, layers run %s of %d, matches %d" % (1e3 * w, 1e3 * c, int(out["stop"]), lg.conf.n_layers, int((out["matches0"] > -1).sum())))
