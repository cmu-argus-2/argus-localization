"""Where the matcher's time goes on the Jetson: 20 photos, top-15 tiles each (first photo is warm-up)."""
import glob, json, os, sys, time
import cv2, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.getcwd())
from data_loading.earthloc_loader import parse_geotile_filename
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from lightglue.utils import rbd

D = "../e2e_alps_full"
t = time.perf_counter(); db = np.load(D + "/db_vecs.npy"); load_s = time.perf_counter() - t
meta = json.load(open(D + "/db_tiles.json"))
qs = json.load(open(D + "/queries.json"))[:21]; qv = np.load(D + "/ws_query_vecs.npy")[:21]
by = {x.tile_id: x for x in map(parse_geotile_filename, glob.glob("../tiles/2021_09/**/*.jpg", recursive=True))}
tiles = [by[m["tile_id"]] for m in meta]
m = SiftLightGlueMatcher(min_inliers=30)
print("fingerprint db: %s, %.0f MB, type %s (CPU RAM), loaded once in %.2f s" % (db.shape, db.nbytes / 1e6, type(db).__name__, load_s))
print("matcher device:", m.device, "| LightGlue weights on:", next(m.matcher.parameters()).device, "| GPU:", torch.cuda.get_device_name(0))
keys = ["photo read+decode", "search", "tile read+decode", "SIFT (CPU)", "LightGlue (GPU)", "homography (CPU)"]
T = dict.fromkeys(keys, 0.0)
def tick(k, t0):
    torch.cuda.synchronize(); T[k] += time.perf_counter() - t0
n = 0
for i, (q, v) in enumerate(zip(qs, qv)):
    if i == 1:
        T = dict.fromkeys(keys, 0.0); n = 0   # drop the warm-up photo
    t0 = time.perf_counter(); photo = np.array(Image.open(D + "/queries/" + q["file"]).convert("RGB")); tick(keys[0], t0)
    t0 = time.perf_counter(); cand = np.argsort(-(db @ v).max(axis=1))[:15]; tick(keys[1], t0)
    t0 = time.perf_counter(); pf = m._extract(photo); tick(keys[3], t0)
    for idx in cand:
        t0 = time.perf_counter(); img = np.array(Image.open(tiles[idx].image_path).convert("RGB")); tick(keys[2], t0)
        t0 = time.perf_counter(); tf = m._extract(img); tick(keys[3], t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.matcher({"image0": pf, "image1": tf})
        tick(keys[4], t0)
        t0 = time.perf_counter()
        mt = rbd(out)["matches"]
        if mt.shape[0] >= 4:
            a = rbd(pf)["keypoints"][mt[:, 0]].cpu().numpy(); b = rbd(tf)["keypoints"][mt[:, 1]].cpu().numpy()
            cv2.findHomography(a, b, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, maxIters=3, confidence=0.999)
        tick(keys[5], t0)
    n += 1
tot = sum(T.values())
print("photo %s px, tile %s px | %d photos" % (photo.shape[:2], img.shape[:2], n))
for k in keys:
    print("  %-20s %7.0f ms/photo  %5.1f%%" % (k, 1e3 * T[k] / n, 100 * T[k] / tot))
print("  %-20s %7.0f ms/photo" % ("total", 1e3 * tot / n))
