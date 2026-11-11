"""Check batched_lightglue (e2e_match.py) against LightGlue run one tile at a time.

For each photo, the top-10 tiles are matched three ways:
  A. one tile at a time, point pruning OFF  -> must equal C (same maths)
  B. one tile at a time, repo default (pruning ON)
  C. batched_lightglue, all 10 tiles in one call
Reports matches and MAGSAC inliers per tile. Works on CPU or GPU.

Run from the argus-localization repo root:
    python deploy/jetson/test_batched_lightglue.py --data ~/e2e_alps_full \
        --tiles-dir /mnt/sdc1/astroloc/data/database/2021_09 --photos 5
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_match import batched_lightglue, homography_check  # noqa: E402
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from lightglue.utils import numpy_image_to_torch  # noqa: E402
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--photos", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--mp", action="store_true", help="LightGlue mixed precision (GPU only)")
    args = ap.parse_args()
    data = os.path.expanduser(args.data)

    db = np.load(os.path.join(data, "db_vecs.npy"))
    meta = json.load(open(os.path.join(data, "db_tiles.json")))
    queries = json.load(open(os.path.join(data, "queries.json")))
    qvecs = np.load(os.path.join(data, "ws_query_vecs.npy"))
    by_id = {t.tile_id: t for t in map(parse_geotile_filename,
                                       glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))}
    tiles = [by_id[m["tile_id"]] for m in meta]

    matcher = SiftLightGlueMatcher()
    lg = matcher.matcher
    lg.conf.mp = args.mp
    print(f"device: {matcher.device} | mixed precision: {args.mp}")

    def extract(img):
        with torch.no_grad():
            return matcher.extractor.extract(numpy_image_to_torch(img).to(matcher.device), resize=matcher.img_size)

    def one_at_a_time(pf, tf, pruning):
        saved = lg.conf.width_confidence
        lg.conf.width_confidence = saved if pruning else -1
        with torch.no_grad():
            out = lg({"image0": pf, "image1": tf})
        lg.conf.width_confidence = saved
        return out["matches"][0]

    same_ac, n_pairs, t_a, t_c = 0, 0, 0.0, 0.0
    best_agree_ab = best_agree_bc = 0
    for qi, (qm, qv) in enumerate(zip(queries[: args.photos], qvecs)):
        photo = np.array(Image.open(os.path.join(data, "queries", qm["file"])).convert("RGB"))
        cand = np.argsort(-(db @ qv).max(axis=1))[: args.top_k]
        pf = extract(photo)
        tfs = [extract(np.array(Image.open(tiles[i].image_path).convert("RGB"))) for i in cand]
        kp0 = pf["keypoints"][0]

        t0 = time.perf_counter()
        a = [one_at_a_time(pf, tf, pruning=False) for tf in tfs]
        t_a += time.perf_counter() - t0
        b = [one_at_a_time(pf, tf, pruning=True) for tf in tfs]
        t0 = time.perf_counter()
        with torch.no_grad():
            c = batched_lightglue(lg, pf, tfs)
        t_c += time.perf_counter() - t0

        def inliers(ms, tf):
            qp = kp0[ms[:, 0]].float().cpu().numpy()
            tp = tf["keypoints"][0][ms[:, 1]].float().cpu().numpy()
            return homography_check(qp, tp, matcher.max_ransac_iters).num_inliers

        rows = []
        for k, tf in enumerate(tfs):
            sa = {tuple(x) for x in a[k].tolist()}
            sc = {tuple(x) for x in c[k].tolist()}
            same_ac += int(sa == sc)
            n_pairs += 1
            rows.append((len(a[k]), len(b[k]), len(c[k]), inliers(a[k], tf), inliers(b[k], tf), inliers(c[k], tf),
                         len(sa & sc) / max(len(sa | sc), 1)))
        best = [int(np.argmax([r[i] for r in rows])) for i in (3, 4, 5)]
        best_agree_ab += int(best[0] == best[1])
        best_agree_bc += int(best[1] == best[2])
        print(f"\nphoto {qi}: matches A/B/C and inliers A/B/C per tile (overlap A vs C)")
        for k, r in enumerate(rows):
            print(f"  tile #{k + 1:2d}: matches {r[0]:4d} {r[1]:4d} {r[2]:4d} | inliers {r[3]:4d} {r[4]:4d} {r[5]:4d} | A∩C {r[6]:.2f}")
        print(f"  winning tile: A #{best[0] + 1}, B #{best[1] + 1}, C #{best[2] + 1}")

    print(f"\nA (one at a time, no pruning) == C (batched) on {same_ac}/{n_pairs} tile pairs")
    print(f"same winning tile: A vs B {best_agree_ab}/{args.photos}, B vs C {best_agree_bc}/{args.photos}")
    print(f"LightGlue time per photo on {matcher.device}: one at a time {t_a / args.photos:.2f} s, batched {t_c / args.photos:.2f} s")


if __name__ == "__main__":
    main()
