"""Find candidate same-pass photo pairs for two-frame OD, and draw a sample to time-check.

A pair = two ISS astronaut photos from the same mission and date, from the held-out
regions (nadir within 2,000 km of one of the 6 evaluation centres), whose footprint
centres are 200-1,000 km apart, at most 200 frame numbers apart, and both with a
correct tile in the 2021 zoom-9 database. The sample takes one pair per photo,
spread across regions.

Run on the workstation from the argus-localization repo root (CPU only):
    python <pkg>/two_frame/find_pairs.py --out-dir ~/pairs --sample 40
Writes pairs_heldout_all.json and pairs_heldout_sample<N>.json.
"""
import argparse, collections, glob, json, os, random, re, sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from astroloc.data.regions import EVAL_REGIONS  # noqa: E402
from data_loading.earthloc_loader import load_query_set, parse_geotile_filename  # noqa: E402
from e2e_prepare import bbox, haversine_km, polygon, positive_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--tiles-dir", default="/mnt/sdc1/astroloc/data/database/2021_09")
    ap.add_argument("--unseen-km", type=float, default=2000)
    ap.add_argument("--min-km", type=float, default=200)
    ap.add_argument("--max-km", type=float, default=1000)
    ap.add_argument("--max-frame-gap", type=int, default=200)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    qs = load_query_set(args.queries_dir)
    tiles = [parse_geotile_filename(p) for p in sorted(glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))]
    polys, bbs = [polygon(t) for t in tiles], np.array([bbox(t) for t in tiles])

    def region_of(q):
        return min(EVAL_REGIONS, key=lambda r: haversine_km(q.meta["nadir_lat"], q.meta["nadir_lon"], *EVAL_REGIONS[r]))

    def unseen(q):
        return haversine_km(q.meta["nadir_lat"], q.meta["nadir_lon"], *EVAL_REGIONS[region_of(q)]) < args.unseen_km

    seq = collections.defaultdict(list)
    for q in qs:
        g = re.match(r"([A-Z0-9]+)-E-(\d+)", q.tile_id.split("@")[0])
        if g and unseen(q):
            seq[(g.group(1), q.timestamp)].append((int(g.group(2)), q))
    pos = {}

    def has_pos(q):
        if q.tile_id not in pos:
            pos[q.tile_id] = bool(positive_indices(q, polys, bbs, 0.2))
        return pos[q.tile_id]

    pairs = []
    for (mission, day), v in seq.items():
        v.sort(key=lambda x: x[0])
        for i, (fa, a) in enumerate(v):
            for fb, b in v[i + 1:]:
                if fb - fa > args.max_frame_gap:
                    break
                ca, cb = a.corners_latlon.mean(0), b.corners_latlon.mean(0)
                d = haversine_km(ca[0], ca[1], cb[0], cb[1])
                if args.min_km <= d <= args.max_km and has_pos(a) and has_pos(b):
                    pairs.append({"mission": mission, "date": day, "frame_a": fa, "frame_b": fb, "gap_frames": fb - fa,
                                  "centre_km": round(d, 1), "a": os.path.basename(a.image_path),
                                  "b": os.path.basename(b.image_path), "region": region_of(a)})
    json.dump(pairs, open(os.path.join(out, "pairs_heldout_all.json"), "w"), indent=0)
    print(f"{len(pairs)} candidate pairs; by region: {collections.Counter(p['region'] for p in pairs).most_common()}")

    random.seed(args.seed)
    by_region = collections.defaultdict(list)
    for p in pairs:
        by_region[p["region"]].append(p)
    picked, used = [], set()
    while len(picked) < args.sample and any(by_region.values()):
        for r in sorted(by_region, key=lambda r: len(by_region[r])):
            cands = [p for p in by_region[r] if p["a"] not in used and p["b"] not in used]
            if cands and len(picked) < args.sample:
                p = random.choice(cands)
                picked.append(p)
                used |= {p["a"], p["b"]}
            by_region[r] = [p for p in by_region[r] if p["a"] not in used and p["b"] not in used]
    path = os.path.join(out, f"pairs_heldout_sample{args.sample}.json")
    json.dump(picked, open(path, "w"), indent=1)
    print(f"sample of {len(picked)} pairs -> {path}")


if __name__ == "__main__":
    main()
