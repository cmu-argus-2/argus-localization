"""Build a test folder (same layout as e2e_prepare.py's) from the confirmed same-pass pairs,
so e2e_match.py can localize every photo. db_vecs.npy / db_tiles.json are not copied:
link them from an existing e2e folder built with the same model.

Run on the workstation from the argus-localization repo root:
    python <pkg>/two_frame/prepare_pair_set.py --pairs pairs_heldout_sample40.json \
        --db-tiles ~/e2e_alps_full/db_tiles.json --out ~/e2e_pairs
"""
import argparse, glob, json, os, shutil, sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from e2e_prepare import bbox, polygon, positive_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--db-tiles", required=True, help="db_tiles.json of the fingerprint database to use")
    ap.add_argument("--queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--tiles-dir", default="/mnt/sdc1/astroloc/data/database/2021_09")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "queries"), exist_ok=True)

    pairs = [p for p in json.load(open(args.pairs)) if p.get("status") == "same pass"]
    meta = json.load(open(os.path.expanduser(args.db_tiles)))
    tiles = [parse_geotile_filename(p) for p in sorted(glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))]
    if [t.tile_id for t in tiles] != [m["tile_id"] for m in meta]:
        sys.exit("ERROR: tile order differs from db_tiles.json")
    polys, bbs = [polygon(t) for t in tiles], np.array([bbox(t) for t in tiles])
    index = {f: os.path.join(r, f) for r, _, fs in os.walk(args.queries_dir) for f in fs}

    qmeta, seen = [], set()
    for p in pairs:
        for side in ("a", "b"):
            f = p[side]
            if f in seen:
                continue
            seen.add(f)
            q = parse_geotile_filename(index[f])
            shutil.copy(index[f], os.path.join(out, "queries", f))
            qmeta.append({"file": f, "centre": q.corners_latlon.mean(0).tolist(),
                          "positives": positive_indices(q, polys, bbs, 0.2)})
    json.dump(qmeta, open(os.path.join(out, "queries.json"), "w"))
    json.dump(pairs, open(os.path.join(out, "pairs.json"), "w"), indent=1)
    print(f"{len(pairs)} pairs, {len(qmeta)} photos -> {out} (now link db_vecs.npy and db_tiles.json into it)")


if __name__ == "__main__":
    main()
