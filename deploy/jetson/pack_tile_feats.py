"""Pack the per-tile SIFT feature files (precompute_tile_feats.py) into one set of
arrays, in fingerprint-database order, so e2e_match.py can memory-map them or load
them all into GPU memory instead of opening 10 .npz files per photo.

Descriptors are stored in float16: with LightGlue in mixed precision (--lightglue-mp)
they are converted to float16 inside LightGlue anyway, so results are unchanged.

Run on the Jetson from ~/Desktop/astroloc:
    python3 pack_tile_feats.py --data e2e_alps_full --feats tile_feats_2021_09 --out tile_feats_pack
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with db_tiles.json (sets the tile order)")
    ap.add_argument("--feats", required=True, help="folder of <tile_id>.npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-keypoints", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tile_ids = [m["tile_id"] for m in json.load(open(os.path.join(args.data, "db_tiles.json")))]
    t, k = len(tile_ids), args.max_keypoints
    arrays = {
        "keypoints": np.lib.format.open_memmap(os.path.join(args.out, "keypoints.npy"), "w+", np.float32, (t, k, 2)),
        "scales": np.lib.format.open_memmap(os.path.join(args.out, "scales.npy"), "w+", np.float32, (t, k)),
        "oris": np.lib.format.open_memmap(os.path.join(args.out, "oris.npy"), "w+", np.float32, (t, k)),
        "keypoint_scores": np.lib.format.open_memmap(os.path.join(args.out, "keypoint_scores.npy"), "w+", np.float32, (t, k)),
        "descriptors": np.lib.format.open_memmap(os.path.join(args.out, "descriptors.npy"), "w+", np.float16, (t, k, 128)),
    }
    counts = np.zeros(t, np.int32)
    image_size = np.zeros((t, 2), np.float32)
    tile_shape = np.zeros((t, 2), np.int32)
    for i, tid in enumerate(tile_ids):
        st = np.load(os.path.join(args.feats, tid + ".npz"))
        n = min(st["keypoints"].shape[1], k)
        counts[i] = n
        for key, arr in arrays.items():
            arr[i, :n] = st[key][0, :n]
        image_size[i] = st["image_size"][0]
        tile_shape[i] = st["tile_shape"]
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{t}", flush=True)
    for arr in arrays.values():
        arr.flush()
    np.save(os.path.join(args.out, "counts.npy"), counts)
    np.save(os.path.join(args.out, "image_size.npy"), image_size)
    np.save(os.path.join(args.out, "tile_shape.npy"), tile_shape)
    json.dump(tile_ids, open(os.path.join(args.out, "tile_ids.json"), "w"))
    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e9
    print(f"packed {t} tiles, {counts.mean():.0f} points/tile on average, {size:.2f} GB in {args.out}")


if __name__ == "__main__":
    main()
