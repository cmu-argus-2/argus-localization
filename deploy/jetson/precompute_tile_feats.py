"""Compute the matcher's SIFT features for every reference tile once, ahead of time.

Tiles never change, so their features do not need to be recomputed for each photo.
Uses the repo's SiftLightGlueMatcher, so the features are exactly what the matcher
would compute itself. e2e_match.py reads them with --tile-feats.

Run from the argus-localization repo root:
    python deploy/jetson/precompute_tile_feats.py \
        --tiles-dir /mnt/sdc1/astroloc/data/database/2021_09 --out ~/tile_feats_2021_09

Writes one <tile_id>.npz per tile (keypoints, scales, orientations, descriptors,
scores, image size, and the tile's pixel shape). Tiles already done are skipped.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True, help="folder holding the tile images")
    ap.add_argument("--out", required=True, help="folder to write the features to")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.tiles_dir), "**", "*.jpg"), recursive=True))
    tiles = [parse_geotile_filename(p) for p in paths]
    matcher = SiftLightGlueMatcher()
    print(f"tiles: {len(tiles)} | matcher on: {matcher.device}")

    t0, done, points = time.time(), 0, 0
    for i, tile in enumerate(tiles):
        path = os.path.join(out, tile.tile_id + ".npz")
        if not os.path.exists(path):
            img = np.array(Image.open(tile.image_path).convert("RGB"))
            feats = {k: v.cpu().numpy() for k, v in matcher._extract(img).items()}
            points += feats["keypoints"].shape[1]
            # write to a temporary name first so an interrupted run leaves no half-written file
            np.savez(path + ".tmp.npz", tile_shape=np.array(img.shape[:2]), **feats)
            os.replace(path + ".tmp.npz", path)
            done += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(tiles)} tiles, {time.time() - t0:.0f} s", flush=True)

    size_mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"computed {done} tiles in {time.time() - t0:.0f} s"
          + (f" ({1e3 * (time.time() - t0) / done:.0f} ms/tile, {points / done:.0f} points/tile)" if done else ""))
    print(f"features on disk: {size_mb:.0f} MB in {out}")


if __name__ == "__main__":
    main()
