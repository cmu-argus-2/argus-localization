"""Build a test set of square, Argus-like frames from the Landsat-8 GeoTIFFs
(/mnt/sda2/geotiffs), in the same layout as e2e_prepare.py's, so e2e_match.py can run the
full flow on them (Jetson or workstation).

Each frame is a simulated Argus image (~4609 x 2593 px, 175 m/px). We take its centre
square (the full height, ~454 km across, as the team's square-image decision), resample it
to 1024 x 1024 like the astronaut photos, and get its exact ground footprint from the
GeoTIFF's own geotransform. Frames more than half no-data are skipped, as in
astroloc/eval/evaluate_argus_geotiffs.py. Only frames with a correct tile (IoU >= 0.2) in
the 2021 zoom-9 database are kept.

The fingerprint database is not rebuilt: link db_vecs.npy and db_tiles.json from an
existing e2e folder built with the same model.

Run on the workstation from the argus-localization repo root (CPU only):
    python deploy/jetson/prepare_geotiff_set.py --per-zone 13 --out ~/e2e_geotiff
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.types import GeoTile  # noqa: E402
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from e2e_prepare import bbox, polygon, positive_indices  # noqa: E402

GEOTIFF_ROOT = "/mnt/sda2/geotiffs"


def centre_square(path, size):
    """Centre square of a frame, resampled to size x size, and its corner lat/lon (BL, TL, TR, BR)."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        c0 = (w - h) // 2
        win = rasterio.windows.Window(c0, 0, h, h)
        arr = src.read([1, 2, 3], window=win, out_shape=(3, size, size), resampling=Resampling.average)
        px = [(c0, h), (c0, 0), (c0 + h, 0), (c0 + h, h)]
        xs, ys = zip(*[src.transform * p for p in px])
        lons, lats = warp_transform(src.crs, "EPSG:4326", list(xs), list(ys))
    img = np.ascontiguousarray(arr.transpose(1, 2, 0))
    return img, np.array(list(zip(lats, lons)))


def to_uint8(img):
    """Landsat composites may not be 8-bit; stretch each frame to 0-255 (2nd-98th percentile)."""
    if img.dtype == np.uint8:
        return img
    valid = img.sum(axis=2) > 0
    lo, hi = np.percentile(img[valid], [2, 98]) if valid.any() else (0, 1)
    return (np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-zone", type=int, default=13)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--tiles-dir", default="/mnt/sdc1/astroloc/data/database/2021_09")
    ap.add_argument("--max-nodata", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "queries"), exist_ok=True)

    tiles = [parse_geotile_filename(p) for p in sorted(glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))]
    polys, bbs = [polygon(t) for t in tiles], np.array([bbox(t) for t in tiles])

    zones = sorted(d for d in os.listdir(GEOTIFF_ROOT) if os.path.isdir(os.path.join(GEOTIFF_ROOT, d)))
    qmeta, skipped_nodata, skipped_nopos = [], 0, 0
    for zone in zones:
        files = sorted(glob.glob(os.path.join(GEOTIFF_ROOT, zone, "*.tif")))
        for i in sorted(set(np.linspace(0, len(files) - 1, min(args.per_zone, len(files))).round().astype(int))):
            img, corners = centre_square(files[i], args.size)
            if (img.sum(axis=2) == 0).mean() > args.max_nodata:
                skipped_nodata += 1
                continue
            q = GeoTile(tile_id=os.path.basename(files[i]), image_path=files[i], corners_latlon=corners)
            pos = positive_indices(q, polys, bbs, 0.2)
            if not pos:
                skipped_nopos += 1
                continue
            c = corners.mean(axis=0)
            fields = [f"{v:.6f}" for latlon in corners for v in latlon]
            name = "@" + "@".join(fields + [os.path.splitext(os.path.basename(files[i]))[0], "20210101",
                                            f"{c[0]:.1f}", f"{c[1]:.1f}", "0", "0"]) + "@.jpg"
            Image.fromarray(to_uint8(img)).save(os.path.join(out, "queries", name), quality=95)
            qmeta.append({"file": name, "centre": c.tolist(), "positives": pos, "zone": zone})
        print(f"  {zone}: {len(qmeta)} frames so far", flush=True)
    json.dump(qmeta, open(os.path.join(out, "queries.json"), "w"))
    print(f"{len(qmeta)} frames kept; skipped {skipped_nodata} mostly no-data, {skipped_nopos} with no correct zoom-9 tile")


if __name__ == "__main__":
    main()
