"""Render the frames of a simulated Argus pass from the Landsat GeoTIFFs, as a test set for
e2e_match.py (same layout as prepare_geotiff_set.py's).

For each pose in frames_poses.json (simulate_pass.py --poses-only), every pixel of the
1024 x 1024 nadir frame is ray-cast to the WGS84 ellipsoid and its colour is sampled from
whichever GeoTIFFs cover that ground point (read at 350 m/px, averaged from 175 m/px).
This is a true perspective view along the orbit, not a crop. Frames more than
--max-nodata empty are skipped. Truth for each frame: its four corner lat/lon (the
filename, like the other test sets) and the database tiles overlapping it (IoU >= 0.2).

Run on the workstation from the argus-localization repo root (CPU only):
    ~/venv-export/bin/python deploy/jetson/od/render_pass.py --poses RUN/frames_poses.json --out ~/e2e_pass_eu
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
from rasterio.warp import transform_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from core.types import GeoTile  # noqa: E402
from data_loading.earthloc_loader import parse_geotile_filename  # noqa: E402
from e2e_prepare import bbox, polygon, positive_indices  # noqa: E402
from simulate_pass import ecef_to_geodetic, ray_ellipsoid  # noqa: E402

GEOTIFF_ROOT = "/mnt/sda2/geotiffs"


def tif_index(zones, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    idx = []
    for z in zones:
        for f in sorted(glob.glob(os.path.join(GEOTIFF_ROOT, z, "*.tif"))):
            with rasterio.open(f) as src:
                idx.append([f, list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))])
    json.dump(idx, open(cache, "w"))
    return idx


_cache = {}


def load_tif(path, decimate):
    if path not in _cache:
        if len(_cache) > 40:
            _cache.clear()
        with rasterio.open(path) as src:
            h, w = src.height // decimate, src.width // decimate
            arr = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
            tr = src.transform * src.transform.scale(src.width / w, src.height / h)
            _cache[path] = (np.ascontiguousarray(arr.transpose(1, 2, 0)), tr, src.crs)
    return _cache[path]


def render(pose, tifs, decimate):
    W, fx = pose["size"], pose["fx"]
    c = (W - 1) / 2
    uu, vv = np.meshgrid(np.arange(W), np.arange(W))
    b = np.stack([(uu - c) / fx, (vv - c) / fx, np.ones_like(uu, float)], -1).reshape(-1, 3)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    R = np.array(pose["body_to_ecef"])
    ground = ray_ellipsoid(np.array(pose["pos_ecef_km"]), b @ R.T)
    lat, lon = ecef_to_geodetic(ground)
    lat, lon = lat.reshape(W, W), lon.reshape(W, W)
    corners = np.array([[lat[W - 1, 0], lon[W - 1, 0]], [lat[0, 0], lon[0, 0]],
                        [lat[0, W - 1], lon[0, W - 1]], [lat[W - 1, W - 1], lon[W - 1, W - 1]]])  # BL TL TR BR
    lo0, lo1, la0, la1 = lon.min(), lon.max(), lat.min(), lat.max()
    cand = [f for f, (x0, y0, x1, y1) in tifs if x0 < lo1 and x1 > lo0 and y0 < la1 and y1 > la0]
    img = np.zeros((W * W, 3), np.uint8)
    todo = np.ones(W * W, bool)
    flat_lat, flat_lon = lat.ravel(), lon.ravel()
    # most central GeoTIFFs first
    cl, cn = lat.mean(), lon.mean()
    cand.sort(key=lambda f: next(abs((bb[0] + bb[2]) / 2 - cn) + abs((bb[1] + bb[3]) / 2 - cl) for p, bb in tifs if p == f))
    for f in cand:
        if not todo.any():
            break
        arr, tr, crs = load_tif(f, decimate)
        sel = np.flatnonzero(todo)
        xs, ys = warp_transform("EPSG:4326", crs, flat_lon[sel].tolist(), flat_lat[sel].tolist())
        col, row = ~tr * (np.array(xs), np.array(ys))
        col, row = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (col >= 0) & (row >= 0) & (col < arr.shape[1]) & (row < arr.shape[0])
        pix = np.zeros((len(sel), 3), np.uint8)
        pix[ok] = arr[row[ok], col[ok]]
        ok &= pix.sum(axis=1) > 0
        img[sel[ok]] = pix[ok]
        todo[sel[ok]] = False
    return img.reshape(W, W, 3), corners, float(todo.mean()), len(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--zones", nargs="+", default=["32S", "32T", "33S", "33T"])
    ap.add_argument("--tiles-dir", default="/mnt/sdc1/astroloc/data/database/2021_09")
    ap.add_argument("--decimate", type=int, default=2, help="read GeoTIFFs at 175 m x this")
    ap.add_argument("--max-nodata", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "queries"), exist_ok=True)

    tiles = [parse_geotile_filename(p) for p in sorted(glob.glob(os.path.join(args.tiles_dir, "**", "*.jpg"), recursive=True))]
    polys, bbs = [polygon(t) for t in tiles], np.array([bbox(t) for t in tiles])
    tifs = tif_index(args.zones, os.path.join(out, "geotiff_index.json"))
    poses = json.load(open(os.path.expanduser(args.poses)))["frames"]

    qmeta = []
    for p in poses:
        img, corners, nodata, ncand = render(p, tifs, args.decimate)
        status = f"{p['stub']}: {ncand} GeoTIFFs, {100 * nodata:.0f}% empty"
        if nodata > args.max_nodata:
            print(status + " -> skipped", flush=True)
            continue
        q = GeoTile(tile_id=p["stub"], image_path="", corners_latlon=corners)
        pos = positive_indices(q, polys, bbs, 0.2)
        c = corners.mean(axis=0)
        fields = [f"{v:.6f}" for latlon in corners for v in latlon]
        name = "@" + "@".join(fields + [p["stub"], "20210101", f"{c[0]:.1f}", f"{c[1]:.1f}", "0", "0"]) + "@.jpg"
        Image.fromarray(img).save(os.path.join(out, "queries", name), quality=95)
        qmeta.append({"file": name, "centre": c.tolist(), "positives": pos, "t": p["t"]})
        print(status + f", {len(pos)} correct database tiles", flush=True)
    json.dump(qmeta, open(os.path.join(out, "queries.json"), "w"))
    print(f"{len(qmeta)} of {len(poses)} frames rendered -> {out}")


if __name__ == "__main__":
    main()
