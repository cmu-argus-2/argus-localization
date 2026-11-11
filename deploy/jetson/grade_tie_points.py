"""Grade localization on the Argus-like GeoTIFF frames (prepare_geotiff_set.py) by its tie points.

Each frame is the centre square of a GeoTIFF resampled to 1024 x 1024, so the true ground point
of any frame pixel follows exactly from the GeoTIFF's geotransform. For every fix this compares
each tie point's lat/lon (from e2e_match.py --save-tie-points) with the true ground point of its
pixel, and prints that next to the logged fix error (footprint centre from the homography
extrapolated to the frame corners).

Run on the workstation (needs the GeoTIFFs and rasterio):
    ~/venv-export/bin/python grade_tie_points.py ~/e2e_geotiff match_results_geo_tp.json
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

GEOTIFF_ROOT = "/mnt/sda2/geotiffs"


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def main():
    data, results = os.path.expanduser(sys.argv[1]), sys.argv[2]
    zone = {q["file"]: q["zone"] for q in json.load(open(os.path.join(data, "queries.json")))}
    recs = json.load(open(os.path.join(data, results)))["records"]
    per_fix, logged, out = [], [], []
    for r in recs:
        if not r.get("tie_points"):
            continue
        stub = [f for f in os.path.splitext(r["file"])[0].split("@") if f][8]
        tp = np.array(r["tie_points"], dtype=float)
        W = r["photo_shape"][1]
        with rasterio.open(os.path.join(GEOTIFF_ROOT, zone[r["file"]], stub + ".tif")) as src:
            h, w = src.height, src.width
            c0 = (w - h) // 2
            col, row = c0 + (tp[:, 0] + 0.5) * h / W, (tp[:, 1] + 0.5) * h / W
            xs, ys = src.transform * (col, row)
            lon, lat = warp_transform(src.crs, "EPSG:4326", list(xs), list(ys))
        e = haversine_km(np.array(lat), np.array(lon), tp[:, 2], tp[:, 3])
        per_fix.append(float(np.median(e)))
        logged.append(r["match_error_km"])
        out.append({"file": r["file"], "logged_km": r["match_error_km"], "tie_median_km": per_fix[-1],
                    "tie_p90_km": float(np.percentile(e, 90)), "n": len(tp),
                    "matched_tile_correct": r.get("matched_tile_correct")})
    per_fix, logged = np.array(per_fix), np.array(logged)
    print(f"{len(per_fix)} fixes of {len(recs)} frames")
    print(f"{'':28} {'median':>7} {'p90':>7} {'max':>7} {'>10 km':>7} {'>50 km':>7}")
    for name, v in [("logged (footprint centre)", logged), ("tie points (median per fix)", per_fix)]:
        print(f"{name:28} {np.median(v):7.1f} {np.percentile(v, 90):7.1f} {v.max():7.1f} "
              f"{(v > 10).sum():7d} {(v > 50).sum():7d}")
    json.dump(out, open(os.path.join(data, "tie_point_grades.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
