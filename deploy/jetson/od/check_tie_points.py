"""Check a rendered pass's localization against truth, per tie point.

For each fix, every tie point's pixel (u, v) is ray-cast from the frame's true pose to the
ground; the distance to the tie point's lat/lon is that point's real error (what OD sees).
Also prints the logged fix error (footprint corners extrapolated from the homography), the
gap between the two truth centres (mean of corners vs centre pixel), and the error of the
centre pixel mapped through a least-squares homography fitted on the tie points.

  .venv/bin/python od/check_tie_points.py   (from this folder; give the run folder as the argument)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_pass import ray_ellipsoid, ecef_to_geodetic, geodetic_to_ecef

R = sys.argv[1] if len(sys.argv) > 1 else "od/runs/real_eu/"
poses = {p["stub"]: p for p in json.load(open(R + "frames_poses.json"))["frames"]}
q = {x["file"]: x for x in json.load(open(R + "e2e_pass_eu/queries.json"))}
recs = json.load(open(R + "match_results_pass.json"))["records"]

def ground(p, uv):
    W, fx = p["size"], p["fx"]; c = (W - 1) / 2
    b = np.column_stack([(uv[:, 0] - c) / fx, (uv[:, 1] - c) / fx, np.ones(len(uv))])
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    g = ray_ellipsoid(np.array(p["pos_ecef_km"]), b @ np.array(p["body_to_ecef"]).T)
    return g

def km(a, b): return np.linalg.norm(a - b, axis=-1)

def dlt(uv, xy):
    A = []
    for (u, v), (x, y) in zip(uv, xy):
        A += [[-u, -v, -1, 0, 0, 0, u * x, v * x, x], [0, 0, 0, -u, -v, -1, u * y, v * y, y]]
    H = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    return H

print(f"{'frame':12} {'logged':>7} {'ctr-pt vs cornermean':>8} {'DLT ctr':>8} {'tie pts med/p90/max km':>24} {'v range':>10}")
rows = []
for r in recs:
    if not r.get("tie_points"): continue
    stub = [f for f in os.path.splitext(r["file"])[0].split("@") if f][8]
    p = poses[stub]
    tp = np.array(r["tie_points"])
    true_g = ground(p, tp[:, :2])
    est_g = geodetic_to_ecef(tp[:, 2], tp[:, 3])
    e = km(true_g, est_g)
    # truth centre: corner mean (queries.json) vs ground point of centre pixel
    cm = q[r["file"]]["centre"]
    cp = ground(p, np.array([[511.5, 511.5]]))[0]
    cm_e = geodetic_to_ecef(np.array([cm[0]]), np.array([cm[1]]))[0]
    # DLT on tie points in local ENU-ish plane (ECEF difference is fine locally? use lat/lon scaled)
    lat0, lon0 = tp[:, 2].mean(), tp[:, 3].mean()
    xy = np.column_stack([(tp[:, 3] - lon0) * np.cos(np.radians(lat0)), tp[:, 2] - lat0])
    H = dlt(tp[:, :2], xy)
    h = H @ np.array([511.5, 511.5, 1.0]); h = h[:2] / h[2]
    dc = geodetic_to_ecef(np.array([h[1] + lat0]), np.array([h[0] / np.cos(np.radians(lat0)) + lon0]))[0]
    rows.append(np.median(e))
    print(f"{stub:12} {r['match_error_km']:7.1f} {km(cp, cm_e):8.1f} {km(dc, cp):8.1f}     {np.median(e):6.1f} {np.percentile(e,90):6.1f} {e.max():7.1f}   {tp[:,1].min():4.0f}-{tp[:,1].max():4.0f}")
print("median of per-frame tie-point medians:", np.median(rows))
