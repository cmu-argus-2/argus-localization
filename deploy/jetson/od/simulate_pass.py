"""Localization -> orbit determination on one simulated Argus pass.

1. Truth: a circular orbit (two-body + J2), nadir-pointing camera, gyro at 1 Hz.
2. Every --capture-s seconds a 1024 x 1024 frame is taken. A grid of its pixels is
   ray-cast to the WGS84 ellipsoid to get the true ground point of each pixel.
3. Localization is simulated from our measured results: a frame gets a fix with
   probability --fix-rate; a fixed frame's ground points are all shifted by one error
   drawn from the real fix errors on the 208 simulated Argus images (--errors-from),
   in a random direction, plus small per-point noise. Each point is a tie point
   (u, v, lat, lon), the same thing the localization pipeline outputs.
4. Tie points -> FSW-Payload's input files: landmark_measurements.csv (body-frame
   bearing from the pixel, landmark ECI from lat/lon) and imu_data.csv.
5. FSW-Payload's Python batch OD (python_od, CasADi/IPOPT) solves position, velocity,
   attitude and gyro bias; the result is compared with the truth.

  .venv/bin/python od/simulate_pass.py --out od/runs/sim_default
  .venv/bin/python od/simulate_pass.py --perfect --out od/runs/sim_perfect   # no localization error

Real localization instead of simulated errors (same orbit arguments in both steps):
  simulate_pass.py --poses-only --out RUN ...          # frame poses -> RUN/frames_poses.json
  render_pass.py (workstation) -> frames; e2e_match.py --save-tie-points (Jetson) -> results
  simulate_pass.py --tie-points-from RESULTS.json --out RUN ...

Simplifications: Earth rotation is GMST about z only (no precession, nutation or polar
motion), in both the simulation and the lat/lon -> ECI conversion; body frame = camera
frame; no drag in the truth or the solve.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# FSW-Payload checkout's python_od folder; override with FSW_PYTHON_OD=/path/to/FSW-Payload/python_od
PYTHON_OD = os.environ.get("FSW_PYTHON_OD", os.path.join(HERE, "..", "..", "repos", "FSW-Payload", "python_od"))
J2000_EPOCH_UNIX_S = 946727936.0   # same constant as FSW-Payload python_od/data_loader.py

MU = 398600.4418          # km^3/s^2
RE = 6378.137             # km, WGS84 equatorial radius
J2 = 1.08262668e-3
F_WGS84 = 1 / 298.257223563
RP = RE * (1 - F_WGS84)   # km, polar radius
E2 = F_WGS84 * (2 - F_WGS84)


# ---------- frames ----------
def gmst_rad(unix_s):
    jd = unix_s / 86400.0 + 2440587.5
    return np.deg2rad((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ecef_to_eci(unix_s):
    return rot_z(gmst_rad(unix_s))


def geodetic_to_ecef(lat_deg, lon_deg, h_km=0.0):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    n = RE / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(n + h_km) * np.cos(lat) * np.cos(lon),
                     (n + h_km) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2) + h_km) * np.sin(lat)], axis=-1)


def ecef_to_geodetic(p):
    """Bowring's method; p (..., 3) km on or near the surface."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    ep2 = (RE ** 2 - RP ** 2) / RP ** 2
    th = np.arctan2(z * RE, r * RP)
    lat = np.arctan2(z + ep2 * RP * np.sin(th) ** 3, r - E2 * RE * np.cos(th) ** 3)
    return np.rad2deg(lat), np.rad2deg(lon)


def ray_ellipsoid(origin, dirs):
    """First intersection of rays (ECEF, km) with the WGS84 ellipsoid; NaN if missed."""
    s = np.array([1 / RE, 1 / RE, 1 / RP])
    o, d = origin * s, dirs * s
    a = (d * d).sum(-1)
    b = 2 * (o * d).sum(-1)
    c = (o * o).sum() - 1
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * a)
    return origin + t[..., None] * dirs


# ---------- orbit and attitude ----------
def accel(r):
    rn = np.linalg.norm(r)
    a = -MU * r / rn ** 3
    z2 = (r[2] / rn) ** 2
    k = 1.5 * J2 * MU * RE ** 2 / rn ** 5
    return a + k * np.array([r[0] * (5 * z2 - 1), r[1] * (5 * z2 - 1), r[2] * (5 * z2 - 3)])


def rk4(r, v, dt):
    def f(r, v):
        return v, accel(r)
    k1 = f(r, v)
    k2 = f(r + dt / 2 * k1[0], v + dt / 2 * k1[1])
    k3 = f(r + dt / 2 * k2[0], v + dt / 2 * k2[1])
    k4 = f(r + dt * k3[0], v + dt * k3[1])
    return (r + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def initial_state(lat0, lon0, alt_km, inc_deg, ascending, unix0):
    """Circular orbit whose sub-satellite point is (lat0, lon0) at unix0."""
    i = np.deg2rad(inc_deg)
    lat = np.deg2rad(lat0)
    u = np.arcsin(np.clip(np.sin(lat) / np.sin(i), -1, 1))
    if not ascending:
        u = np.pi - u
    alpha = np.deg2rad(lon0) + gmst_rad(unix0)            # right ascension of the point
    raan = alpha - np.arctan2(np.cos(i) * np.sin(u), np.cos(u))
    rot = rot_z(raan) @ np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]]) @ rot_z(u)
    a = RE + alt_km
    return rot @ np.array([a, 0, 0]), rot @ np.array([0, np.sqrt(MU / a), 0])


def nadir_attitude(r, v):
    """Body(=camera)-to-ECI rotation: z to nadir, y along -orbit normal, x ~ along track."""
    z = -r / np.linalg.norm(r)
    h = np.cross(r, v)
    y = -h / np.linalg.norm(h)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def mat_to_quat_xyzw(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def log_so3(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-12:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2026-11-05T00:00:00Z")
    ap.add_argument("--lat0", type=float, default=30.0)
    ap.add_argument("--lon0", type=float, default=-120.0)
    ap.add_argument("--alt-km", type=float, default=600.0, help="FSW-Payload's OD initialiser assumes 600 km")
    ap.add_argument("--inc-deg", type=float, default=97.8, help="sun-synchronous at ~600 km")
    ap.add_argument("--descending", action="store_true")
    ap.add_argument("--duration-s", type=int, default=600)
    ap.add_argument("--capture-s", type=int, default=20)
    ap.add_argument("--fx", type=float, default=3548.0 * 1024 / 2592,
                    help="focal length in pixels for the 1024 x 1024 frame (Argus K, centre square of 2592 resized)")
    ap.add_argument("--grid", type=int, default=8, help="tie points per frame = grid^2")
    ap.add_argument("--fix-rate", type=float, default=0.596, help="measured on the 208 simulated Argus images")
    ap.add_argument("--errors-from", default=os.path.join(HERE, "geo_fix_errors_source.json"),
                    help="match_results JSON whose fix errors (km) are resampled")
    ap.add_argument("--point-noise-km", type=float, default=0.3)
    ap.add_argument("--sigma-rad", type=float, default=None,
                    help="per-measurement bearing sigma given to OD (default: median fix error / altitude)")
    ap.add_argument("--gyro-noise", type=float, default=0.0008726, help="rad/s, FSW-Payload's value")
    ap.add_argument("--gyro-bias", type=float, nargs=3, default=[0.002, -0.001, 0.0015], help="rad/s")
    ap.add_argument("--perfect", action="store_true", help="every frame fixed, no localization error")
    ap.add_argument("--covariance", action="store_true", help="also compute OD covariance (slow, ~3 min)")
    ap.add_argument("--poses-only", action="store_true",
                    help="write truth, gyro and every frame's pose (frames_poses.json) for render_pass.py, then stop")
    ap.add_argument("--tie-points-from", default=None,
                    help="e2e_match.py --save-tie-points results on the rendered frames: use their real tie points")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    unix0 = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    r, v = initial_state(args.lat0, args.lon0, args.alt_km, args.inc_deg, not args.descending, unix0)

    # truth at 1 Hz
    n = args.duration_s + 1
    ts = unix0 + np.arange(n, dtype=float)
    R_true, V_true, Rot = np.zeros((n, 3)), np.zeros((n, 3)), []
    for k in range(n):
        R_true[k], V_true[k] = r, v
        Rot.append(nadir_attitude(r, v))
        r, v = rk4(r, v, 1.0)
    # gyro: body-frame rate from consecutive attitudes, + bias + white noise
    gyro = np.zeros((n, 3))
    for k in range(n):
        k1 = min(k, n - 2)
        w = log_so3(Rot[k1].T @ Rot[k1 + 1])
        gyro[k] = w + np.array(args.gyro_bias) + rng.normal(0, args.gyro_noise, 3)

    W = 1024
    c = (W - 1) / 2
    if args.poses_only:
        poses = []
        for k in range(0, n, args.capture_s):
            C = ecef_to_eci(ts[k])
            poses.append({"stub": f"pass_t{k:05d}", "t": float(k), "unix_s": float(ts[k]),
                          "pos_ecef_km": (C.T @ R_true[k]).tolist(), "body_to_ecef": (C.T @ Rot[k]).tolist(),
                          "fx": args.fx, "size": W})
        json.dump({"settings": vars(args), "frames": poses}, open(os.path.join(args.out, "frames_poses.json"), "w"))
        np.savetxt(os.path.join(args.out, "truth.csv"), np.column_stack([ts, R_true, V_true]), delimiter=",",
                   header="unix_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s", comments="", fmt="%.6f")
        print(f"{len(poses)} frame poses -> {args.out}/frames_poses.json")
        return

    errs_km = np.array([rec["match_error_km"] for rec in json.load(open(args.errors_from))["records"]
                        if rec.get("fix") and rec.get("match_error_km") is not None])
    sigma = args.sigma_rad or float(np.median(errs_km)) / args.alt_km
    real = {}
    if args.tie_points_from:
        for rec in json.load(open(args.tie_points_from))["records"]:
            stub = [f for f in os.path.splitext(rec["file"])[0].split("@") if f][8]
            real[stub] = rec

    # frames and simulated (or real) localization
    g = np.linspace(0.08 * W, 0.92 * W, args.grid)
    uu, vv = np.meshgrid(g, g)
    uv = np.column_stack([uu.ravel(), vv.ravel()])
    rows, frames = [], []
    for k in range(0, n, args.capture_s):
        if args.tie_points_from:
            rec = real.get(f"pass_t{k:05d}")
            frame = {"t": float(ts[k] - unix0), "fix": bool(rec and rec["fix"] and rec.get("tie_points")),
                     "rendered": rec is not None}
            if frame["fix"]:
                tp = np.array(rec["tie_points"], dtype=float)
                b_cam = np.column_stack([(tp[:, 0] - c) / args.fx, (tp[:, 1] - c) / args.fx, np.ones(len(tp))])
                b_cam /= np.linalg.norm(b_cam, axis=1, keepdims=True)
                C = ecef_to_eci(ts[k])
                lm_eci = (C @ geodetic_to_ecef(tp[:, 2], tp[:, 3]).T).T
                frame.update({"error_km": float(rec["match_error_km"]), "inliers": int(rec["best_inliers"]),
                              "matched_tile_correct": bool(rec.get("matched_tile_correct")),
                              "centre_est": [float(tp[:, 2].mean()), float(tp[:, 3].mean())]})
                gid = len([f for f in frames if f["fix"]])
                for j in range(len(tp)):
                    rows.append([ts[k] * 1000, *b_cam[j], *lm_eci[j], gid, sigma])
            frames.append(frame)
            continue
        fixed = args.perfect or rng.random() < args.fix_rate
        frame = {"t": float(ts[k] - unix0), "fix": bool(fixed)}
        if fixed:
            jitter = rng.uniform(-20, 20, uv.shape)
            uvk = np.clip(uv + jitter, 0, W - 1)
            b_cam = np.column_stack([(uvk[:, 0] - c) / args.fx, (uvk[:, 1] - c) / args.fx, np.ones(len(uvk))])
            b_cam /= np.linalg.norm(b_cam, axis=1, keepdims=True)
            C = ecef_to_eci(ts[k])
            ground_ecef = ray_ellipsoid(C.T @ R_true[k], (C.T @ (Rot[k] @ b_cam.T)).T)
            lat, lon = ecef_to_geodetic(ground_ecef)
            # localization error: one shift for the whole frame + small per-point noise
            e_km = 0.0 if args.perfect else float(rng.choice(errs_km))
            ang = rng.uniform(0, 2 * np.pi)
            de = e_km * np.cos(ang) + (0 if args.perfect else 1) * rng.normal(0, args.point_noise_km, len(lat))
            dn = e_km * np.sin(ang) + (0 if args.perfect else 1) * rng.normal(0, args.point_noise_km, len(lat))
            lat_est = lat + np.rad2deg(dn / RE)
            lon_est = lon + np.rad2deg(de / (RE * np.cos(np.deg2rad(lat))))
            lm_eci = (C @ geodetic_to_ecef(lat_est, lon_est).T).T
            frame.update({"error_km": e_km, "centre_true": [float(lat.mean()), float(lon.mean())],
                          "centre_est": [float(lat_est.mean()), float(lon_est.mean())]})
            gid = len([f for f in frames if f["fix"]])
            for j in range(len(lat)):
                rows.append([ts[k] * 1000, *b_cam[j], *lm_eci[j], gid, sigma])
        frames.append(frame)

    with open(os.path.join(args.out, "landmark_measurements.csv"), "w") as f:
        f.write("timestamp_ms,bearing_x,bearing_y,bearing_z,eci_x_km,eci_y_km,eci_z_km,group,sigma\n")
        for rw in rows:
            f.write(f"{rw[0]:.0f},{rw[1]:.9f},{rw[2]:.9f},{rw[3]:.9f},{rw[4]:.6f},{rw[5]:.6f},{rw[6]:.6f},{rw[7]},{rw[8]:.6g}\n")
    with open(os.path.join(args.out, "imu_data.csv"), "w") as f:
        f.write("Timestamp_ms,Gyro_X_dps,Gyro_Y_dps,Gyro_Z_dps\n")
        for k in range(n):
            f.write(f"{ts[k] * 1000:.0f},{np.rad2deg(gyro[k, 0]):.9f},{np.rad2deg(gyro[k, 1]):.9f},{np.rad2deg(gyro[k, 2]):.9f}\n")
    np.savetxt(os.path.join(args.out, "truth.csv"), np.column_stack([ts, R_true, V_true]), delimiter=",",
               header="unix_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s", comments="", fmt="%.6f")

    # solve with FSW-Payload's Python OD, reading the files just written
    sys.path.insert(0, os.path.abspath(PYTHON_OD))
    from data_loader import load_imu_csv, load_landmark_csv
    from optimizer import build_and_solve_with_outlier_rejection
    from residuals import IntegratorType
    lm, starts, unc = load_landmark_csv(os.path.join(args.out, "landmark_measurements.csv"))
    gy = load_imu_csv(os.path.join(args.out, "imu_data.csv"))
    res = build_and_solve_with_outlier_rejection(
        lm, starts, gy, landmark_uncertainties=unc, integrator_type=IntegratorType.RK4,
        use_j2=True, use_drag=False, compute_covariance=args.covariance, mahal_threshold=5.0,
        max_iterations=10, landmark_huber_M=3.0)

    t_est = np.asarray(res["state_timestamps"]) + J2000_EPOCH_UNIX_S
    idx = np.searchsorted(ts, t_est - 1e-6)
    pos_err = np.linalg.norm(np.asarray(res["positions"]) - R_true[idx], axis=1)
    vel_err = np.linalg.norm(np.asarray(res["velocities"]) - V_true[idx], axis=1) * 1000
    init_err = np.linalg.norm(np.asarray(res["init_positions"]) - R_true[idx], axis=1)
    fixed = [f for f in frames if f["fix"]]
    summary = {
        "frames": len(frames), "fixed_frames": len(fixed),
        "tie_points": len(rows), "sigma_rad": sigma,
        "frame_fix_error_km_median": float(np.median([f["error_km"] for f in fixed])) if fixed else None,
        "od_position_error_km": {"median": float(np.median(pos_err)), "max": float(pos_err.max()),
                                 "end": float(pos_err[-1])},
        "od_velocity_error_m_s": {"median": float(np.median(vel_err)), "max": float(vel_err.max())},
        "initial_guess_position_error_km_median": float(np.median(init_err)),
        "gyro_bias_true": args.gyro_bias, "gyro_bias_est": np.asarray(res["gyro_bias"]).ravel().tolist(),
        "outlier_landmarks": int(np.sum(res["lmk_outlier_flags"])),
        "od_iterations": int(res["n_od_iterations"]),
        "settings": vars(args),
    }
    json.dump({"summary": summary, "frames": frames}, open(os.path.join(args.out, "od_summary.json"), "w"), indent=1)
    np.savetxt(os.path.join(args.out, "od_states.csv"),
               np.column_stack([t_est, res["positions"], res["velocities"], pos_err, vel_err]), delimiter=",",
               header="unix_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s,pos_err_km,vel_err_m_s", comments="", fmt="%.6f")
    plot(args.out, ts, unix0, R_true, t_est, np.asarray(res["positions"]), pos_err, frames)

    s = summary
    print(f"frames {s['frames']}, fixed {s['fixed_frames']}, tie points {s['tie_points']}, sigma {sigma:.4f} rad")
    if fixed:
        print(f"per-frame localization error (median): {s['frame_fix_error_km_median']:.1f} km")
    print(f"OD position error: median {s['od_position_error_km']['median']:.2f} km, "
          f"max {s['od_position_error_km']['max']:.2f} km, end {s['od_position_error_km']['end']:.2f} km "
          f"(initial guess {s['initial_guess_position_error_km_median']:.1f} km)")
    print(f"OD velocity error: median {s['od_velocity_error_m_s']['median']:.1f} m/s")
    print(f"gyro bias true {np.round(args.gyro_bias, 5).tolist()} est {np.round(s['gyro_bias_est'], 5).tolist()}")
    print(f"outlier landmarks {s['outlier_landmarks']}, OD iterations {s['od_iterations']} -> {args.out}")


def plot(out, ts, unix0, R_true, t_est, R_est, pos_err, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def subpoint(Rs, tt):
        p = np.array([ecef_to_eci(t).T @ r for r, t in zip(Rs, tt)])
        lat, lon = ecef_to_geodetic(p / np.linalg.norm(p, axis=1, keepdims=True) * RE)
        return lat, lon

    lt, ln = subpoint(R_true, ts)
    le, lne = subpoint(R_est, t_est)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(ln, lt, "k-", lw=2, label="true ground track")
    ax[0].plot(lne, le, "C1--", lw=1.5, label="OD estimate")
    fx = [f for f in frames if f["fix"]]
    if fx:
        ax[0].scatter([f["centre_est"][1] for f in fx], [f["centre_est"][0] for f in fx], s=18, c="C0",
                      label="localization fix (" + ("mean of tie points" if "rendered" in fx[0] else "frame centre") + ")")
    ax[0].set_xlabel("longitude (deg)")
    ax[0].set_ylabel("latitude (deg)")
    ax[0].set_title("Ground track")
    ax[0].legend(fontsize=8)
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].plot(t_est - unix0, pos_err, "C1-")
    for f in frames:
        ax[1].axvline(f["t"], color="C0" if f["fix"] else "0.85", lw=0.8, zorder=0)
    ax[1].set_xlabel("time from start (s)  [blue: frame with fix, grey: no fix]")
    ax[1].set_ylabel("OD position error (km)")
    ax[1].set_title("OD position error vs truth")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "od_pass.png"), dpi=130)


if __name__ == "__main__":
    main()
