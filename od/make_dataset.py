#!/usr/bin/env python3
"""
Writes a synthetic OD dataset folder in the flight OD's own input format
(landmark_measurements.csv + imu_data.csv, see od/data_loader.py), plus a
truth.csv the flight format doesn't have, so od/ipopt_od.py can score itself.

What is simulated vs. real:
  • Orbit: 420 km, 51.6 deg circular start, propagated with two-body + J2
    (RK4, 1 s substeps) -- the same physics od/residuals.py models with --j2.
  • Attitude: nadir-pointing (body z = nadir, body x = along-track), which is
    what the bearing cone in simulate_bearing_measurements already assumes.
  • Gyro: true body rate from consecutive attitudes, plus a constant bias and
    white noise at the flight optimizer's own GYRO_WN_STD_DEV.
  • Landmark bearings: integration/od_integration_test.py's
    simulate_bearing_measurements, so the bearing noise is bootstrapped from
    our real faithful_lora_v2 error dataset (except --noise legacy), and sigma
    comes from our fitted error models. Bearings are rotated ECI -> body
    before writing, because the flight OD estimates attitude itself.

Usage (from the repository root, pri_env):
    python -m od.make_dataset output/od_datasets/sim_regression --noise regression
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from integration.od_integration_test import (
    confidence_tier_sigma_rad,
    load_empirical_angular_noise_pool_rad,
    load_empirical_full_pool,
    load_empirical_noise_and_confidence,
    load_fitted_error_regression,
    load_svr_error_model,
    simulate_bearing_measurements,
)
from od.data_loader import write_imu_csv, write_landmark_csv
from od.optimizer import GYRO_WN_STD_DEV
from od.residuals import GM_EARTH, J2, R_EARTH

EPOCH_UNIX_MS = 1767225600000  # 2026-01-01T00:00:00Z
ALTITUDE_KM = 420.0
INCLINATION_DEG = 51.6
ISS_RANGE_KM = 400.0
LEGACY_ERROR_KM = 3.0
TRUE_GYRO_BIAS_RAD_S = np.array([4e-4, -2e-4, 3e-4])
MODEL_NAME = "faithful_lora_v2"


def _accel(r: np.ndarray) -> np.ndarray:
    r_sq = r @ r
    r_norm = np.sqrt(r_sq)
    a = -GM_EARTH / r_norm**3 * r
    factor = 1.5 * J2 * GM_EARTH * R_EARTH**2 / (r_sq * r_sq * r_norm)
    z2 = r[2] ** 2 / r_sq
    return a + factor * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])


def propagate_truth(n_states: int, dt: float, substep: float = 1.0) -> np.ndarray:
    """(n_states, 6) [r_km, v_km_s], two-body + J2, RK4."""
    r0 = R_EARTH + ALTITUDE_KM
    v0 = np.sqrt(GM_EARTH / r0)
    inc = np.radians(INCLINATION_DEG)
    x = np.array([r0, 0.0, 0.0, 0.0, v0 * np.cos(inc), v0 * np.sin(inc)])

    def f(s):
        return np.concatenate([s[3:], _accel(s[:3])])

    n_sub = int(round(dt / substep))
    h = dt / n_sub
    states = np.zeros((n_states, 6))
    states[0] = x
    for i in range(1, n_states):
        for _ in range(n_sub):
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        states[i] = x
    return states


def nadir_attitude(states: np.ndarray) -> Rotation:
    """Body-to-ECI rotations: body z = nadir, body x = along-track."""
    z = -states[:, :3] / np.linalg.norm(states[:, :3], axis=1, keepdims=True)
    v = states[:, 3:]
    x = v - np.sum(v * z, axis=1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.stack([x, y, z], axis=2))


def simulate_gyro(attitude: Rotation, dt: float, rng: np.random.Generator) -> np.ndarray:
    """(M, 3) measured body rates [rad/s]: true rate + bias + white noise.
    The rate at state i is the one that carries q_i to q_{i+1}, matching
    angular_dynamics_residual_fix_bias (q_{i+1} = q_i * exp((w_i - b) dt))."""
    rel = attitude[:-1].inv() * attitude[1:]
    w = rel.as_rotvec() / dt
    w = np.vstack([w, w[-1]])
    return w + TRUE_GYRO_BIAS_RAD_S + rng.normal(scale=GYRO_WN_STD_DEV, size=w.shape)


def noise_kwargs(noise: str) -> tuple[dict, float | None]:
    """simulate_bearing_measurements kwargs for a noise mode, plus the constant
    sigma to write when the mode carries no per-measurement weights."""
    if noise == "legacy":
        sigma = np.arctan(LEGACY_ERROR_KM / ISS_RANGE_KM)
        return dict(angular_noise_std_rad=sigma), sigma
    if noise == "empirical":
        pool = load_empirical_angular_noise_pool_rad(MODEL_NAME, ISS_RANGE_KM)
        # robust (MAD-style) sigma: the pool is heavy-tailed, its RMS is not a 1-sigma
        return dict(angular_noise_std_rad=0.0, empirical_noise_pool_rad=pool), 1.4826 * np.median(pool)
    if noise == "tier":
        pool = load_empirical_noise_and_confidence(MODEL_NAME, ISS_RANGE_KM)
        return dict(angular_noise_std_rad=0.0, confidence_pool=pool,
                    tier_sigma_rad=confidence_tier_sigma_rad(*pool)), None
    if noise == "regression":
        return dict(angular_noise_std_rad=0.0, regression_pool=load_empirical_full_pool(MODEL_NAME, ISS_RANGE_KM),
                    regression_coefs=load_fitted_error_regression(MODEL_NAME), range_km=ISS_RANGE_KM), None
    if noise == "svr":
        return dict(angular_noise_std_rad=0.0, svr_pool=load_empirical_full_pool(MODEL_NAME, ISS_RANGE_KM),
                    svr_bundle=load_svr_error_model(MODEL_NAME), range_km=ISS_RANGE_KM), None
    raise ValueError(f"unknown noise mode {noise!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic flight-format OD dataset")
    parser.add_argument("out_folder", type=Path)
    parser.add_argument("--noise", default="regression",
                        choices=["legacy", "empirical", "tier", "regression", "svr"])
    parser.add_argument("--frames", type=int, default=30, help="Number of image frames")
    parser.add_argument("--frame-dt", type=float, default=30.0, help="Seconds between frames")
    parser.add_argument("--gyro-dt", type=float, default=10.0, help="Seconds between gyro samples (= OD states)")
    parser.add_argument("--landmarks-per-frame", type=int, default=137)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stride = int(round(args.frame_dt / args.gyro_dt))
    if abs(stride * args.gyro_dt - args.frame_dt) > 1e-9:
        raise ValueError("--frame-dt must be a multiple of --gyro-dt")
    n_states = (args.frames - 1) * stride + 1
    rng = np.random.default_rng(args.seed)

    states = propagate_truth(n_states, args.gyro_dt)
    attitude = nadir_attitude(states)
    gyro = simulate_gyro(attitude, args.gyro_dt, rng)
    state_ms = EPOCH_UNIX_MS + np.round(np.arange(n_states) * args.gyro_dt * 1000).astype(np.int64)

    frame_idx = np.arange(0, n_states, stride)
    kwargs, const_sigma = noise_kwargs(args.noise)
    bearing_eci, landmark_m, meas_frame, sigma = simulate_bearing_measurements(
        states[frame_idx] * 1000.0, rng=rng, landmarks_per_frame=args.landmarks_per_frame, **kwargs
    )
    if const_sigma is not None:
        sigma = np.full(len(sigma), const_sigma)
    state_of_meas = frame_idx[meas_frame]
    bearing_body = attitude[state_of_meas].inv().apply(bearing_eci)

    rows = np.zeros((len(bearing_body), 9))
    rows[:, 0] = state_ms[state_of_meas]
    rows[:, 1:4] = bearing_body
    rows[:, 4:7] = landmark_m / 1000.0
    rows[:, 7] = meas_frame
    rows[:, 8] = sigma

    out = args.out_folder
    out.mkdir(parents=True, exist_ok=True)
    write_landmark_csv(out / "landmark_measurements.csv", rows)
    write_imu_csv(out / "imu_data.csv", state_ms, gyro)

    quat = attitude.as_quat()  # [x, y, z, w], same convention as od/quaternion.py
    with open(out / "truth.csv", "w") as f:
        f.write("timestamp_ms,pos_x_km,pos_y_km,pos_z_km,vel_x_kms,vel_y_kms,vel_z_kms,"
                "quat_x,quat_y,quat_z,quat_w\n")
        for t, s, q in zip(state_ms, states, quat):
            f.write(f"{int(t)}," + ",".join(f"{v:.12g}" for v in np.concatenate([s, q])) + "\n")

    meta = {
        "noise": args.noise,
        "model": MODEL_NAME,
        "frames": args.frames,
        "frame_dt_s": args.frame_dt,
        "gyro_dt_s": args.gyro_dt,
        "landmarks_per_frame": args.landmarks_per_frame,
        "seed": args.seed,
        "true_gyro_bias_rad_s": TRUE_GYRO_BIAS_RAD_S.tolist(),
        "median_sigma_rad": float(np.median(sigma)),
    }
    with open(out / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(rows)} landmark rows, {n_states} gyro rows → {out}")


if __name__ == "__main__":
    main()
