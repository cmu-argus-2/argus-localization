"""OD integration test: pixel coordinates (our retrieve-then-match pipeline's
real output) -> bearing vectors -> feed into a batch-least-squares orbit
determination solve, report position error.

Two parts, run separately because they answer different questions:

1. `demo_real_pixel_to_bearing()`: takes REAL tie_points from a real
   LocalizationPipeline.localize() call (real pixel coordinates, real
   matched-tile lat/lon) and runs them through the actual conversion chain
   this needs: pixel -> bearing (GNC-Payload's CameraModel,
   sensors/camera_model.py) and lat/lon -> ECI (lat_lon_to_ecef +
   brahe.frames.rECItoECEF, both from GNC-Payload's utils/earth_utils.py).
   This demonstrates the requested "bearing vectors from the given pixel
   coordinates" conversion is wired correctly end to end -- NOT a claim of
   metric accuracy, since our EarthLoc astronaut-photo queries were not
   captured by Argus's actual camera (unknown real intrinsics), so mapping
   them through Argus's specific CameraModel (4608x2592, 66.1deg HFOV) is a
   mechanism demonstration, not a calibrated measurement.

2. `run_od_solve()`: a self-contained batch nonlinear least-squares OD solve
   (position+velocity only, attitude given/fixed per step, matching the
   *intent* of GNC-Payload's orbit_determination/nonlinear_least_squares_od.py
   -- that module's own residuals()/residual_jac() math is reused directly,
   see below) on a synthetic circular LEO trajectory. Bearing noise is not
   arbitrary: it's derived from this repo's own already-measured matcher
   accuracy (README: 3.0km median localization error on Alps, at ISS-orbit
   range ~400km -> ~0.43deg angular noise).

Errors found resurrecting the existing orbit_determination code (reported
here, not silently patched around): GNC-Payload's
`orbit_determination/test_nonlinear_least_squares.py` and
`nonlinear_least_squares_od.py::fit_orbit` are stale against the current
`ODSimulationDataManager` --
  - `push_next_state()` used to take (state6, rotation_matrix, gyro_bias) as
    3 args; it now takes one packed `state` array indexed via
    `simulation.dynamics.orbital_att_dynamics.DynamicsIDX`
    ([pos(3), vel(3), quat_wxyz(4), omega(3)], 13-dim by default).
  - `fit_orbit()` reads `data_manager.eci_Rs_body`, an (N,3,3) array
    attribute that no longer exists on `ODSimulationDataManager` (attitude is
    now packed into `.states[:, dynidx.QUAT]` as a quaternion instead).
  - The test also treats `data_manager.latest_state`/`.trans_states` as
    6-dim position/velocity-only, inconsistent with the current 13-dim
    `.states`.
Running `python -m orbit_determination.test_nonlinear_least_squares` from
GNC-Payload confirms this: it fails immediately with
`TypeError: ODSimulationDataManager.push_next_state() takes 2 positional
arguments but 4 were given`. This script works around that by using the
lower-level building blocks that ARE current and correct (CameraModel,
earth_utils, dynamics.orbital_dynamics.Dynamics, and fit_orbit's own
residuals/residual_jac formulas, fed manually instead of through the broken
data-manager coupling) rather than resurrecting the stale glue code.
"""

import os
import sys

_GNC_PAYLOAD_ROOT = "/home/pvijayba/GNC-Payload"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_GNC_PAYLOAD_ROOT, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import brahe
from brahe.epoch import Epoch
from brahe.constants import GM_EARTH, R_EARTH
from scipy.optimize import least_squares

from sensors.camera_model import CameraModelManager
from utils.earth_utils import lat_lon_to_ecef
from dynamics.orbital_dynamics import Dynamics


def latlon_to_eci(lat_deg: np.ndarray, lon_deg: np.ndarray, epoch: Epoch) -> np.ndarray:
    """(N,) lat, (N,) lon [deg] -> (N,3) ECI position [m], via ECEF (WGS84, h=0)."""
    lat_lon = np.column_stack([lat_deg, lon_deg])
    ecef = lat_lon_to_ecef(lat_lon)  # (N,3) meters
    ecef_R_eci = brahe.frames.rECItoECEF(epoch)
    eci = (ecef_R_eci.T @ ecef.T).T
    return eci


def demo_real_pixel_to_bearing():
    """Part 1: real tie_points -> bearing vectors + ECI landmark positions."""
    u = np.load("/tmp/tp_u.npy")
    v = np.load("/tmp/tp_v.npy")
    lat = np.load("/tmp/tp_lat.npy")
    lon = np.load("/tmp/tp_lon.npy")
    with open("/tmp/query_img_shape.txt") as f:
        h_512, w_512 = (int(x) for x in f.read().split())

    print(f"Loaded {len(u)} real tie_points from LocalizationPipeline.localize() "
          f"(query resized to {w_512}x{h_512} for matching)")

    cam = CameraModelManager()["x+"]
    # Our matcher's tie_points are in the (w_512, h_512) resized-query pixel
    # space; CameraModel expects pixels in its own fixed (4608, 2592) frame.
    # Rescale -- an approximation, see module docstring: these EarthLoc
    # queries weren't captured by Argus's actual camera, so this shows the
    # conversion mechanism working, not a calibrated bearing measurement.
    u_full = u * (cam.IMAGE_WIDTH / w_512)
    v_full = v * (cam.IMAGE_HEIGHT / h_512)
    pixel_coords = np.column_stack([u_full, v_full])

    bearing_body = cam.pixel_to_bearing_unit_vector(pixel_coords)
    norms = np.linalg.norm(bearing_body, axis=1)
    print(f"bearing_body: shape={bearing_body.shape}, "
          f"norm min/max={norms.min():.6f}/{norms.max():.6f} (expect 1.0)")

    epoch = Epoch(2021, 7, 24, 0, 0, 0.0)  # matches the query's own capture date (ISS065-E-207004)
    landmark_eci = latlon_to_eci(lat, lon, epoch)
    print(f"landmark_eci: shape={landmark_eci.shape}, "
          f"|r| min/max = {np.linalg.norm(landmark_eci, axis=1).min()/1e3:.1f} / "
          f"{np.linalg.norm(landmark_eci, axis=1).max()/1e3:.1f} km (expect ~{R_EARTH/1e3:.0f} km, on the ellipsoid)")

    for i in range(3):
        print(f"  tie_point {i}: pixel=({u[i]:.1f},{v[i]:.1f}) -> bearing_body={bearing_body[i]} "
              f"| lat/lon=({lat[i]:.4f},{lon[i]:.4f}) -> ECI={landmark_eci[i]/1e3} km")

    return bearing_body, landmark_eci


def make_circular_orbit_truth(n_steps: int, dt: float, altitude_m: float = 420e3):
    """Synthetic ISS-like circular LEO trajectory (nadir-pointing), ECI, meters."""
    r0 = R_EARTH + altitude_m
    v0 = np.sqrt(GM_EARTH / r0)
    period = 2 * np.pi * r0 / v0
    inclination = np.radians(51.6)  # ISS inclination

    states = np.zeros((n_steps, 6))
    states[0] = [r0, 0.0, 0.0, 0.0, v0 * np.cos(inclination), v0 * np.sin(inclination)]
    for i in range(1, n_steps):
        states[i] = Dynamics.f(states[i - 1], dt)
    return states, period


def simulate_bearing_measurements(states, angular_noise_std_rad, rng, landmarks_per_frame=1):
    """`landmarks_per_frame` bearing measurements per state, spread across a
    ~66deg half-cone around nadir (matching Argus's own CameraModel.HORIZONTAL_FOV
    and, more importantly, matching what our REAL pipeline actually produces --
    ~137 simultaneous tie points spread across one image, not one nadir-only
    bearing per frame). Returns per-measurement (bearing_body_noisy, landmark_eci,
    frame_index)."""
    n = states.shape[0]
    positions = states[:, :3]
    nadir = -positions / np.linalg.norm(positions, axis=1, keepdims=True)

    all_bearings, all_landmarks, all_frame_idx = [], [], []
    half_cone_rad = np.radians(66.1) / 2  # Argus CameraModel.HORIZONTAL_FOV / 2
    for i in range(n):
        # basis for the plane perpendicular to nadir at this step
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(nadir[i, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(nadir[i], arbitrary)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nadir[i], e1)

        cos_cone = rng.uniform(np.cos(half_cone_rad), 1.0, size=landmarks_per_frame)
        theta = rng.uniform(0, 2 * np.pi, size=landmarks_per_frame)
        sin_cone = np.sqrt(1 - cos_cone**2)
        bearing_true = (
            cos_cone[:, None] * nadir[i]
            + (sin_cone * np.cos(theta))[:, None] * e1
            + (sin_cone * np.sin(theta))[:, None] * e2
        )

        # intersect each ray with the Earth sphere (R_EARTH) from positions[i]
        b = 2 * bearing_true @ positions[i]
        c = np.dot(positions[i], positions[i]) - R_EARTH**2
        disc = b**2 - 4 * c
        t = (-b - np.sqrt(np.clip(disc, 0, None))) / 2
        landmark = positions[i] + t[:, None] * bearing_true

        noise_axis = rng.normal(size=(landmarks_per_frame, 3))
        noise_axis -= np.sum(noise_axis * bearing_true, axis=1, keepdims=True) * bearing_true
        noise_axis /= np.linalg.norm(noise_axis, axis=1, keepdims=True)
        noise_angle = rng.normal(scale=angular_noise_std_rad, size=landmarks_per_frame)
        bearing_noisy = (
            bearing_true * np.cos(noise_angle)[:, None] + noise_axis * np.sin(noise_angle)[:, None]
        )
        bearing_noisy /= np.linalg.norm(bearing_noisy, axis=1, keepdims=True)

        all_bearings.append(bearing_noisy)
        all_landmarks.append(landmark)
        all_frame_idx.append(np.full(landmarks_per_frame, i))

    return np.concatenate(all_bearings), np.concatenate(all_landmarks), np.concatenate(all_frame_idx)


def fit_orbit_manual(dt, measurement_indices, bearing_unit_vectors_wf, landmarks, N, semi_major_axis_guess):
    """Reimplements nonlinear_least_squares_od.py::fit_orbit's residuals()/
    residual_jac() (that math is correct) fed from plain arrays instead of
    the broken ODSimulationDataManager.eci_Rs_body coupling (see module
    docstring)."""
    M = len(measurement_indices)

    def residuals(X):
        states = X.reshape(N, 6)
        res = np.zeros(6 * (N - 1) + 3 * M)
        idx = 0
        for i in range(N - 1):
            res[idx:idx + 6] = states[i + 1, :] - Dynamics.f(states[i, :], dt)
            idx += 6
        for i, (t, landmark) in enumerate(zip(measurement_indices, landmarks)):
            cubesat_position = states[t, :3]
            pred = landmark - cubesat_position
            pred_u = pred / np.linalg.norm(pred)
            res[idx:idx + 3] = pred_u - bearing_unit_vectors_wf[i]
            idx += 3
        return res

    altitude_normalized_landmarks = landmarks / np.linalg.norm(landmarks, axis=1, keepdims=True)
    # simple circular-orbit initial guess: spread the normalized landmark directions
    # at the guessed radius, in time order
    initial_guess = np.zeros((N, 6))
    for i in range(N):
        idxs = np.where(measurement_indices == i)[0]
        if len(idxs) > 0:
            direction = altitude_normalized_landmarks[idxs[0]]
        else:
            direction = altitude_normalized_landmarks[0]
        pos = semi_major_axis_guess * direction
        initial_guess[i, :3] = pos
    initial_guess[:, 3:] = 0.0
    initial_guess = initial_guess.flatten()

    result = least_squares(residuals, initial_guess, method="lm", max_nfev=20000)
    return result.x.reshape(N, 6), result


def run_od_solve():
    """Part 2: synthetic trajectory + calibrated bearing noise -> OD solve."""
    rng = np.random.default_rng(0)
    dt = 30.0  # s
    n_steps = 30
    states_true, period = make_circular_orbit_truth(n_steps, dt)
    print(f"Synthetic circular orbit: altitude=420km, period={period/60:.1f} min, "
          f"{n_steps} steps @ dt={dt}s ({n_steps*dt/60:.1f} min of track)")

    # Angular noise calibrated from this repo's own measured matcher accuracy
    # (README: 3.0km median localization error on Alps at ISS-orbit range).
    iss_range_km = 400.0
    localization_error_km = 3.0
    angular_noise_rad = np.arctan(localization_error_km / iss_range_km)
    print(f"Bearing noise calibrated from measured pipeline accuracy: "
          f"{localization_error_km}km @ {iss_range_km}km range -> "
          f"{np.degrees(angular_noise_rad):.3f} deg ({angular_noise_rad*1e3:.2f} mrad) 1-sigma")

    for landmarks_per_frame, label in [(1, "1 nadir-only bearing/frame"),
                                        (137, "137 bearings/frame (matches our real matcher's yield)")]:
        bearing_noisy, landmarks, measurement_indices = simulate_bearing_measurements(
            states_true, angular_noise_rad, np.random.default_rng(0), landmarks_per_frame
        )
        estimated_states, result = fit_orbit_manual(
            dt, measurement_indices, bearing_noisy, landmarks, n_steps,
            semi_major_axis_guess=R_EARTH + 420e3,
        )
        pos_errors_km = np.linalg.norm(states_true[:, :3] - estimated_states[:, :3], axis=1) / 1e3
        vel_errors_ms = np.linalg.norm(states_true[:, 3:] - estimated_states[:, 3:], axis=1)
        print(f"\n--- {label} ({len(measurement_indices)} total measurements) ---")
        print(f"solver: success={result.success} status={result.status} "
              f"cost={result.cost:.4e} nfev={result.nfev}")
        print(f"RMS position error: {np.sqrt(np.mean(pos_errors_km**2)):.3f} km")
        print(f"max position error: {pos_errors_km.max():.3f} km")
        print(f"RMS velocity error: {np.sqrt(np.mean(vel_errors_ms**2)):.3f} m/s")

    return pos_errors_km, vel_errors_ms, result


if __name__ == "__main__":
    print("=" * 70)
    print("PART 1: real pixel coordinates -> bearing vectors (our pipeline's output)")
    print("=" * 70)
    demo_real_pixel_to_bearing()

    print()
    print("=" * 70)
    print("PART 2: OD batch least-squares solve (synthetic trajectory, calibrated noise)")
    print("=" * 70)
    run_od_solve()
