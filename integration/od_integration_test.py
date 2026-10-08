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

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core.path_setup import ensure_repo_root_first

ensure_repo_root_first(_REPO_ROOT)

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


def load_empirical_angular_noise_pool_rad(
    model_name: str, range_km: float,
    dataset_path: str = "output/error_dataset/combined.jsonl",
) -> np.ndarray:
    """Real per-measurement angular noise magnitudes, bootstrapped from
    scripts/collect_error_dataset.py's actual full-pipeline error_km values
    for one model's "fix" rows (900+/model) -- the alternative to assuming a
    single Gaussian sigma. Deliberately NOT the fitted log(error_km) ~
    num_inliers + retrieval_similarity regression from scripts/fit_error_model.py:
    that regression's R^2 was 0.008-0.032 (confirmed, not a modeling-effort
    gap -- confidence signals genuinely don't predict error magnitude below
    ~200 inliers), so forcing a clean parametric curve onto it would be less
    honest than resampling the real, heavy-tailed, non-Gaussian distribution
    directly. See repo memory current_model_numbers / the error-modeling
    session for the full finding.
    """
    angular_noise_rad, _inliers = load_empirical_noise_and_confidence(model_name, range_km, dataset_path)
    return angular_noise_rad


def load_empirical_noise_and_confidence(
    model_name: str, range_km: float,
    dataset_path: str = "output/error_dataset/combined.jsonl",
) -> tuple[np.ndarray, np.ndarray]:
    """Real (angular_noise_rad, num_inliers) PAIRS for one model's fix rows --
    kept joint (not two independent pools) so bootstrapping preserves
    whatever real relationship exists between confidence and error, including
    the one real structure found (outlier risk only drops above ~200
    inliers; flat/noisy below that -- see fit_error_model.py's binned
    diagnostic), for weighted least squares (confidence_tier_sigma_rad below).
    """
    angular_noise_rad, inliers, _sim = load_empirical_full_pool(model_name, range_km, dataset_path)
    return angular_noise_rad, inliers


def load_empirical_full_pool(
    model_name: str, range_km: float,
    dataset_path: str = "output/error_dataset/combined.jsonl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real (angular_noise_rad, num_inliers, retrieval_similarity) TRIPLES for
    one model's fix rows, kept joint for bootstrapping. Adds
    retrieval_similarity on top of load_empirical_noise_and_confidence so
    regression_sigma_rad can evaluate scripts/fit_error_model.py's actual
    fitted model (which uses both covariates), not just the num_inliers-only
    2-tier split.
    """
    import json

    errors_km, inliers, sims = [], [], []
    for line in open(dataset_path):
        row = json.loads(line)
        if row["model"] == model_name and row["status"] == "fix" and row["error_km"]:
            errors_km.append(row["error_km"])
            inliers.append(row["num_inliers"])
            sims.append(row["retrieval_similarity"])
    errors_km = np.array(errors_km)
    return np.arctan(errors_km / range_km), np.array(inliers), np.array(sims)


def load_fitted_error_regression(
    model_name: str, error_model_path: str = "output/error_dataset/error_model.json",
) -> dict:
    """The actual scripts/fit_error_model.py output for one model: log(error_km)
    = b0 + b1*num_inliers + b2*retrieval_similarity. R^2 was 0.008-0.032 across
    all 3 models (confirmed weak, not a modeling-effort gap -- see that
    script's docstring and the error-modeling session), so regression_sigma_rad
    uses it anyway on request, but it predicts error far less reliably than
    the empirical 2-tier split in confidence_tier_sigma_rad.
    """
    import json

    with open(error_model_path) as f:
        return json.load(f)[model_name]


def load_svr_error_model(model_name: str, model_dir: str = "output/error_dataset") -> dict:
    """Loads scripts/fit_error_model_svr.py's fitted SVR + scaler for one
    model. 5-fold CV R^2 was -0.005 to 0.021 across all 3 models (confirmed:
    at or below the "just predict the mean" baseline, worse than even the
    weak linear fit's 0.008-0.032) -- included for the requested head-to-head
    comparison, not because it's expected to help.
    """
    import pickle

    with open(f"{model_dir}/svr_model_{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def svr_sigma_rad(inliers: np.ndarray, similarity: np.ndarray, svr_bundle: dict, range_km: float) -> np.ndarray:
    """Per-measurement angular sigma predicted by the fitted SVR (log_sigma_km
    = svr.predict on scaled [num_inliers, retrieval_similarity]), converted to
    radians at range_km -- the SVR analog of regression_sigma_rad."""
    X = np.column_stack([inliers, similarity])
    X_scaled = svr_bundle["scaler"].transform(X)
    log_sigma_km = svr_bundle["svr"].predict(X_scaled)
    sigma_km = np.exp(log_sigma_km)
    return np.arctan(sigma_km / range_km)


def regression_sigma_rad(inliers: np.ndarray, similarity: np.ndarray, coefs: dict, range_km: float) -> np.ndarray:
    """Per-measurement angular sigma predicted directly by the fitted
    log(error_km) ~ num_inliers + retrieval_similarity regression (coefs from
    load_fitted_error_regression), converted to radians at range_km. This is
    the continuous alternative to confidence_tier_sigma_rad's 2-tier split --
    weaker-grounded (R^2 0.008-0.032) but uses the actual least-squares fit
    the user asked for, not a post-hoc binned heuristic.
    """
    log_sigma_km = coefs["b0_intercept"] + coefs["b1_num_inliers"] * inliers + coefs["b2_retrieval_similarity"] * similarity
    sigma_km = np.exp(log_sigma_km)
    return np.arctan(sigma_km / range_km)


def confidence_tier_sigma_rad(
    angular_noise_rad: np.ndarray, inliers: np.ndarray, high_confidence_inlier_threshold: float = 200.0,
) -> tuple[float, float]:
    """Two-tier angular noise sigma (low-confidence, high-confidence), split
    at high_confidence_inlier_threshold -- the binned diagnostic found outlier
    probability is flat/noisy for num_inliers below ~200 and drops sharply
    above it, not a smooth function, so a 2-tier split is the honest
    granularity here (not a continuous confidence-weighted curve, which the
    R^2~0.01-0.03 regression showed doesn't exist). Returns (sigma_low_rad,
    sigma_high_rad), each the real population std of that tier's angular
    noise -- a priori calibrated values, not read off the live sample being
    weighted.
    """
    high_mask = inliers >= high_confidence_inlier_threshold
    sigma_low = float(np.std(angular_noise_rad[~high_mask]))
    sigma_high = float(np.std(angular_noise_rad[high_mask])) if high_mask.any() else sigma_low
    return sigma_low, sigma_high


def simulate_bearing_measurements(
    states, angular_noise_std_rad, rng, landmarks_per_frame=1,
    empirical_noise_pool_rad=None, confidence_pool=None, tier_sigma_rad=None,
    regression_pool=None, regression_coefs=None, range_km=None,
    svr_pool=None, svr_bundle=None,
):
    """`landmarks_per_frame` bearing measurements per state, spread across a
    ~66deg half-cone around nadir (matching Argus's own CameraModel.HORIZONTAL_FOV
    and, more importantly, matching what our REAL pipeline actually produces --
    ~137 simultaneous tie points spread across one image, not one nadir-only
    bearing per frame). Returns per-measurement (bearing_body_noisy, landmark_eci,
    frame_index, weight_sigma_rad).

    If `empirical_noise_pool_rad` is given, each measurement's noise MAGNITUDE
    is bootstrapped (sampled with replacement) from that real, measured pool
    instead of drawn from a single fixed-sigma Gaussian (angular_noise_std_rad
    is then ignored) -- see load_empirical_angular_noise_pool_rad. weight_sigma_rad
    is all-ones in this mode (equal weighting).

    If `confidence_pool` ((angular_noise_rad, num_inliers) arrays from
    load_empirical_noise_and_confidence) and `tier_sigma_rad` ((sigma_low,
    sigma_high) from confidence_tier_sigma_rad) are both given, noise is
    bootstrapped JOINTLY with num_inliers (same sampled index for both, so
    the real -- weak but real -- relationship is preserved) and
    weight_sigma_rad[i] is set to the pre-calibrated tier sigma matching that
    measurement's bootstrapped num_inliers, for weighted least squares
    (fit_orbit_manual divides each bearing residual by its weight_sigma_rad).
    This is NOT the per-sample noise magnitude itself -- using that would be
    circular (the optimizer wouldn't know the true error, only the tier a
    real num_inliers value would put it in).

    If `regression_pool` ((angular_noise_rad, num_inliers, retrieval_similarity)
    from load_empirical_full_pool), `regression_coefs` (from
    load_fitted_error_regression) and `range_km` are all given, weight_sigma_rad
    is instead the CONTINUOUS value regression_sigma_rad predicts from the
    bootstrapped num_inliers/similarity -- the actual least-squares fit, as
    opposed to confidence_tier_sigma_rad's 2-tier split.
    """
    n = states.shape[0]
    positions = states[:, :3]
    nadir = -positions / np.linalg.norm(positions, axis=1, keepdims=True)

    all_bearings, all_landmarks, all_frame_idx, all_weight_sigma = [], [], [], []
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
        if svr_pool is not None:
            pool_noise_rad, pool_inliers, pool_sim = svr_pool
            boot_idx = rng.integers(0, len(pool_noise_rad), size=landmarks_per_frame)
            noise_angle = pool_noise_rad[boot_idx] * rng.choice([-1.0, 1.0], size=landmarks_per_frame)
            weight_sigma = svr_sigma_rad(pool_inliers[boot_idx], pool_sim[boot_idx], svr_bundle, range_km)
        elif regression_pool is not None:
            pool_noise_rad, pool_inliers, pool_sim = regression_pool
            boot_idx = rng.integers(0, len(pool_noise_rad), size=landmarks_per_frame)
            noise_angle = pool_noise_rad[boot_idx] * rng.choice([-1.0, 1.0], size=landmarks_per_frame)
            weight_sigma = regression_sigma_rad(pool_inliers[boot_idx], pool_sim[boot_idx], regression_coefs, range_km)
        elif confidence_pool is not None:
            pool_noise_rad, pool_inliers = confidence_pool
            sigma_low, sigma_high = tier_sigma_rad
            boot_idx = rng.integers(0, len(pool_noise_rad), size=landmarks_per_frame)
            noise_angle = pool_noise_rad[boot_idx] * rng.choice([-1.0, 1.0], size=landmarks_per_frame)
            weight_sigma = np.where(pool_inliers[boot_idx] >= 200.0, sigma_high, sigma_low)
        elif empirical_noise_pool_rad is not None:
            noise_angle = rng.choice(empirical_noise_pool_rad, size=landmarks_per_frame, replace=True)
            noise_angle *= rng.choice([-1.0, 1.0], size=landmarks_per_frame)  # pool is |angle|, randomize sign
            weight_sigma = np.ones(landmarks_per_frame)
        else:
            noise_angle = rng.normal(scale=angular_noise_std_rad, size=landmarks_per_frame)
            weight_sigma = np.ones(landmarks_per_frame)
        bearing_noisy = (
            bearing_true * np.cos(noise_angle)[:, None] + noise_axis * np.sin(noise_angle)[:, None]
        )
        bearing_noisy /= np.linalg.norm(bearing_noisy, axis=1, keepdims=True)

        all_bearings.append(bearing_noisy)
        all_landmarks.append(landmark)
        all_frame_idx.append(np.full(landmarks_per_frame, i))
        all_weight_sigma.append(weight_sigma)

    return (
        np.concatenate(all_bearings), np.concatenate(all_landmarks),
        np.concatenate(all_frame_idx), np.concatenate(all_weight_sigma),
    )


def fit_orbit_manual(
    dt, measurement_indices, bearing_unit_vectors_wf, landmarks, N, semi_major_axis_guess,
    weight_sigma=None,
):
    """Reimplements nonlinear_least_squares_od.py::fit_orbit's residuals()/
    residual_jac() (that math is correct) fed from plain arrays instead of
    the broken ODSimulationDataManager.eci_Rs_body coupling (see module
    docstring).

    `weight_sigma`, if given, is a per-measurement angular sigma (radians)
    from confidence_tier_sigma_rad -- each bearing residual is divided by its
    own sigma before being squared and summed by least_squares, which is
    exactly weighted least squares (minimizing sum((r_i/sigma_i)^2) instead
    of sum(r_i^2)): a low-confidence measurement contributes less to the
    cost for the same raw angular error. The dynamics-consistency block is
    left unweighted (sigma=1) -- this is reweighting trust in the SENSOR
    model only, not in the propagated dynamics. None means equal-weighted
    (every sigma=1), matching the original unweighted behavior.
    """
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
            r = pred_u - bearing_unit_vectors_wf[i]
            if weight_sigma is not None:
                r = r / weight_sigma[i]
            res[idx:idx + 3] = r
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
    dt = 30.0  # s
    n_steps = 30
    states_true, period = make_circular_orbit_truth(n_steps, dt)
    print(f"Synthetic circular orbit: altitude=420km, period={period/60:.1f} min, "
          f"{n_steps} steps @ dt={dt}s ({n_steps*dt/60:.1f} min of track)")

    iss_range_km = 400.0
    # OLD calibration: single fixed Gaussian sigma from one stale number
    # (3.0km median on Alps, 50 queries, pre-error-modeling-session).
    legacy_localization_error_km = 3.0
    legacy_angular_noise_rad = np.arctan(legacy_localization_error_km / iss_range_km)
    print(f"[legacy] fixed-Gaussian noise calibrated from one old number: "
          f"{legacy_localization_error_km}km @ {iss_range_km}km range -> "
          f"{np.degrees(legacy_angular_noise_rad):.3f} deg 1-sigma")

    # NEW calibration: bootstrap real per-measurement angular noise from the
    # actual 900+-fix error dataset (scripts/collect_error_dataset.py),
    # faithful_lora_v2 (current best model) -- heavy-tailed and non-Gaussian,
    # not forced into a single-sigma assumption.
    model_name = "faithful_lora_v2"
    empirical_pool_rad = load_empirical_angular_noise_pool_rad(model_name, iss_range_km)
    print(f"[empirical] {model_name}: bootstrapping from {len(empirical_pool_rad)} real measured errors, "
          f"median={np.degrees(np.median(empirical_pool_rad)):.3f} deg, "
          f"p90={np.degrees(np.percentile(empirical_pool_rad, 90)):.3f} deg")

    # CONFIDENCE-WEIGHTED calibration: same real noise realizations, but the
    # solver downweights low-confidence measurements using the one real
    # structure found (outlier risk only drops above ~200 inliers).
    confidence_pool = load_empirical_noise_and_confidence(model_name, iss_range_km)
    tier_sigma = confidence_tier_sigma_rad(*confidence_pool)
    print(f"[confidence-weighted, 2-tier] {model_name}: tier sigma low={np.degrees(tier_sigma[0]):.3f} deg "
          f"(<200 inliers), high={np.degrees(tier_sigma[1]):.3f} deg (>=200 inliers)")

    # REGRESSION-weighted: the actual scripts/fit_error_model.py least-squares
    # fit (log(error_km) ~ num_inliers + retrieval_similarity), used directly
    # as the per-measurement sigma instead of the 2-tier heuristic above.
    regression_pool = load_empirical_full_pool(model_name, iss_range_km)
    regression_coefs = load_fitted_error_regression(model_name)
    print(f"[confidence-weighted, regression] {model_name}: using fit_error_model.py's fit "
          f"(R^2={regression_coefs['r_squared']:.3f}) directly as per-measurement sigma")

    svr_bundle = load_svr_error_model(model_name)
    svr_summary = json.load(open("output/error_dataset/error_model_svr.json"))[model_name]
    print(f"[confidence-weighted, SVR] {model_name}: using fit_error_model_svr.py's fit "
          f"(5-fold CV R^2={svr_summary['cv_r_squared']:.3f}) directly as per-measurement sigma")

    for noise_label, noise_kwargs, weighted in [
        ("legacy fixed-Gaussian", dict(angular_noise_std_rad=legacy_angular_noise_rad), False),
        ("empirical bootstrap, unweighted", dict(angular_noise_std_rad=0.0, empirical_noise_pool_rad=empirical_pool_rad), False),
        ("empirical bootstrap, 2-tier confidence-weighted",
         dict(angular_noise_std_rad=0.0, confidence_pool=confidence_pool, tier_sigma_rad=tier_sigma), True),
        ("empirical bootstrap, regression confidence-weighted",
         dict(angular_noise_std_rad=0.0, regression_pool=regression_pool, regression_coefs=regression_coefs,
              range_km=iss_range_km), True),
        ("empirical bootstrap, SVR confidence-weighted",
         dict(angular_noise_std_rad=0.0, svr_pool=regression_pool, svr_bundle=svr_bundle,
              range_km=iss_range_km), True),
    ]:
        for landmarks_per_frame, label in [(1, "1 nadir-only bearing/frame"),
                                            (137, "137 bearings/frame (matches our real matcher's yield)")]:
            bearing_noisy, landmarks, measurement_indices, weight_sigma = simulate_bearing_measurements(
                states_true, rng=np.random.default_rng(0), landmarks_per_frame=landmarks_per_frame, **noise_kwargs
            )
            estimated_states, result = fit_orbit_manual(
                dt, measurement_indices, bearing_noisy, landmarks, n_steps,
                semi_major_axis_guess=R_EARTH + 420e3,
                weight_sigma=weight_sigma if weighted else None,
            )
            pos_errors_km = np.linalg.norm(states_true[:, :3] - estimated_states[:, :3], axis=1) / 1e3
            vel_errors_ms = np.linalg.norm(states_true[:, 3:] - estimated_states[:, 3:], axis=1)
            print(f"\n--- [{noise_label}] {label} ({len(measurement_indices)} total measurements) ---")
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
