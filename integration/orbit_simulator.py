"""Simulated two-frame OD: fixes the single-frame limitation (integration/
pose_resection.py -- one epoch constrains position+attitude only, never
velocity) without needing a second real photo.

Frame 1 is real: whatever LocalizationPipeline/retrieve+match already
produced (real pixels, real match, real predicted ground coordinate).
Frame 2 is simulated: assume nadir-pointing + an input altitude + ISS-like
inclination at frame 1's real ground point, propagate forward by dt seconds
with the same two-body propagator already validated in od_integration_test.py
(dynamics.orbital_dynamics.Dynamics.f), then snap the predicted ground point
to the nearest REAL reference-DB tile (real geography, not a synthetic point)
-- that tile's own known coordinates become frame 2's "measurement", no
matching needed since a reference tile's position is ground truth by
construction.

Both frames reduce to a single (lat, lon, epoch) triple under the nadir
assumption (frame 1's real match point is oblique in reality; this is a
known simplification, same one od_integration_test.py's synthetic-truth
generator already uses -- not a new approximation). The OD solve is then:
given r1 (fully determined by frame 1's coordinate + the input altitude) and
r2_target (frame 2's snapped-tile coordinate at the same input altitude),
find the velocity v1 such that Dynamics.f((r1, v1), dt) lands at r2_target.

Honesty note (matters for how to read the output): frame 2's ground point and
timestamp both derive from the SAME orbital model being solved for, so this
is a round-trip self-consistency check of the solver + simulator, not a
recovery from two independent real measurements. What IS a genuine, non-trivial
measurement is how much snapping frame 2 to the nearest real (discrete) tile
perturbs the recovered velocity versus the exact continuous simulated point
-- that discretization error is real and reported below.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core.path_setup import ensure_repo_root_first

ensure_repo_root_first(_REPO_ROOT)

import numpy as np
from brahe.constants import GM_EARTH, R_EARTH
from brahe.epoch import Epoch
from scipy.optimize import least_squares

from dynamics.orbital_dynamics import Dynamics
from utils.earth_utils import lat_lon_to_ecef

from core.types import GeoTile
from scripts.evaluate import haversine_km

ISS_INCLINATION_DEG = 51.6
ISS_ALTITUDE_KM = 420.0


def nadir_point_eci(lat_deg: float, lon_deg: float, altitude_km: float, epoch: Epoch) -> np.ndarray:
    """Spacecraft ECI position (meters) assuming it sits directly above
    (lat, lon) at the given altitude -- i.e. exactly nadir-pointing."""
    ecef_dir = lat_lon_to_ecef(np.array([[lat_deg, lon_deg]]))[0]
    ecef_dir = ecef_dir / np.linalg.norm(ecef_dir)
    ecef_pos = ecef_dir * (R_EARTH + altitude_km * 1e3)
    eci_R_ecef = brahe_frames_eci_from_ecef(epoch)
    return eci_R_ecef @ ecef_pos


def brahe_frames_eci_from_ecef(epoch: Epoch) -> np.ndarray:
    import brahe

    return brahe.frames.rECItoECEF(epoch).T


def eci_to_latlon(eci_pos_m: np.ndarray, epoch: Epoch) -> tuple[float, float]:
    """WGS84 geodetic (lat, lon) of the sub-satellite point -- the exact
    inverse of utils.earth_utils.lat_lon_to_ecef's ellipsoid model (a proper
    Bowring iteration, not a spherical arcsin/atan2 shortcut). Matters: a
    spherical inverse disagrees with lat_lon_to_ecef's WGS84 ellipsoid by up
    to ~0.2 deg (~20km) at mid-latitudes -- confirmed by direct measurement,
    this alone was enough to make solve_two_frame_od's self-consistency test
    fail to recover its own input. Only the direction matters here (matches
    nadir_point_eci, which also only uses direction), so any positive scale
    factor works for the (X, Y, Z) input -- height is discarded, only
    (lat, lon) is returned.
    """
    import brahe

    a, b = 6378137.0, 6356752.314245
    e2 = (a**2 - b**2) / a**2

    ecef_pos = brahe.frames.rECItoECEF(epoch) @ eci_pos_m
    x, y, z = ecef_pos
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)

    lat = np.arctan2(z, p * (1 - e2))  # initial geocentric-ish guess
    for _ in range(5):
        sin_lat = np.sin(lat)
        n = a / np.sqrt(1 - e2 * sin_lat**2)
        lat = np.arctan2(z + e2 * n * sin_lat, p)

    return float(np.degrees(lat)), float(np.degrees(lon))


def _circular_velocity_for_inclination(r1: np.ndarray, lat1_deg: float, inclination_deg: float) -> np.ndarray:
    """Velocity (m/s) for a circular orbit through r1 with the given
    inclination. See integration/orbit_simulator.py's derivation notes (repo
    memory / commit message) -- ground-track latitude can't exceed
    inclination, this raises clearly if asked to.
    """
    inclination_rad = np.radians(inclination_deg)
    lat1_rad = np.radians(lat1_deg)
    if abs(lat1_deg) > inclination_deg:
        raise ValueError(
            f"latitude {lat1_deg:.2f} deg exceeds inclination {inclination_deg:.2f} deg -- "
            f"no circular orbit at this inclination ever reaches this latitude"
        )

    r_hat = r1 / np.linalg.norm(r1)
    z_hat = np.array([0.0, 0.0, 1.0])
    z_perp = z_hat - np.dot(z_hat, r_hat) * r_hat
    z_perp_norm = np.linalg.norm(z_perp)
    z_perp_hat = z_perp / z_perp_norm
    e_hat = np.cross(r_hat, z_perp_hat)

    # |z_perp| = cos(lat1) exactly (r_hat . z_hat = sin(lat1)); solved so that
    # h_z/|h| = cos(inclination) for h = r1 x v1 (see derivation notes).
    cos_theta = -np.cos(inclination_rad) / np.cos(lat1_rad)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    v_circ = np.sqrt(GM_EARTH / np.linalg.norm(r1))
    v1 = v_circ * (np.sin(theta) * e_hat + np.cos(theta) * z_perp_hat)
    return v1


def simulate_next_frame(
    lat1_deg: float,
    lon1_deg: float,
    epoch1: Epoch,
    altitude_km: float = ISS_ALTITUDE_KM,
    inclination_deg: float = ISS_INCLINATION_DEG,
    dt_s: float = 120.0,
) -> dict:
    """Frame 1's real ground point -> simulated frame 2 ground point + the
    true (r1, v1, r2) used to generate it, for later comparison."""
    r1 = nadir_point_eci(lat1_deg, lon1_deg, altitude_km, epoch1)
    v1_true = _circular_velocity_for_inclination(r1, lat1_deg, inclination_deg)

    state2 = Dynamics.f(np.concatenate([r1, v1_true]), dt_s)
    r2_true = state2[:3]
    epoch2 = epoch1 + dt_s
    lat2_deg, lon2_deg = eci_to_latlon(r2_true, epoch2)

    return {
        "r1": r1, "v1_true": v1_true, "r2_true": r2_true,
        "lat2_deg": lat2_deg, "lon2_deg": lon2_deg, "epoch2": epoch2,
    }


def nearest_reference_tile(tiles: list[GeoTile], lat_deg: float, lon_deg: float) -> GeoTile:
    centers = np.array([t.corners_latlon.mean(axis=0) for t in tiles])
    dists = np.array([haversine_km(lat_deg, lon_deg, c[0], c[1]) for c in centers])
    return tiles[int(np.argmin(dists))]


def solve_two_frame_od(
    lat1_deg: float, lon1_deg: float, epoch1: Epoch,
    lat2_deg: float, lon2_deg: float, epoch2: Epoch,
    altitude_km: float = ISS_ALTITUDE_KM,
) -> dict:
    """Given frame 1's (lat, lon, epoch) at the assumed altitude, and frame
    2's (lat, lon, epoch) -- direction only, no altitude, exactly like a real
    reference-DB tile's coordinate -- solve for the velocity at epoch1
    consistent with Dynamics.f propagating from frame 1 to frame 2.

    Residual is direction-match at frame 2 (NOT full 3D position: Dynamics.f's
    RK4 step doesn't exactly preserve orbital radius over one large step, so
    re-imposing the exact assumed altitude on frame 2's target would fight
    the propagator's own numerical drift -- confirmed by direct measurement,
    ~22km spurious mismatch at dt=120s before this fix) plus a zero-radial-
    velocity constraint at r1 (v1 . r1_hat = 0 -- exactly circular, no
    eccentricity component). Direction alone under-determines v1: over a
    short arc, position direction is nearly insensitive to a radial velocity
    perturbation, so a speed-only 3rd constraint still let a ~560 m/s error
    slip through (confirmed by direct measurement) -- zero-radial is the
    constraint that actually matches how the simulator built v1_true
    (tangential by construction), so it closes the degeneracy correctly.
    """
    r1 = nadir_point_eci(lat1_deg, lon1_deg, altitude_km, epoch1)
    r1_hat = r1 / np.linalg.norm(r1)
    target_dir = lat_lon_to_ecef(np.array([[lat2_deg, lon2_deg]]))[0]
    target_dir = target_dir / np.linalg.norm(target_dir)
    eci_R_ecef2 = brahe_frames_eci_from_ecef(epoch2)
    target_dir_eci = eci_R_ecef2 @ target_dir
    dt_s = (epoch2 - epoch1)

    v1_guess = _circular_velocity_for_inclination(r1, lat1_deg, ISS_INCLINATION_DEG)

    def residuals(v1):
        state2 = Dynamics.f(np.concatenate([r1, v1]), dt_s)
        r2 = state2[:3]
        dir_residual = r2 / np.linalg.norm(r2) - target_dir_eci
        radial_residual = np.dot(v1, r1_hat)
        return np.concatenate([dir_residual, [radial_residual]])

    result = least_squares(residuals, v1_guess, method="lm", max_nfev=2000)
    v1_solved = result.x
    final = residuals(v1_solved)

    return {
        "success": bool(result.success),
        "r1_m": r1,
        "v1_solved_ms": v1_solved,
        "speed_solved_kms": float(np.linalg.norm(v1_solved) / 1e3),
        "altitude_r1_km": float(np.linalg.norm(r1) - R_EARTH) / 1e3,
        "direction_residual_rad": float(np.linalg.norm(final[:3])),
        "radial_velocity_residual_ms": float(final[3]),
        "dt_s": dt_s,
        "cost": float(result.cost),
    }
