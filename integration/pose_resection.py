"""Single-frame position+attitude resection from bearing/landmark
correspondences -- the honest "OD output" for ONE real image.

Why this isn't full orbit determination: `od_integration_test.py::fit_orbit_manual`
(and FSW-Payload's real Ceres batch optimizer) solve for position+velocity
across MULTIPLE frames linked by orbital dynamics, with attitude treated as
KNOWN input (from a star tracker/IMU) at each frame -- see
measurement_residuals.hpp's LandmarkCostFunctor. A single real astronaut
photo gives neither: no second frame to constrain velocity/dynamics, and no
real attitude telemetry (these EarthLoc/GAPE queries aren't Argus captures).

So this solves the only thing a single frame's bearings actually constrain:
position AND attitude jointly, from many bearing-landmark correspondences
(classic photogrammetric space resection / PnP, just parameterized with
bearing vectors instead of pixel+intrinsics -- the same problem). Velocity is
not observable from one epoch and is not estimated here.

Reports RMS angular residual (degrees) as the fit's own error metric -- NOT
error against known spacecraft truth, which doesn't exist for these photos.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

R_EARTH_M = 6378137.0  # WGS84 semi-major axis


def solve_single_frame_pose(
    bearing_body: np.ndarray,  # (N,3) unit vectors, camera body frame
    landmark_eci_m: np.ndarray,  # (N,3) meters
    altitude_guess_m: float = 420e3,  # ISS-like default initial guess only
) -> dict:
    n = bearing_body.shape[0]
    if n < 6:
        return {"success": False, "error": f"need >=6 correspondences for a well-posed 6-DOF resection, got {n}"}

    centroid_dir = landmark_eci_m.mean(axis=0)
    centroid_dir /= np.linalg.norm(centroid_dir)
    pos_guess = centroid_dir * (R_EARTH_M + altitude_guess_m)
    x0 = np.concatenate([pos_guess, np.zeros(3)])  # [position(3), rotvec(3)]

    def residuals(x):
        position, rotvec = x[:3], x[3:6]
        r_eci_to_body = Rotation.from_rotvec(rotvec).as_matrix()
        line_of_sight = landmark_eci_m - position[None, :]
        los_unit = line_of_sight / np.linalg.norm(line_of_sight, axis=1, keepdims=True)
        predicted_body = (r_eci_to_body @ los_unit.T).T
        return (predicted_body - bearing_body).flatten()

    result = least_squares(residuals, x0, method="lm", max_nfev=20000)
    position_solved = result.x[:3]

    final_residuals = residuals(result.x).reshape(n, 3)
    # for unit vectors a,b: |a-b|^2 = 2 - 2cos(theta) -> theta = arccos(1 - |a-b|^2/2)
    sq_err = np.sum(final_residuals**2, axis=1)
    angular_residual_rad = np.arccos(np.clip(1.0 - sq_err / 2.0, -1.0, 1.0))

    return {
        "success": bool(result.success),
        "position_eci_km": position_solved / 1e3,
        "altitude_km": (np.linalg.norm(position_solved) - R_EARTH_M) / 1e3,
        "rms_angular_residual_deg": float(np.degrees(np.sqrt(np.mean(angular_residual_rad**2)))),
        "max_angular_residual_deg": float(np.degrees(angular_residual_rad.max())),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "num_measurements": n,
    }
