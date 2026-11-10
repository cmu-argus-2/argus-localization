"""
CSV loading and state timestamp generation for the batch orbit determination problem.

Ported from cmu-argus-2/FSW-Payload python_od/data_loader.py @ 4a72819 (2026-06-01).
The readers and get_state_timestamps() are verbatim. Added below them: CSV
writers producing the same two files the flight OD reads, and
landmark_rows_from_adapter(), which converts integration/batchopt_adapter.py's
output (meters, Julian-date timestamps) into those files' units (km, unix ms).
"""
from pathlib import Path
import math

import numpy as np

# Unix timestamp of the J2000 epoch (2000-01-01 12:00:00 TT ≈ UTC)
J2000_EPOCH_UNIX_S: float = 946727936.0
DPS_TO_RADPS: float = math.pi / 180.0


def load_landmark_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read landmark_measurements.csv.

    Columns: timestamp_ms, bearing_x, bearing_y, bearing_z,
             eci_x_km, eci_y_km, eci_z_km, group, sigma

    Returns
    -------
    landmark_measurements : (N, 7) float64
        [t_j2000, bx, by, bz, ex, ey, ez]
    group_starts : (N,) bool
        True at the first row of each new landmark group.
    uncertainties : (N,) float64
        Per-measurement sigma values.
    """
    path = Path(path)
    rows: list[list[float]] = []
    group_starts: list[bool] = []
    uncertainties: list[float] = []
    prev_group = -1

    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split(",")
            if len(tok) < 9:
                continue
            try:
                t_j2000 = float(tok[0]) / 1000.0 - J2000_EPOCH_UNIX_S
                bx, by, bz = float(tok[1]), float(tok[2]), float(tok[3])
                ex, ey, ez = float(tok[4]), float(tok[5]), float(tok[6])
                group      = int(tok[7])
                sigma      = float(tok[8])
            except (ValueError, IndexError):
                continue
            rows.append([t_j2000, bx, by, bz, ex, ey, ez])
            group_starts.append(group != prev_group)
            uncertainties.append(sigma)
            prev_group = group

    return (
        np.array(rows,         dtype=np.float64),
        np.array(group_starts, dtype=bool),
        np.array(uncertainties, dtype=np.float64),
    )


def load_imu_csv(path: Path) -> np.ndarray:
    """
    Read imu_data.csv.

    Columns: Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps

    Returns
    -------
    gyro_measurements : (M, 4) float64
        [t_j2000, wx_rad_s, wy_rad_s, wz_rad_s]
    """
    path = Path(path)
    rows: list[list[float]] = []

    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split(",")
            if len(tok) < 4:
                continue
            try:
                t_j2000 = float(tok[0]) / 1000.0 - J2000_EPOCH_UNIX_S
                wx = float(tok[1]) * DPS_TO_RADPS
                wy = float(tok[2]) * DPS_TO_RADPS
                wz = float(tok[3]) * DPS_TO_RADPS
            except (ValueError, IndexError):
                continue
            rows.append([t_j2000, wx, wy, wz])

    return np.array(rows, dtype=np.float64)


def get_state_timestamps(
    landmark_measurements: np.ndarray,  # (N, 7): [t_j2000, ...]
    landmark_group_starts: np.ndarray,  # (N,) bool
    gyro_measurements:     np.ndarray,  # (M, 4): [t_j2000, ...]
) -> tuple[list[float], list[int]]:
    """
    Mirrors C++ get_state_timestamps() in src/navigation/batch_optimization.cpp.

    State timestamps = exactly the gyro measurement timestamps.
    Each landmark group is snapped to the nearest gyro timestamp index.

    Returns
    -------
    state_timestamps       : list[float]  — one entry per gyro measurement
    landmark_group_indices : list[int]    — index into state_timestamps per group
    """
    state_timestamps = [float(gyro_measurements[k, 0])
                        for k in range(len(gyro_measurements))]

    landmark_group_indices: list[int] = []
    M = len(state_timestamps)
    n_lmk = len(landmark_measurements)

    for i in range(n_lmk):
        if not landmark_group_starts[i]:
            continue
        t_lmk = float(landmark_measurements[i, 0])

        # Binary search for nearest gyro timestamp
        lo, hi = 0, M - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if state_timestamps[mid] < t_lmk:
                lo = mid + 1
            else:
                hi = mid

        # lo is the first index >= t_lmk; compare with lo-1
        if lo == 0:
            best_k = 0
        elif lo == M:
            best_k = M - 1
        else:
            if state_timestamps[lo] - t_lmk < t_lmk - state_timestamps[lo - 1]:
                best_k = lo
            else:
                best_k = lo - 1

        landmark_group_indices.append(best_k)

    return state_timestamps, landmark_group_indices


# ── Writers (argus-localization additions) ──────────────────────────────────────

LANDMARK_CSV_HEADER = "timestamp_ms,bearing_x,bearing_y,bearing_z,eci_x_km,eci_y_km,eci_z_km,group,sigma"
IMU_CSV_HEADER = "Timestamp_ms,Gyro_X_dps,Gyro_Y_dps,Gyro_Z_dps"
JD_UNIX_EPOCH: float = 2440587.5


def landmark_rows_from_adapter(
    measurements: np.ndarray,   # (N, 7) from batchopt_adapter.to_batchopt_measurements
    group_starts: np.ndarray,   # (N,) bool
    sigma_rad:    np.ndarray,   # (N,) per-measurement bearing sigma [rad]
) -> np.ndarray:
    """
    Convert batchopt_adapter output into landmark_measurements.csv rows.

    The adapter emits [jd, bearing_body(3), landmark_eci_m(3)]; the flight OD
    expects [unix_ms, bearing_body(3), landmark_eci_km(3), group, sigma].

    Returns
    -------
    rows : (N, 9) float64, ready for write_landmark_csv
    """
    rows = np.zeros((len(measurements), 9))
    rows[:, 0] = np.round((measurements[:, 0] - JD_UNIX_EPOCH) * 86400.0 * 1000.0)
    rows[:, 1:4] = measurements[:, 1:4]
    rows[:, 4:7] = measurements[:, 4:7] / 1000.0
    rows[:, 7] = np.cumsum(group_starts) - 1
    rows[:, 8] = sigma_rad
    return rows


def write_landmark_csv(path: Path, rows: np.ndarray) -> None:
    """Write (N, 9) [unix_ms, bx, by, bz, ex_km, ey_km, ez_km, group, sigma] rows."""
    with open(path, "w") as f:
        f.write(LANDMARK_CSV_HEADER + "\n")
        for row in rows:
            f.write(f"{int(row[0])}," + ",".join(f"{v:.12g}" for v in row[1:7])
                    + f",{int(row[7])},{row[8]:.12g}\n")


def write_imu_csv(path: Path, timestamps_ms: np.ndarray, gyro_rad_s: np.ndarray) -> None:
    """Write imu_data.csv from (M,) unix-ms timestamps and (M, 3) body rates [rad/s]."""
    gyro_dps = gyro_rad_s / DPS_TO_RADPS
    with open(path, "w") as f:
        f.write(IMU_CSV_HEADER + "\n")
        for t, w in zip(timestamps_ms, gyro_dps):
            f.write(f"{int(t)}," + ",".join(f"{v:.12g}" for v in w) + "\n")
