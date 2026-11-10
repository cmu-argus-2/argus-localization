#!/usr/bin/env python3
"""
Batch orbit determination using CasADi + IPOPT.

  • Reads landmark_measurements.csv and imu_data.csv from a dataset folder
  • Runs the fixed-bias batch OD optimizer
  • Writes CSV results to output/od_results/<dataset>/<unix_ms>/
  • If the folder also has truth.csv (od/make_dataset.py writes one), scores
    the estimate against it and adds the errors to od_result.json

Ported from cmu-argus-2/FSW-Payload python_od/ipopt_od.py @ 4a72819
(2026-06-01). Differences: --j2/--drag/--rk4 are BooleanOptionalAction
(upstream uses store_true with default=True, so they can never be turned
off), the output folder, the IPOPT/outlier knobs exposed as flags, and the
truth scoring.

Usage (from the repository root, pri_env):
    python -m od.ipopt_od <dataset_folder> [--no-j2] [--no-drag] [--no-rk4]

Example:
    python -m od.make_dataset output/od_datasets/sim_regression --noise regression
    python -m od.ipopt_od output/od_datasets/sim_regression
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from od.data_loader import J2000_EPOCH_UNIX_S, load_imu_csv, load_landmark_csv
from od.optimizer import build_and_solve_with_outlier_rejection, save_results
from od.residuals import IntegratorType


def score_against_truth(results: dict, truth_csv: Path) -> dict:
    """RMS/max position, velocity and attitude error of the estimate vs truth.csv."""
    truth = np.loadtxt(truth_csv, delimiter=",", skiprows=1)
    truth_t = truth[:, 0] / 1000.0 - J2000_EPOCH_UNIX_S
    idx = [int(np.argmin(np.abs(truth_t - t))) for t in results["state_timestamps"]]
    truth = truth[idx]

    pos_err_km = np.linalg.norm(results["positions"] - truth[:, 1:4], axis=1)
    vel_err_ms = np.linalg.norm(results["velocities"] - truth[:, 4:7], axis=1) * 1000.0
    q_est = np.asarray(results["quaternions"])
    q_est = q_est / np.linalg.norm(q_est, axis=1, keepdims=True)
    att_err_deg = np.degrees(
        (Rotation.from_quat(truth[:, 7:11]).inv() * Rotation.from_quat(q_est)).magnitude()
    )
    return {
        "rms_position_error_km": float(np.sqrt(np.mean(pos_err_km**2))),
        "max_position_error_km": float(pos_err_km.max()),
        "rms_velocity_error_ms": float(np.sqrt(np.mean(vel_err_ms**2))),
        "rms_attitude_error_deg": float(np.sqrt(np.mean(att_err_deg**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Python batch OD (CasADi/IPOPT)")
    parser.add_argument("dataset_folder", type=Path,
                        help="Path to dataset folder containing "
                             "landmark_measurements.csv and imu_data.csv")
    parser.add_argument("--j2",   action=argparse.BooleanOptionalAction, default=True,
                        help="J2 perturbation (default: on)")
    parser.add_argument("--drag", action=argparse.BooleanOptionalAction, default=True,
                        help="Atmospheric drag, Cd estimated (default: on)")
    parser.add_argument("--rk4",  action=argparse.BooleanOptionalAction, default=True,
                        help="RK4 integrator; --no-rk4 uses forward Euler (default: RK4)")
    parser.add_argument("--covariance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mahal-threshold", type=float, default=5.0)
    parser.add_argument("--huber-m", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--ipopt-print-level", type=int, default=5)
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    run_unix_ms    = int(time.time() * 1000)

    # ── Load measurements ───────────────────────────────────────────────────────
    lm_csv  = dataset_folder / "landmark_measurements.csv"
    imu_csv = dataset_folder / "imu_data.csv"

    print(f"Loading landmark measurements from {lm_csv} …")
    landmark_measurements, landmark_group_starts, landmark_uncertainties = \
        load_landmark_csv(lm_csv)

    print(f"Loading IMU data from {imu_csv} …")
    gyro_measurements = load_imu_csv(imu_csv)

    print(f"Landmark rows   : {len(landmark_measurements)}")
    print(f"Landmark groups : {landmark_group_starts.sum()}")
    print(f"Gyro rows       : {len(gyro_measurements)}")

    results_dir = Path("output/od_results") / dataset_folder.name / str(run_unix_ms)

    # ── Solve ───────────────────────────────────────────────────────────────────
    integrator = IntegratorType.RK4 if args.rk4 else IntegratorType.FORWARD_EULER

    t0 = time.time()
    results = build_and_solve_with_outlier_rejection(
        landmark_measurements,
        landmark_group_starts,
        gyro_measurements,
        landmark_uncertainties = landmark_uncertainties,
        integrator_type    = integrator,
        use_j2             = args.j2,
        use_drag           = args.drag,
        cd_nominal         = 2.2,
        cd_std             = 1.0 if args.drag else None,
        ipopt_opts         = {"ipopt.print_level": args.ipopt_print_level},
        compute_covariance = args.covariance,
        mahal_threshold    = args.mahal_threshold,
        max_iterations     = args.max_iterations,
        landmark_huber_M   = args.huber_m,
    )
    run_time_ms = int((time.time() - t0) * 1000)

    # ── Report ──────────────────────────────────────────────────────────────────
    bias = np.asarray(results["gyro_bias"]).ravel()
    print(f"\nEstimated gyro bias : {bias} rad/s")
    print(f"Final position      : {results['positions'][-1]} km")
    print(f"Final velocity      : {results['velocities'][-1]} km/s")
    print(f"Solve time          : {run_time_ms / 1000:.1f} s")

    meta = {
        "dataset_folder":        str(dataset_folder),
        "run_unix_ms":           run_unix_ms,
        "run_time_ms":           run_time_ms,
        "use_j2":                args.j2,
        "use_drag":              args.drag,
        "integrator":            integrator.value,
        "inputs": {
            "num_landmark_rows":   int(len(landmark_measurements)),
            "num_landmark_groups": int(landmark_group_starts.sum()),
            "num_gyro_rows":       int(len(gyro_measurements)),
        },
        "outputs": {
            "num_state_estimates": int(len(results["state_timestamps"])),
            "covariance_available": "pos_var" in results,
        },
    }

    truth_csv = dataset_folder / "truth.csv"
    if truth_csv.exists():
        errors = score_against_truth(results, truth_csv)
        meta["truth_errors"] = errors
        print(f"RMS position error  : {errors['rms_position_error_km']:.3f} km "
              f"(max {errors['max_position_error_km']:.3f} km)")
        print(f"RMS velocity error  : {errors['rms_velocity_error_ms']:.3f} m/s")
        print(f"RMS attitude error  : {errors['rms_attitude_error_deg']:.3f} deg")

    # ── Save ────────────────────────────────────────────────────────────────────
    save_results(results, results_dir, meta=meta)


if __name__ == "__main__":
    main()
