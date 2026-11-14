# Localization → orbit determination (OD)

Runs localization tie points through FSW-Payload's Python batch OD (CasADi/IPOPT) on a simulated Argus pass and compares the estimated orbit with the truth. Results: `../RUNBOOK.md` → Localization → OD.

| File | Runs on | What it does |
|---|---|---|
| `simulate_pass.py` | Mac / any | simulates the orbit (two-body + J2), nadir camera and gyro; writes FSW-Payload's OD input files (`landmark_measurements.csv`, `imu_data.csv`), runs OD, scores it, plots `od_pass.png` |
| `render_pass.py` | workstation | renders each frame of a pass from the Landsat GeoTIFFs (every pixel ray-cast to the ground) as a test set for `e2e_match.py` |
| `check_tie_points.py` | Mac / any | each tie point's real error on a rendered pass, from the true poses |
| `geo_fix_errors_source.json` | — | measured per-fix errors that the simulated-localization mode samples from |
| `runs/` | — | results: `real_eu/` (real Jetson tie points, 1.4 km with `--sigma-rad 0.002`; `real_eu_sigma002/`), `geo208/` (tie-point grades of the 208 Argus-like frames) |

**Needs:** `numpy`, `scipy`, `casadi`, `matplotlib`, and an FSW-Payload checkout (`FSW_PYTHON_OD=/path/to/FSW-Payload/python_od`, or `repos/FSW-Payload/python_od` beside this repo).

**Modes of `simulate_pass.py`:**
- `--perfect`: exact tie points (checks the chain).
- default: simulated localization with the measured fix rate and errors.
- `--poses-only`: write each frame's pose for `render_pass.py`, no OD.
- `--tie-points-from RESULTS.json`: real tie points from `e2e_match.py --save-tie-points`.

Example (real tie points): see `../README.md` §5.8b.
