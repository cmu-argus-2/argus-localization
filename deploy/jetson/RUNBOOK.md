# Jetson Localization Runbook

Running AstroLoc localization on the Jetson Orin Nano: photo → retriever → search → matcher → location.

## Status (current)

**The full localization flow runs on the Jetson** — TensorRT retriever, database search and the SIFT + LightGlue matcher in one process — and reproduces the workstation.

**Now:** full-size AstroLoc retriever (106.5M parameters; ~0.1 s of the total) + TensorRT LightGlue: 0.44 s and ~3.3 J per photo. Fix on 54% of astronaut photos and 60% of Argus-like frames; on Argus-like frames the tie points are within 0.7 km median. Real tie points into FSW-Payload OD on a simulated pass: 1.4 km orbit error.

### Retrieval only (rough location)

| Jetson Orin Nano 8GB, TensorRT FP32 | Nano v2 | Full-size |
|---|---|---|
| Top-1 / Top-10 / Top-100 | 34.5% / 50.5% / 60.5% | **68.5% / 89.0% / 98.5%** |
| Median error | 1,042 km | **79 km** |
| On the 169 photos excluded from training | 33.7%, 1,303 km | **64.5%, 86 km** |
| Time per photo | ~67 ms | ~109 ms |
| Board memory, idle → peak (of 7,620 MB) | 2,345 → 2,686 MB | 2,523 → 3,351 MB |
| Board power, peak | 6.9 W | 8.2 W |
| Fingerprint database | 62 MB | 249 MB |

Same top tile as the workstation on all 200 photos, for both models.

### Full flow: full-size retriever + matcher

First full-flow run (top 15 tiles, before the speed work). The matcher declares a fix when a tile has ≥ 30 inlier points.

| 200 photos | Jetson | Workstation |
|---|---|---|
| Correct tile in the top 15 | 91% | 91% |
| Fix rate | 54.5% (109) | 54.0% (108) |
| Fixes on a correct tile | 92.7% | 92.6% |
| Fix error: median / 90th percentile / max | **15.9 km** / 33.2 km / 272 km | 16.0 km / 33.7 km / 272 km |
| Fixes more than 50 km off | 4 | 4 |

- Same fix / no-fix decision as the workstation on 199 of 200 photos; where both fix, the locations differ by 0.02 km median (5.9 km max).
- On the 169 photos excluded from training: 53.8% fix rate, 16.1 km median error.

### Speed on the Jetson: 3.3 s → 0.44 s per photo

Each step was checked on the same 200 photos; accuracy held at every step (fix on ~55% of photos, ~16 km median error).

| Change | Time per photo | Energy per photo |
|---|---|---|
| Full flow on the Jetson, first version | 3.3 s | ~26 J |
| Tile SIFT features computed once and stored | 1.3 s | ~14 J |
| Top 10 candidate tiles instead of 15 | 0.95 s | ~10 J |
| LightGlue in half precision | 0.81 s | ~7 J |
| Search on the GPU + tile features in one file | 0.73 s | ~6 J |
| LightGlue as a TensorRT FP16 engine, 1,024 points | 0.64 s | ~5 J |
| **Stop at the first tile with ≥ 30 inliers instead of matching all 10 (current)** | **0.44 s mean** (0.34–0.38 s median) | **~3.3 J** |

With the stop rule, a photo that gets a fix takes 0.2–0.4 s; a photo with no fix still checks all 10 tiles (~0.65 s, the worst case).

Tried and not adopted: see *Experiment log* below.

### Accuracy: original vs now

Same 200 held-out photos.
- **Fix rate:** % of photos the system locates confidently (≥ 30 matched points on a tile). The rest get "no fix": no answer rather than a guess.
- **Error:** km between the estimated and the true photo centre, for fixes only.

| | Original (3.3 s) | TensorRT LightGlue, all 10 tiles (0.65 s) | **Current: + stop at first good tile (0.45 s mean)** |
|---|---|---|---|
| Setup | top 15 tiles, full precision | top 10 tiles, LightGlue as a TensorRT FP16 engine | same, stop at the first tile with ≥ 30 inliers |
| Fix rate | 54.5% (109) | 54.0% (108) | 54.0% (108) |
| Median error | 15.9 km | 15.9 km | 15.6 km |
| 90% of fixes within | 33.2 km | 32.0 km | 32.5 km |
| Fixes > 50 km off | 4 | 3 | 3 |

Retrieval is identical in all three (right tile ranked first on 68.5% of photos).

### Setup
- **Nano v2:** `nano_train/checkpoints_v2/final.pt` (27M, 512-d).
- **Full-size:** `astroloc_train/checkpoints_faithful_lora_v2/final.pt` (106.5M, 2048-d).
- **Database:** all 7,602 zoom-9 tiles from 2021, 4 rotations each, plus their stored SIFT features (3.2 GB).
- **Test:** 200 astronaut photos near the Alps; "169 photos" = those training can't have seen.

### Takeaways
- **Full-size beats nano v2:** 2× the hit rate, 13× lower error.
- **The matcher is essential:** 79 km → 16 km by footprint centre on astronaut photos; tie points within 0.7 km on Argus-like frames.
- **8 GB is enough:** 4.0 GB peak memory for the whole flow.
- **Time is the main cost:** 0.44 s per photo on average now (3.3 s originally), mostly LightGlue on the GPU.
- **Wrong fixes:** 3 of 108 astronaut-photo fixes are > 50 km off (none on Argus-like frames); filter before OD.

## Square Argus-like frames

208 simulated Argus frames: centre squares of the Landsat-8 GeoTIFFs (`/mnt/sda2/geotiffs`, 13 per zone across 16 zones), ~454 km across, resampled to 1024 × 1024. Built with `prepare_geotiff_set.py`; on the Jetson in `e2e_geotiff/`. Same database (2021 zoom 9, full-size model).

| 208 frames, Jetson | LightGlue PyTorch, half precision | LightGlue TensorRT, 1,024 points, all 10 tiles | **+ stop at first tile with ≥ 30 inliers (current)** | LightGlue TensorRT, 768 points |
|---|---|---|---|---|
| Fix rate | 57.2% (119) | 59.6% (124) | **59.6% (124)** | 42.8% (89) |
| Median error / 90% within | 6.7 / 11.0 km | 6.7 / 10.9 km | **6.5 / 10.0 km** | 6.5 / 10.1 km |
| Fixes > 50 km off | 0 | 0 | 0 | 0 |
| **Tie-point error, median / worst fix** | | | **0.7 / 1.0 km** | |
| Time per photo | 0.74 s | 0.64 s | **0.44 s mean, 0.34 s median** | 0.56 s |
| Energy per photo | ~6 J | ~5 J | **~3.3 J** | ~4 J |

- **The real accuracy is ~0.7 km.** "Median error" above is the frame centre found by extrapolating the homography from the matched tile (~60 km) to the corners of the ~454 km frame, which inflates small errors. Graded at the tie points (each point vs the true ground point of its pixel from the GeoTIFF geotransform, `grade_tie_points.py`): 0.7 km median per fix, 0.8 km p90, 1.0 km worst, on all 124 fixes. The tie points are what OD uses.
- Near-nadir frames like Argus's localize far better than astronaut photos: 6.7 km vs 16 km median by the same centre measure, and no fix more than 50 km off. (Astronaut photos have truth only at the corners, so they can't be graded at the tie points.)
- **Fixes depend on latitude/season:** southern zones (R, S; ~25–40° N) fix 12–13 of 13 in most zones; northern zones (T; ~40–48° N) fail often (16T 0, 17T 1, 32T 3, 54T 3, 10T 6, 33T 7 of 13). The northern GeoTIFFs are mosaics of Landsat scenes from different seasons, many snow-covered or winter-brown with hard seams, while the database tiles (2021) are cloud-free, snow-free summer-looking imagery. See Known issues.
- Correct tile among the 10 checked: 77.9%. Each frame is larger than a tile, so only a few tiles overlap it well.
- **Stopping at the first good tile keeps every fix:** tiles checked drop from 10 to 5.1 per frame on average.
- **TensorRT LightGlue at 1,024 points beats the PyTorch version here** (+5 fixes, 0.1 s faster), so it is the current setup. On the astronaut photos it loses 2 fixes. Fewer points (768) costs too many fixes on these frames.

## Real two-frame pairs

OD needs at least two photos from the same pass. These are real consecutive photos, not a simulated second frame. Files are in `deploy/jetson/two_frame/`.

| Step | Result |
|---|---|
| Held-out-region photos, same mission and day, 200–1,000 km apart, both with a correct tile | 15,275 candidate pairs (3,037 photos) |
| Sample of 40 pairs, capture times from NASA's photo pages | **38 confirmed same pass**: ISS moved ~6.9 km/s between the two NASA nadir points |
| Time gap within a pass | 12–147 s, median 55 s |
| Both photos localized on the Jetson (current setup) | **8 of 38 pairs**, 6 of them with both errors under 50 km |

- **Fix rate:** 43% per photo here, against 55% on the Alps test. Needing both photos of a pair to get a fix leaves about 1 pair in 5.
- **Scaling up:** about 1 pair in 5 would give roughly 3,000 usable pairs from the 15,275 candidates.
- **One fix with 245 inliers was 90 km off**, so even a high inlier count doesn't guarantee a good fix.
- **Next:**
  - fetch times for more pairs
  - run OD on the pairs with both fixes. The astronaut photos are oblique, so the photo centre is not below the ISS; the OD step has to allow for that or use the tie points.

## Localization → OD

`od/simulate_pass.py` runs one simulated Argus pass through FSW-Payload's Python batch OD (`python_od` in an FSW-Payload checkout, default `repos/FSW-Payload/python_od`, or set `FSW_PYTHON_OD`; CasADi/IPOPT; needs `casadi` and `scipy`). OD solves position, velocity, attitude and gyro bias from landmark bearings plus gyro.

1. **Truth:** circular orbit at 600 km (two-body + J2), 97.8° inclination, nadir-pointing camera, gyro at 1 Hz with bias and noise.
2. **Frames:** a 1024 × 1024 frame every 20 s (Argus K, centre square). Each frame's 8 × 8 pixel grid is ray-cast to the WGS84 ellipsoid.
3. **Simulated localization:**
   - a fix on 59.6% of frames;
   - each fixed frame is shifted by one error drawn from the 208 simulated Argus images' real fix errors (median ~7 km), in a random direction, plus 0.3 km per-point noise;
   - every point is a tie point (u, v, lat, lon), like the pipeline's output.
4. **OD input:** tie points → `landmark_measurements.csv` (body-frame bearing, landmark ECI, per-point sigma = median error / altitude) + `imu_data.csv`, FSW-Payload's own format.
5. **Output:** `od/runs/<name>/`: `od_summary.json`, `od_states.csv`, `truth.csv`, `od_pass.png` (ground track, OD error over time).

| Pass (3 random seeds each) | Frames with a fix | OD position error, median | OD velocity error, median |
|---|---|---|---|
| Perfect localization (check of the chain) | 31 of 31 | 0.15 km | 1.0 m/s |
| 10 min, measured fix rate and errors | 19–21 of 31 | 2.4–2.9 km | 4–19 m/s |
| 25 min, measured fix rate and errors | ~45 of 76 | 0.8–2.3 km | 2–5 m/s |

- **OD beats each single fix:** ~7 km per frame becomes ~1–3 km after OD, because the orbit dynamics average many frames.
- **A longer arc helps most:** 10 min leaves the velocity weak, so the error grows toward the ends.
- **Simplifications:**
  - Earth rotation is GMST only;
  - body frame = camera frame, pointing at nadir (the plan; the real camera may end up partly sideways);
  - focal length fixed at the config value (fx 3548; camera spec not final);
  - no drag;
  - no wrong fixes (none on the simulated Argus images);
  - errors are simulated from measured statistics (see below for real frames).

### Real localization along the orbit

The same pass, but the tie points come from the real pipeline on the Jetson:

1. `simulate_pass.py --poses-only` writes each frame's position and attitude.
2. `od/render_pass.py` (workstation) ray-casts every pixel of each 1024 × 1024 frame to the ground and samples the Landsat GeoTIFFs. Frames are true perspective views; frames more than 20% empty are skipped.
3. The Jetson runs the default setup with `--save-tie-points`.
4. `simulate_pass.py --tie-points-from` feeds those tie points to OD.

Pass: 600 km, 97.8°, from Libya (31.5° N, 13.5° E) north over Tunisia, Sicily, Sardinia/Corsica and northern Italy. A frame every 10 s for 330 s, 29 of 34 rendered (the last 5 leave the GeoTIFF coverage).

| | Result |
|---|---|
| Frames with a fix | 11 of 29 (6 frames are mostly sea and have no correct tile) |
| Tie-point error (each point vs the true ground point of its pixel) | **~1.2 km median**, ≤ 2.2 km max on every fix; no wrong fixes |
| Fix error as logged (footprint centre) | 6.1 km median, 35 and 74 km on two frames |
| OD position error | **1.40 km median**, 2.9 km max, with point sigma 0.002 rad (≈ 1.2 km); 2.13 km with sigma from the logged fix error (0.011 rad) |
| OD velocity error | 12 m/s median (20 m/s with 0.011 rad) |
| Gyro bias | recovered to ~5% |

- **The logged fix error overstates the real error.** It extrapolates the homography from the matched tile (~60 km) to the corners of the whole ~440 km frame. When the tie points sit in a small part of the frame (t40, t50: a strip at the top) that extrapolation is far off, even though the points themselves are right. A least-squares homography on the tie points, mapped at the centre pixel, gives 1.2 km median (31 km on t50). On rendered frames the truth "centre" (mean of the 4 corners) is also 2–4 km from the centre pixel's ground point.
- **OD uses the tie points, so it was never affected;** only the point sigma was set too loose. `od/check_tie_points.py` prints these per-frame numbers.
- Same on the 208 Argus-like frames: tie points 0.7 km median against 6.5 km logged (see *Square Argus-like frames*).
- **Fix rate is lower than on the 208 Argus-like frames (38% vs 60%) because of where the pass goes, not the rendering:** the southern half (Libya–Sicily, zones 32S/33S) fixes most frames; northern Italy and the Alps (32T/33T) fix 1 of 12. The 208-frame set's own 32T frames fix 3 of 13 for the same reason: snow and winter scenes in the GeoTIFFs vs a summer-looking database (see Known issues).
- Files: `od/runs/real_eu/` (`match_results_pass.json`, `od_pass.png`, `od_summary.json`).

## Where the time goes

Measured on the PyTorch LightGlue setup (half precision, 0.73 s); the TensorRT engine now takes ~34 ms per tile in place of ~58 ms, and the stop rule checks ~5 tiles instead of 10. 50 photos, Jetson in 15 W mode, nothing else running. "CPU" is the Jetson's ARM CPU (6 cores) and "GPU" is its GPU. Timing each step separately adds ~0.07 s, so the total reads 0.81 s against 0.73 s in a plain run. Produced by `profile_full.py --lightglue-mp --tile-pack ../tile_feats_pack --search-on-gpu`.

| Step | Runs on | ms per photo | Share |
|---|---|---|---|
| Read photo from disk | disk | 0.8 | 0.1% |
| Decode JPEG | CPU | 11.3 | 1.4% |
| Resize to 224 + normalize | CPU | 7.6 | 0.9% |
| Retriever (TensorRT), with copies in/out | GPU | 51.9 | 6.4% |
| Search 7,602 tiles | GPU | 5.2 | 0.6% |
| Photo SIFT: copy to GPU, resize to 512 | GPU | 34.8 | 4.3% |
| Photo SIFT: OpenCV SIFT | CPU, ~3 cores | 85.2 | 10.6% |
| Tile features: read from the packed file (10 tiles) | CPU | 1.2 | 0.2% |
| Tile features: copy to GPU | GPU | 11.8 | 1.5% |
| **LightGlue, 10 tiles** | **GPU** | **581.4** | **72.1%** |
| Results to CPU + MAGSAC check | CPU | 11.0 | 1.4% |
| Decide + georeference | CPU | 3.7 | 0.5% |
| **Total** | | **806** | |

- **LightGlue is the bottleneck** (72%, ~58 ms per tile). Everything before it, including the 224 px retriever, takes ~0.21 s.
- **The next-biggest step is SIFT on the photo** (85 ms on the CPU). OpenCV's SIFT has no GPU version here.
- **Inside LightGlue, the ARM CPU is the limit, not the GPU:**
  - The CPU thread launching its ~850 small GPU kernels per tile is busy 99% of the time.
  - The GPU averages 53% busy.
  - Before half precision, the GPU was the limit: busy 81% of each call (`nsys`), 59% of that in 32-bit matrix multiplies.
- **Disk is not a bottleneck:** reading the photo takes under 1 ms, and the 10 tiles' features ~1 ms from the packed file.
- **Clocks in 15 W mode:** CPU up to 1.51 GHz, GPU up to 625 MHz.

### Memory

The GPU has no separate memory; it shares the board's 7.6 GB.

| | MB |
|---|---|
| Board RAM, running (peak) | 3,890 (3,896) |
| This process, peak | 1,969 |
| … fingerprint database | 238, held twice: a CPU copy and a GPU copy for the search |
| … TensorRT engine weights + working memory | 409 + 14 |
| … PyTorch GPU tensors, peak (reserved), including the GPU database copy | 335 (366) |
| … the rest: CUDA, cuDNN and cuBLAS libraries, Python | ~1,200 |

### Power and battery (approximate)

**Jetson** (`tegrastats` VDD_IN, module total input, 15 W mode): **~7.6 W while localizing, ~3.3 J per photo** (4.3 W idle).

**Battery** (Argus BatteryBoard v3.1, [cmu-argus-2/Avionics-BatteryBoard](https://github.com/cmu-argus-2/Avionics-BatteryBoard)): 6 × Panasonic NCR18650GA in 2S3P, 7.2 V, 10.35 Ah ≈ **74.5 Wh**.

| | Energy | Share of battery |
|---|---|---|
| One photo | 3.3 J | ~0.001% |
| One pass as planned (2 photos) | ~7 J | ~0.003% |
| 100 photos | 330 J (0.09 Wh) | ~0.1% |
| Localizing nonstop for 1 h | ~7.6 Wh | ~10% |

- The main board switches the Jetson on about twice per orbit (capture, process, downlink) and buffers the data until a ground-station pass, so it never sits idle.
- Not included: Jetson startup (not yet measured), converter losses, cold, depth-of-discharge limits.

### What would make it faster, biggest likely win first
Done and rejected items are in the Experiment log. The TensorRT LightGlue engine takes ~34 ms per tile.
1. **Higher power mode (MAXN SUPER, JetPack 6.2):** GPU 625 → 1,020 MHz. A LightGlue of the same size runs at 21.6 ms per pair on an Orin Nano in this mode (kornia/lightglue on Hugging Face). Needs a reflash, so it's a team decision.
2. **Tracking mode:** after the first fix, use OD's predicted position, skip the retriever and match only the 1–3 tiles near it. Part of the OD integration.
3. **Photo SIFT (CPU) in parallel with the retriever (GPU):** up to ~50 ms.
4. **SIFT on the GPU (CudaSift):** ~80 ms, but all tile features must be recomputed with it.
5. **LightGlue's own early exit and point pruning:** off in the fixed-size TensorRT engine; hard to add.
6. **XFeat + LighterGlue:** ~8 ms per pair, public weights, but poor beyond ~45° of rotation.

## Experiment log

One change at a time against a same-day baseline, on the 200 astronaut photos (and the 208 Argus-like frames where noted). Per-run results are on the Jetson in `e2e_alps_full/` and `e2e_geotiff/` (`match_results_<tag>.json`, `match_run_<tag>.log`).

| Change | Time per photo | Accuracy | Energy | Decision |
|---|---|---|---|---|
| Store tile SIFT features (3.2 GB) | 3.3 → 1.3 s | identical | 26 → 14 J | adopted |
| Top 10 tiles instead of 15 | 1.31 → 0.95 s | identical | 14 → 10 J | adopted |
| Top 5 tiles | 0.59 s | −2 of 109 fixes | 6 J | no |
| LightGlue half precision | 0.95 → 0.81 s | +1 fix | 10 → 7 J | adopted |
| 512 SIFT points (alone or + half precision) | ~0.8 s | −19 fixes | 6–7 J | no |
| Search on GPU + memory-mapped tile features | 0.82 → 0.73 s | identical | 7 → 6 J | adopted |
| Tile features in GPU memory | 0.74 s | identical, +1.8 GB | 6 J | no |
| Batched LightGlue, PyTorch | 0.83 s | −1 fix | 8 J | no |
| **LightGlue TensorRT FP16** | 0.74 → 0.65 s | −2 fixes; Argus-like +5 | 6 → 5 J | **adopted** |
| LightGlue TensorRT FP32 | 1.18 s | +2 fixes | 13 J | no: slower |
| Tiny retriever (27M, 512-d) | 0.70 s | 54.0% vs 55.0% fix; −0.5 GB | 6 J | backup |
| 10 tiles per TensorRT call | −6% engine time | — | — | no |
| 768 points, TensorRT | 0.56 s | Argus-like −31 of 119 | 4 J | no |
| Tiny + 768 points | 0.55 s | −6 fixes | 4 J | no |
| LightGlue 7 of 9 layers (FastGlue idea) | 0.57 s | Argus-like −6 of 124 | 5 J | no (option if speed matters more) |
| LightGlue 6 or 5 layers | — | Argus-like −6 to −8 | — | no |
| **Stop at first tile with ≥ 30 inliers** | 0.65 → 0.44 s mean | same fixes on both sets | 5 → 3.3 J | **adopted** |

The workstation (RTX 4090) runs the flow in 0.22 s with stored features.

## What runs on the Jetson

One Python process (`e2e_match.py`). The ARM CPU (6 cores) and the GPU share memory.

| Step | What happens | Runs on |
|---|---|---|
| 1. Retriever | photo (224 × 224) → 2048-number fingerprint, `astroloc_full_fp32.trt` | GPU |
| 2. Search | against 7,602 tiles × 4 rotations, keep the top 10 | GPU |
| 3. Feature points | photo SIFT, 1,024 points at 512 px; tile points read from `tile_feats_pack` | CPU |
| 4. Matching | LightGlue TensorRT FP16 engine, photo vs each tile in retrieval order | GPU |
| 5. Geometry check | MAGSAC homography (3 px), count inliers; stop at the first tile with ≥ 30 | CPU |
| 6. Output | the tile's inlier points as tie points (photo pixel → lat/lon) and the photo's footprint | CPU |

- **Repo code used unmodified** (argus-localization `cc75c2b`): `SiftLightGlueMatcher`, `Georeferencer`, `core/types.py`, `data_loading/earthloc_loader.py`. Ours: `e2e_match.py`, `lightglue_trt.py`, the tile-feature packing and the TensorRT conversions.
- **Differences from `core/pipeline.py`:** TensorRT retriever (cosine 0.99997 vs PyTorch); photo SIFT extracted once per photo; stop rule instead of always checking every tile.
- **Dependencies:** JetPack 6.0 (L4T R36.3, CUDA 12.2, TensorRT 8.6.2, OpenCV 4.8.0) on the board; PyTorch 2.4.0a0 (NVIDIA wheel), numpy 1.26.4, kornia 0.7.1, LightGlue `eb42fee` and PyTorch's small deps in `~/Desktop/astroloc/pydeps` (`pip3 install --no-deps --target`, used via `PYTHONPATH`; nothing system-wide, no sudo).
- **Disk:** engines ~0.5 GB, fingerprints 249 MB, `pydeps` 2.0 GB, packed tile features 2.15 GB, tile images 720 MB (their filenames carry the tile corners).
- **Handoff to OD:** `--save-tie-points` writes each fix's tie points; `od/simulate_pass.py` turns them into FSW-Payload's input files (see *Localization → OD*).

## Files

**Workstation** (`~/`): `nano_v2_static.onnx`, `astroloc_full_static.onnx`, `e2e_alps_v2/`, `e2e_alps_full/` (database + test photos + workstation results). The workstation's stored tile features (3.2 GB) were deleted after the run because the home disk is 97% full; `precompute_tile_feats.py` rebuilds them in 6 min.
**Jetson** (`~/Desktop/astroloc/`): `nano_v2_fp32.trt`, `astroloc_full_fp32.trt`, `e2e_alps_v2/`, `e2e_alps_full/`, `e2e_jetson.py`, `e2e_match.py`, `run_match_full.sh`, `tiles/2021_09/` (tile images), `tile_feats_2021_09/` (stored tile SIFT features), `precompute_tile_feats.py`, `profile_match.py`, `repo/` (matcher and georeferencing code from argus-localization), `pydeps/` (PyTorch, kornia, LightGlue), `wheels/` (installers).

Superseded: `nano_fp32.trt` / `nano_static.onnx` (nano **v1**, 3% top-1), `nano_fp16.trt` (FP16 rejected), `astroloc_fp32.trt` / `astroloc_static.onnx` (oldest full-size checkpoint).

## Reproduce

Commands are for the full-size model. For nano v2, use `nano_train/checkpoints_v2/final.pt` and the `nano_v2` / `e2e_alps_v2` names.

### 1. Workstation

Build the database, export the model, run the workstation side:
```bash
cd ~/argus-localization
CKPT=/mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora_v2/final.pt
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_prepare.py --checkpoint $CKPT --out ~/e2e_alps_full
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/export_onnx.py --checkpoint $CKPT \
    --output ~/astroloc_full_static.onnx --opset 17 --batch 1 --static-batch
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_match.py \
    --data ~/e2e_alps_full --tiles-dir /mnt/sdc1/astroloc/data/database/2021_09
```
Check `nvidia-smi` first and use a free GPU.

### 2. Jetson, retrieval only
```bash
cd ~/Desktop/astroloc
scp soham@172.24.58.65:~/astroloc_full_static.onnx .
scp -r soham@172.24.58.65:~/e2e_alps_full .
/usr/src/tensorrt/bin/trtexec --onnx=astroloc_full_static.onnx --saveEngine=astroloc_full_fp32.trt
python3 e2e_jetson.py --engine astroloc_full_fp32.trt --data e2e_alps_full
```
Run `tegrastats` alongside for board memory and power.

### 3. Jetson, matcher setup (once)

The Jetson has no internet, so everything is downloaded elsewhere and copied into `~/Desktop/astroloc/`:
- `wheels/`: NVIDIA's PyTorch wheel for JetPack 6.0 (`developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl`), numpy 1.26.4 for aarch64 / Python 3.10, and the pure-Python wheels in the dependency table.
- `pydeps/kornia`, `pydeps/lightglue`: copied from the workstation's `~/venv-export/lib/python3.12/site-packages/`.
- `~/.cache/torch/hub/checkpoints/sift_lightglue_v0-1_arxiv.pth`: copied from the same path on the workstation.
- `tiles/2021_09/`: copied from `/mnt/sdc1/astroloc/data/database/2021_09`.
- `repo/`: the five argus-localization files listed under *Code used*, plus the empty `__init__.py` files.
```bash
cd ~/Desktop/astroloc
pip3 install --no-deps --no-index --target pydeps wheels/*.whl
printf 'from .lightglue import LightGlue  # noqa\nfrom .sift import SIFT  # noqa\n' > pydeps/lightglue/__init__.py
```

### 4. Jetson, full flow
```bash
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../e2e_match.py --data ../e2e_alps_full \
    --tiles-dir ../tiles/2021_09 --engine ../astroloc_full_fp32.trt
```
`~/Desktop/astroloc/run_match_full.sh` runs the same command with `tegrastats` logging. Results land in `e2e_alps_full/match_results.json`; the logs are `match_run.log` and `tegrastats_match.log`.

### 5. Jetson, full flow with tile features computed in advance
```bash
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../precompute_tile_feats.py \
    --tiles-dir ../tiles/2021_09 --out ../tile_feats_2021_09          # once, 16 min
~/Desktop/astroloc/run_match_full.sh pre --tile-feats ../tile_feats_2021_09
```
Results land in `match_results_pre.json`, with logs `match_run_pre.log` and `tegrastats_match_pre.log`. The same two scripts and `--tile-feats` flag work on the workstation.

### 6. Top 15 / 10 / 5 comparison
```bash
cd ~/Desktop/astroloc
for k in 15 10 5; do ./run_match_full.sh k$k --tile-feats ../tile_feats_2021_09 --top-k $k; done
python3 compare_topk.py --data e2e_alps_full --ks 15 10 5
# half precision / fewer points: archive/run_speed.sh, then
python3 compare_topk.py --data e2e_alps_full --tags base10 mp kp512 mp_kp512
```

### 7. Stage-by-stage pictures (`walkthrough/PIPELINE-WALKTHROUGH.md`)
```bash
# Jetson
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../pipeline_trace.py --out ../trace
# Mac, from deploy/jetson/, after copying ~/Desktop/astroloc/trace to walkthrough/pipeline_walkthrough/data (needs matplotlib)
python3 visualize_trace.py --trace walkthrough/pipeline_walkthrough/data --out walkthrough/pipeline_walkthrough
```

## Scripts (in `deploy/jetson/`)

| Script | Runs on | Purpose |
|---|---|---|
| `e2e_prepare.py` | workstation | builds the zoom-9 database, picks test photos, workstation results |
| `e2e_jetson.py` | Jetson | retrieval-only test with the TensorRT engine, compares to workstation |
| `e2e_match.py` | both | adds the matcher on top of retrieval; `--engine` runs the TensorRT retriever in the same process, `--tile-feats` uses stored tile features, `--stop-inliers N` stops at the first tile with ≥ N inliers, `--log-tiles` records every tile's inliers, `--save-tie-points` stores each fix's (u, v, lat, lon) points; every fix logs `tie_error_km` (tie points vs truth from the photo's corners, the error OD sees) next to `match_error_km` (footprint centre) |
| `precompute_tile_feats.py` | both | computes and stores the SIFT features of every tile, once |
| `run_match_full.sh` | Jetson | runs `e2e_match.py` with `tegrastats` logging; the first argument tags the output files |
| `archive/profile_match.py` | Jetson | times each matcher stage on 20 photos (before stored tile features) |
| `compare_topk.py` | Jetson | compares full-flow runs side by side (`--ks` or `--tags`) |
| `od/simulate_pass.py` | Mac | simulated Argus pass → simulated (or, with `--tie-points-from`, real) localization tie points → FSW-Payload Python OD, compared with truth; `--poses-only` writes frame poses for rendering |
| `od/render_pass.py` | workstation | renders a pass's frames from the Landsat GeoTIFFs as an e2e test set |
| `grade_tie_points.py` | workstation | per-tie-point error on the Argus-like GeoTIFF frames (truth from each GeoTIFF's geotransform) |
| `od/check_tie_points.py` | Mac | per-tie-point error of a rendered pass against the true poses |
| `simulate_stop.py` | any | replays stop-at-first-good-tile thresholds on a `--log-tiles` result |
| `run_stop.sh` | Jetson | timed runs, all 10 tiles vs `--stop-inliers 30`, on both test sets |
| `lightglue_trt.py` | workstation, Jetson | LightGlue as a fixed-size network: export to ONNX (`export --points --batch --layers`), check against PyTorch (`check`); used by `e2e_match.py --lightglue-trt` |
| `trt_lg_bench.py` | Jetson | time each step around the TensorRT LightGlue engine |
| `prepare_geotiff_set.py` | workstation | square Argus-like test frames from the Landsat GeoTIFFs |
| `archive/run_speed.sh` | Jetson | current vs half precision vs 512 points vs both |
| `archive/run_speed2.sh` | Jetson | current vs small CPU fixes vs preloaded tile features vs batched LightGlue |
| `pack_tile_feats.py` | Jetson | packs the 7,602 tile feature files into one set of arrays (`--tile-pack`) |
| `archive/test_batched_lightglue.py` | either | checks batched LightGlue against one tile at a time |
| `profile_full.py` | Jetson | time, CPU, GPU, memory and disk cost of every step (run with `tegrastats` for power) |
| `lg_bench.py` | Jetson | one LightGlue pair repeated, for `nsys profile -t cuda,nvtx` |
| `pipeline_trace.py`, `visualize_trace.py` | Jetson, Mac | record and draw every stage for one success and one failure; see `walkthrough/PIPELINE-WALKTHROUGH.md` |
| `export_onnx.py` | workstation | PyTorch → ONNX, with a parity check |
| `check_engine.py`, `test_images.py`, `archive/compare_precision.py` | Jetson | engine sanity, real-image check, FP32 vs FP16 |

## Environment

- **Workstation:** `~/venv-export` (inherits `sat_env_vision`'s CUDA torch, plus onnx, onnxruntime, onnxscript, shapely, `numpy<2.2`, kornia 0.7.1, LightGlue). No sudo.
- **Jetson:** versions are listed under *What runs on the Jetson*. `trtexec` is at `/usr/src/tensorrt/bin/trtexec`. pycuda 2022.2.2 is used by `e2e_jetson.py`. 838 GB free disk. No internet, so packages are downloaded elsewhere and copied over.
- **Mac → Jetson:** VS Code needs Local Network permission (System Settings → Privacy & Security). Or route through the workstation: `ssh -J soham@172.24.58.65 argus@<jetson-ip>`. The Jetson's address changes (currently 172.26.11.80); run `hostname -I` on it to check.

## Known issues

1. **Pick the right checkpoint.** Nano v1 (`nano_train/checkpoints/`) scores 3% top-1; v2 scores 34.5%. There are ten full-size checkpoints, two from the latest training round (`checkpoints_faithful_lora_v2`, `checkpoints_faithful_dynamic_v2`); only `checkpoints_faithful_lora_v2` has been tested, and the repo's own OD code names it the current best model. A newer AstroLoc-tiny run, `nano_train/checkpoints_faithful_dynamic`, nearly matches full-size on the 200 photos and runs on the Jetson (0.70 s, 0.5 GB less memory; see Experiment log); full-size stays the default.
2. **LoRA checkpoints saved before commit `cc75c2b` (nano v2 among them) need their weight names remapped.** That commit renames LoRA weights (`blocks.N.` → `blocks.N.block.`). Our scripts remap the names on load, and so does the repo's own loader from `d92f5bc` on.
3. **TensorRT needs a static batch and opset 17.** Export with `--static-batch --batch 1 --opset 17`.
4. **FP16 rejected.** 2× faster but cosine 0.877 vs FP32 — enough to retrieve wrong tiles.
5. **Preprocessing is outside the engine** — resize to 224, ImageNet-normalize before calling it.
6. **Database folders are `{year}_{zoom}`** — `2021_09` is year 2021, zoom 9, not a month.
7. **The Jetson's PyTorch needs numpy ≥ 1.23**; the system has 1.21.5, so `pydeps` carries its own numpy.
8. **LightGlue's `__init__.py` imports every extractor**, and one needs torchvision. The copy in `pydeps` imports only `LightGlue` and `SIFT`.
9. **Image shape: square images.**
   - The team uses square images, so our square test photos (1024 × 1024) match the flight input.
   - The Argus camera sensor is 4608 × 2592.
   - The GeoTIFF evaluation (`/mnt/sdc1/astroloc/reference_db/argus_geotiff_eval`) supports the square choice. With a centre square crop, R@1 is 71.1% against 55.0% squashing the whole frame. With the matcher, the crop gives a fix on 68.8% of frames and a 5.4 km median error.
10. **Snow / winter imagery fails to match (paused, come back to later).** Northern GeoTIFF zones (~40–48° N) fix 0–7 of 13 frames against 12–13 in the south, because their Landsat mosaics include snow-covered and winter scenes while the database tiles look like summer. A real Argus photo over snow in winter would hit the same problem. Options when we return: a winter database (the other years 2018–2020 may differ in season; unchecked), or reporting north and south separately. For now we go with the current database.

## Next steps

1. **OD** — chain runs on real localization (1.4 km; tie points ~1.2 km). Next: OD with 2 photos per pass, combined over several passes (the planned operation).
2. **Default setup: top 10 tiles, search on the GPU, packed tile features, TensorRT LightGlue (1,024 points), stop at the first tile with ≥ 30 inliers** (0.44 s per photo on average). Remaining speed options: see *What would make it faster*.
3. **Reject wrong fixes** — 3 of 108 astronaut-photo fixes are more than 50 km off; none on Argus-like frames.
4. **Coordinate error for OD** — error models for OD weighting are in progress (`od/make_dataset.py` on master).
