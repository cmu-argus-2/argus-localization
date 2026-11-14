# Jetson Localization Runbook

Running AstroLoc localization on the Jetson Orin Nano: photo → retriever → search → matcher → location → OD.

## Status (current)

**The full localization flow runs on the Jetson** — TensorRT retriever, database search and the SIFT + LightGlue matcher in one process — and reproduces the workstation.

### Model tested (flight candidate)

| Part | What runs |
|---|---|
| **Retriever** | **AstroLoc full-size**, checkpoint `astroloc_train/checkpoints_faithful_lora_v2/final.pt`: DINOv2-base (LoRA-tuned) + SALAD + linear layer, **106.5M parameters**, 2,048-number fingerprint, 224 × 224 input. TensorRT FP32 engine `astroloc_full_fp32.trt` |
| **Database** | all 7,602 zoom-9 tiles from 2021 (whole Earth), 4 rotations each; fingerprints 249 MB |
| **Search** | on the GPU, top 10 tiles |
| **Matcher** | OpenCV SIFT (1,024 points at 512 px) + LightGlue (authors' SIFT weights) as a TensorRT FP16 engine `lightglue_sift_k1024_fp16.trt`; tile SIFT features precomputed and packed |
| **Fix rule** | MAGSAC homography (3 px); stop at the first tile with ≥ 30 inliers |
| **Output** | tie points (photo pixel → lat/lon) for OD, plus the photo footprint |

### Results with this model (Jetson, 15 W mode)

| | 200 astronaut photos (Alps) | 208 Argus-like frames (Landsat) |
|---|---|---|
| Right tile ranked first (retrieval only) | 68.5% | 56.7% |
| Fix rate | 54.0% (108) | 59.6% (124) |
| Error of a fix | 15.6 km median (footprint centre; the only truth available) | **0.7 km median, 1.0 km worst (tie points)** |
| Fixes > 50 km off | 3 | 0 |
| Time / energy per photo | 0.45 s mean / ~3.3 J | 0.44 s mean / ~3.3 J |

- **OD:** the Jetson's real tie points into FSW-Payload's OD on a simulated pass give **1.40 km** orbit error (see *Localization → OD*).
- **Cost:** ~7.6 W while localizing, 3.9 GB peak board memory (of 7.6 GB).
- **Time:** a photo with a fix takes 0.2–0.4 s; one with no fix checks all 10 tiles (~0.65 s, the worst case). 3.3 s per photo originally; each step is in the *Experiment log*.
- **Wrong fixes:** 3 of 108 astronaut-photo fixes are > 50 km off (none on Argus-like frames); filter before OD.
- **Backup model:** AstroLoc-tiny (`nano_train/checkpoints_faithful_dynamic`, 27M parameters, 512 numbers): 54.0% vs 55.0% fix, 0.70 s, 0.5 GB less memory (setup before the stop rule).

## Retrieval only: full-size vs nano v2

| Jetson Orin Nano 8GB, TensorRT FP32 | Nano v2 (27M, 512-d) | **Full-size (106.5M, 2048-d)** |
|---|---|---|
| Top-1 / Top-10 / Top-100 | 34.5% / 50.5% / 60.5% | **68.5% / 89.0% / 98.5%** |
| Median error | 1,042 km | **79 km** |
| On the 169 photos excluded from training | 33.7%, 1,303 km | **64.5%, 86 km** |
| Time per photo | ~67 ms | ~109 ms |
| Board memory, idle → peak (of 7,620 MB) | 2,345 → 2,686 MB | 2,523 → 3,351 MB |
| Board power, peak | 6.9 W | 8.2 W |
| Fingerprint database | 62 MB | 249 MB |

- Same top tile as the workstation on all 200 photos, for both models.
- Full-size: 2× the hit rate and 13× lower error than nano v2, so it is the flight candidate.
- Nano v2 is `nano_train/checkpoints_v2/final.pt`.

## Full flow on astronaut photos

200 astronaut photos near the Alps; "169 photos" = those training can't have seen. Full-size retriever throughout.

| | Original (3.3 s) | TensorRT LightGlue, all 10 tiles (0.65 s) | **Current: + stop at first good tile (0.45 s mean)** |
|---|---|---|---|
| Setup | top 15 tiles, full precision | top 10 tiles, LightGlue as a TensorRT FP16 engine | same, stop at the first tile with ≥ 30 inliers |
| Fix rate | 54.5% (109) | 54.0% (108) | 54.0% (108) |
| Median error | 15.9 km | 15.9 km | 15.6 km |
| 90% of fixes within | 33.2 km | 32.0 km | 32.5 km |
| Fixes > 50 km off | 4 | 3 | 3 |

- **Fix rate:** % of photos located confidently (≥ 30 matched points on a tile). The rest get "no fix": no answer rather than a guess.
- **Error:** km between the estimated and true photo centre (footprint centre), for fixes only. Astronaut photos have truth only at the corners, so they can't be graded at the tie points.
- **Matches the workstation:** on the original setup, the same fix / no-fix decision on 199 of 200 photos; where both fix, the locations differ by 0.02 km median (5.9 km max). Correct tile in the top 15 on 91% of photos on both machines; 92.7% of fixes on a correct tile.
- On the 169 photos excluded from training: 53.8% fix rate, 16.1 km median (original setup).

## Square Argus-like frames

208 simulated Argus frames: centre squares of the Landsat-8 GeoTIFFs (`/mnt/sda2/geotiffs`, 13 per zone across 16 zones), ~454 km across, resampled to 1024 × 1024. Built with `prepare_geotiff_set.py`; on the Jetson in `e2e_geotiff/`. Full-size model, same database.

| 208 frames, Jetson, TensorRT LightGlue (1,024 points) | All 10 tiles | **Stop at first tile with ≥ 30 inliers (current)** |
|---|---|---|
| Fix rate | 59.6% (124) | **59.6% (124)** |
| **Tie-point error: median / p90 / worst fix** | | **0.7 / 0.8 / 1.0 km** |
| Footprint-centre error, median / 90% within | 6.7 / 10.9 km | 6.5 / 10.0 km |
| Fixes > 50 km off | 0 | 0 |
| Tiles checked per frame | 10 | 5.1 |
| Time / energy per photo | 0.64 s / ~5 J | **0.44 s mean (0.34 s median) / ~3.3 J** |

- **Tie-point error is the real accuracy and what OD uses:** each tie point vs the true ground point of its pixel from the GeoTIFF geotransform (`grade_tie_points.py`). The footprint-centre error extrapolates the homography from the matched tile (~60 km) to the corners of the ~454 km frame, which inflates small errors.
- Near-nadir frames like Argus's localize better than astronaut photos: 6.7 km vs 16 km by the same centre measure, and no fix more than 50 km off.
- **Fixes depend on latitude/season:** southern zones (R, S; ~25–40° N) fix 12–13 of 13 in most zones; northern zones (T; ~40–48° N) fail often (16T 0, 17T 1, 32T 3, 54T 3, 10T 6, 33T 7 of 13). See Known issue 10.
- Correct tile among the 10 checked: 77.9%. Each frame is larger than a tile, so only a few tiles overlap it well.

## Localization → OD

`od/simulate_pass.py` runs a simulated Argus pass through FSW-Payload's Python batch OD (`python_od` in an FSW-Payload checkout, default `repos/FSW-Payload/python_od`, or set `FSW_PYTHON_OD`; CasADi/IPOPT; needs `casadi` and `scipy`). OD solves position, velocity, attitude and gyro bias from landmark bearings plus gyro.

1. **Truth:** circular orbit at 600 km (two-body + J2), 97.8° inclination, nadir-pointing camera, gyro at 1 Hz with bias and noise. `--poses-only` writes each frame's position and attitude.
2. **Frames:** `od/render_pass.py` (workstation) ray-casts every pixel of each 1024 × 1024 frame to the ground and samples the Landsat GeoTIFFs: true perspective views; frames more than 20% empty are skipped.
3. **Localization:** the Jetson runs the current setup with `--save-tie-points`.
4. **OD input:** `--tie-points-from` turns the tie points into `landmark_measurements.csv` (body-frame bearing, landmark ECI, per-point sigma) + `imu_data.csv`, FSW-Payload's own format.
5. **Output:** `od/runs/<name>/`: `od_summary.json`, `od_states.csv`, `truth.csv`, `od_pass.png` (ground track, OD error over time).

Pass: Libya (31.5° N, 13.5° E) north over Tunisia, Sicily, Sardinia/Corsica and northern Italy; a frame every 10 s for 330 s, 29 of 34 rendered (the last 5 leave the GeoTIFF coverage).

| Real pass, full-size model on the Jetson | Result |
|---|---|
| Frames with a fix | 11 of 29 (6 frames are mostly sea and have no correct tile) |
| Tie-point error (each point vs the true ground point of its pixel) | **~1.2 km median**, ≤ 2.2 km max on every fix; no wrong fixes |
| OD position error | **1.40 km median**, 2.9 km max, point sigma 0.002 rad (≈ 1.2 km, the measured tie-point error) |
| OD velocity error | 12 m/s median |
| Gyro bias | recovered to ~5% |
| Same pass, sigma from the footprint-centre error (0.011 rad) | 2.13 km, 20 m/s |

- **Fix rate is 38% here vs 60% on the 208 frames because of where the pass goes:** the southern half (Libya–Sicily, 32S/33S) fixes most frames; northern Italy and the Alps (32T/33T) fix 1 of 12 (snow/winter GeoTIFFs, Known issue 10).
- **Chain check:** with perfect localization, OD gives 0.15 km and 1.0 m/s. In simulation, a 25 min arc beats 10 min: over 10 min the velocity is weakly constrained.
- `od/check_tie_points.py` prints per-frame tie-point errors. Files: `od/runs/real_eu/`, `od/runs/real_eu_sigma002/`.
- **Simplifications:** Earth rotation is GMST only; body frame = camera frame, pointing at nadir (the plan; the real camera may end up partly sideways); focal length fixed at the config value (fx 3548; camera spec not final); no drag.
- **Not yet tested:** the planned operation, ~2 photos per pass combined over several passes.

### Real two-frame pairs (astronaut photos)

38 real same-pass photo pairs (12–147 s apart, times from NASA's photo pages); both photos localized on 8 of 38. These photos are oblique, so the main plan uses Argus-like nadir frames instead. Details: `two_frame/README.md`.

## Where the time goes — ⚠️ EARLIER SETUP (0.73 s), NOT CURRENT

> ⚠️ **Old run, kept for the breakdown.** PyTorch LightGlue (half precision), all 10 tiles: **0.73 s per photo**. The current setup is **0.44 s**: the TensorRT LightGlue engine takes ~34 ms per tile instead of ~58 ms, and the stop rule checks ~5 tiles instead of 10. The step shares below are from the old run.

50 photos, 15 W mode, nothing else running; per-step timing adds ~0.07 s. `profile_full.py --lightglue-mp --tile-pack ../tile_feats_pack --search-on-gpu`.

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

- **LightGlue is the bottleneck.** Everything before it, including the retriever, takes ~0.21 s. Next is photo SIFT (85 ms on the CPU; OpenCV's SIFT has no GPU version here).
- **Inside PyTorch LightGlue, the ARM CPU was the limit:** the thread launching ~850 small GPU kernels per tile was busy 99% of the time, the GPU 53%. This is why the TensorRT engine helps.
- **Disk is not a bottleneck.** Clocks in 15 W mode: CPU up to 1.51 GHz, GPU up to 625 MHz.

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
1. **Higher power mode (MAXN SUPER, JetPack 6.2):** GPU 625 → 1,020 MHz. A LightGlue of the same size runs at 21.6 ms per pair on an Orin Nano in this mode (kornia/lightglue on Hugging Face). Needs a reflash, so it's a team decision.
2. **Tracking mode:** after the first fix, use OD's predicted position, skip the retriever and match only the 1–3 tiles near it.
3. **Photo SIFT (CPU) in parallel with the retriever (GPU):** up to ~50 ms.
4. **SIFT on the GPU (CudaSift):** ~80 ms, but all tile features must be recomputed with it.
5. **LightGlue's own early exit and point pruning:** off in the fixed-size TensorRT engine; hard to add.
6. **XFeat + LighterGlue:** ~8 ms per pair, public weights, but poor beyond ~45° of rotation.

## Experiment log

One change at a time against a same-day baseline, full-size model, on the 200 astronaut photos (and the 208 Argus-like frames where noted). **Bold** rows were adopted, in order; together they take 3.3 s → 0.44 s per photo. Per-run results are on the Jetson in `e2e_alps_full/` and `e2e_geotiff/` (`match_results_<tag>.json`, `match_run_<tag>.log`).

| Change | Time per photo | Accuracy | Energy | Decision |
|---|---|---|---|---|
| **Store tile SIFT features (3.2 GB)** | 3.3 → 1.3 s | identical | 26 → 14 J | adopted |
| **Top 10 tiles instead of 15** | 1.31 → 0.95 s | identical | 14 → 10 J | adopted |
| Top 5 tiles | 0.59 s | −2 of 109 fixes | 6 J | no |
| **LightGlue half precision** | 0.95 → 0.81 s | +1 fix | 10 → 7 J | adopted |
| 512 SIFT points (alone or + half precision) | ~0.8 s | −19 fixes | 6–7 J | no |
| **Search on GPU + tile features in one packed file** | 0.82 → 0.73 s | identical | 7 → 6 J | adopted |
| Tile features in GPU memory | 0.74 s | identical, +1.8 GB | 6 J | no |
| Batched LightGlue, PyTorch | 0.83 s | −1 fix | 8 J | no |
| **LightGlue TensorRT FP16, 1,024 points** | 0.74 → 0.65 s | −2 fixes; Argus-like +5 (124 vs 119) | 6 → 5 J | adopted |
| LightGlue TensorRT FP32 | 1.18 s | +2 fixes | 13 J | no: slower |
| Tiny retriever (27M, 512-d) | 0.70 s | 54.0% vs 55.0% fix; −0.5 GB | 6 J | backup |
| 10 tiles per TensorRT call | −6% engine time | — | — | no |
| 768 points, TensorRT | 0.56 s | Argus-like 89 vs 124 fixes | 4 J | no |
| Tiny + 768 points | 0.55 s | −6 fixes | 4 J | no |
| LightGlue 7 of 9 layers (FastGlue idea) | 0.57 s | Argus-like −6 of 124 | 5 J | no (option if speed matters more) |
| LightGlue 6 or 5 layers | — | Argus-like −6 to −8 | — | no |
| **Stop at first tile with ≥ 30 inliers** | 0.65 → 0.44 s mean | same fixes on both sets | 5 → 3.3 J | adopted |

The workstation (RTX 4090) runs the flow in 0.22 s with stored features.

## What runs on the Jetson

One Python process (`e2e_match.py`). The ARM CPU (6 cores) and the GPU share memory.

| Step | What happens | Runs on |
|---|---|---|
| 1. Retriever | photo (224 × 224) → 2048-number fingerprint, `astroloc_full_fp32.trt` (full-size, `faithful_lora_v2`) | GPU |
| 2. Search | against 7,602 tiles × 4 rotations, keep the top 10 | GPU |
| 3. Feature points | photo SIFT, 1,024 points at 512 px; tile points read from `tile_feats_pack` | CPU |
| 4. Matching | LightGlue TensorRT FP16 engine, photo vs each tile in retrieval order | GPU |
| 5. Geometry check | MAGSAC homography (3 px), count inliers; stop at the first tile with ≥ 30 | CPU |
| 6. Output | the tile's inlier points as tie points (photo pixel → lat/lon) and the photo's footprint | CPU |

- **Repo code used unmodified** (argus-localization `cc75c2b`): `SiftLightGlueMatcher`, `Georeferencer`, `core/types.py`, `data_loading/earthloc_loader.py`. Ours: `e2e_match.py`, `lightglue_trt.py`, the tile-feature packing and the TensorRT conversions.
- **Differences from `core/pipeline.py`:** TensorRT retriever (cosine 0.99997 vs PyTorch); photo SIFT extracted once per photo; stop rule instead of always checking every tile.
- **Dependencies:** JetPack 6.0 (L4T R36.3, CUDA 12.2, TensorRT 8.6.2, OpenCV 4.8.0) on the board; PyTorch 2.4.0a0 (NVIDIA wheel), numpy 1.26.4, kornia 0.7.1, LightGlue `eb42fee` and PyTorch's small deps in `~/Desktop/astroloc/pydeps` (`pip3 install --no-deps --target`, used via `PYTHONPATH`; nothing system-wide, no sudo).
- **Disk:** engines ~0.5 GB, fingerprints 249 MB, `pydeps` 2.0 GB, packed tile features 2.15 GB, tile images 720 MB (their filenames carry the tile corners).
- **Handoff to OD:** `--save-tie-points` writes each fix's tie points; `od/simulate_pass.py` turns them into FSW-Payload's input files.

## Files

**Jetson** (`~/Desktop/astroloc/`):
- Engines: `astroloc_full_fp32.trt` (full-size retriever), `lightglue_sift_k1024_fp16.trt`, `nano_v2_fp32.trt`.
- Data: `tiles/2021_09/` (tile images), `tile_feats_pack/` (packed tile SIFT features, used), `tile_feats_2021_09/` (per-tile features, input to the pack).
- Test sets and results: `e2e_alps_full/` (200 astronaut photos + database), `e2e_geotiff/` (208 Argus-like frames), `e2e_pass_eu/` (real OD pass), `e2e_alps_v2/` (nano v2).
- Code: `e2e_match.py`, `run_match_full.sh`, `run_stop.sh`, `e2e_jetson.py`, `precompute_tile_feats.py`, `pack_tile_feats.py`, `lightglue_trt.py`, `profile_full.py`, `repo/` (matcher and georeferencing code from argus-localization), `pydeps/` (PyTorch, kornia, LightGlue), `wheels/` (installers).

**Workstation** (`~/`): `astroloc_full_static.onnx`, `nano_v2_static.onnx`, `e2e_alps_full/`, `e2e_alps_v2/` (database + test photos + workstation results). Stored tile features were deleted because the home disk is 97% full; `precompute_tile_feats.py` rebuilds them in 6 min.

**Superseded:** `nano_fp32.trt` / `nano_static.onnx` (nano **v1**, 3% top-1), `nano_fp16.trt` (FP16 rejected), `astroloc_fp32.trt` / `astroloc_static.onnx` (oldest full-size checkpoint).

## Reproduce

Commands are for the full-size model. For nano v2, use `nano_train/checkpoints_v2/final.pt` and the `nano_v2` / `e2e_alps_v2` names.

### 1. Run the current setup (Jetson)
```bash
cd ~/Desktop/astroloc
./run_match_full.sh mine --top-k 10 --tile-pack ../tile_feats_pack --search-on-gpu \
    --lightglue-trt ../lightglue_sift_k1024_fp16.trt --stop-inliers 30 --save-tie-points
# → e2e_alps_full/match_results_mine.json, match_run_mine.log, tegrastats_match_mine.log
```
`D=../e2e_geotiff ./run_match_full.sh …` runs the Argus-like frames. `run_match_full.sh` uses the full-size retriever engine and logs `tegrastats`.

### 2. Workstation: database, ONNX, workstation results
```bash
cd ~/argus-localization
CKPT=/mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora_v2/final.pt
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_prepare.py --checkpoint $CKPT --out ~/e2e_alps_full
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/export_onnx.py --checkpoint $CKPT \
    --output ~/astroloc_full_static.onnx --opset 17 --batch 1 --static-batch
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_match.py \
    --data ~/e2e_alps_full --tiles-dir /mnt/sdc1/astroloc/data/database/2021_09
python deploy/jetson/lightglue_trt.py export --out ~/lightglue_sift_k1024.onnx
```
Check `nvidia-smi` first and use a free GPU.

### 3. Jetson: engines and retrieval-only test
```bash
cd ~/Desktop/astroloc
scp soham@172.24.58.65:~/astroloc_full_static.onnx soham@172.24.58.65:~/lightglue_sift_k1024.onnx .
scp -r soham@172.24.58.65:~/e2e_alps_full .
/usr/src/tensorrt/bin/trtexec --onnx=astroloc_full_static.onnx --saveEngine=astroloc_full_fp32.trt
/usr/src/tensorrt/bin/trtexec --onnx=lightglue_sift_k1024.onnx --saveEngine=lightglue_sift_k1024_fp16.trt --fp16
python3 e2e_jetson.py --engine astroloc_full_fp32.trt --data e2e_alps_full
```
Run `tegrastats` alongside for board memory and power.

### 4. Jetson: matcher setup (once)

The Jetson has no internet, so everything is downloaded elsewhere and copied into `~/Desktop/astroloc/`:
- `wheels/`: NVIDIA's PyTorch wheel for JetPack 6.0 (`developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl`), numpy 1.26.4 for aarch64 / Python 3.10, and PyTorch's pure-Python dependencies.
- `pydeps/kornia`, `pydeps/lightglue`: copied from the workstation's `~/venv-export/lib/python3.12/site-packages/`.
- `~/.cache/torch/hub/checkpoints/sift_lightglue_v0-1_arxiv.pth`: copied from the same path on the workstation.
- `tiles/2021_09/`: copied from `/mnt/sdc1/astroloc/data/database/2021_09`.
- `repo/`: the argus-localization files listed under *What runs on the Jetson*, plus the empty `__init__.py` files.
```bash
cd ~/Desktop/astroloc
pip3 install --no-deps --no-index --target pydeps wheels/*.whl
printf 'from .lightglue import LightGlue  # noqa\nfrom .sift import SIFT  # noqa\n' > pydeps/lightglue/__init__.py
cd repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../precompute_tile_feats.py \
    --tiles-dir ../tiles/2021_09 --out ../tile_feats_2021_09          # 16 min
cd .. && python3 pack_tile_feats.py --data e2e_alps_full --feats tile_feats_2021_09 --out tile_feats_pack
```

### 5. Compare runs
```bash
cd ~/Desktop/astroloc
python3 compare_topk.py --data e2e_alps_full --tags <tag1> <tag2>   # or --ks 15 10 5
./run_stop.sh                                                        # all 10 tiles vs --stop-inliers 30, both test sets
```

### 6. Stage-by-stage pictures (`walkthrough/PIPELINE-WALKTHROUGH.md`)
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
| `e2e_match.py` | both | the full flow. `--engine` TensorRT retriever, `--tile-pack` packed tile features, `--search-on-gpu`, `--lightglue-trt` TensorRT LightGlue, `--stop-inliers N` stop at the first tile with ≥ N inliers, `--log-tiles` every tile's inliers, `--save-tie-points` each fix's (u, v, lat, lon). Every fix logs `tie_error_km` (tie points vs truth from the photo's corners, the error OD sees) and `match_error_km` (footprint centre) |
| `run_match_full.sh` | Jetson | runs `e2e_match.py` with the full-size engine and `tegrastats` logging; the first argument tags the output files |
| `run_stop.sh` | Jetson | timed runs, all 10 tiles vs `--stop-inliers 30`, on both test sets |
| `e2e_prepare.py` | workstation | builds the zoom-9 database, picks test photos, workstation results |
| `e2e_jetson.py` | Jetson | retrieval-only test with the TensorRT engine, compares to workstation |
| `export_onnx.py` | workstation | PyTorch → ONNX, with a parity check |
| `lightglue_trt.py` | workstation, Jetson | LightGlue as a fixed-size network: export to ONNX (`export --points --batch --layers`), check against PyTorch (`check`) |
| `precompute_tile_feats.py` | both | SIFT features of every tile, once |
| `pack_tile_feats.py` | Jetson | packs the 7,602 tile feature files into one set of arrays (`--tile-pack`) |
| `prepare_geotiff_set.py` | workstation | square Argus-like test frames from the Landsat GeoTIFFs |
| `grade_tie_points.py` | workstation | per-tie-point error on the Argus-like frames (truth from each GeoTIFF's geotransform) |
| `od/simulate_pass.py` | Mac | simulated Argus pass → tie points (simulated, or real with `--tie-points-from`) → FSW-Payload Python OD, compared with truth; `--poses-only` writes frame poses |
| `od/render_pass.py` | workstation | renders a pass's frames from the Landsat GeoTIFFs as a test set |
| `od/check_tie_points.py` | Mac | per-tie-point error of a rendered pass against the true poses |
| `compare_topk.py` | Jetson | compares runs side by side (`--ks` or `--tags`) |
| `simulate_stop.py` | any | replays stop-rule thresholds on a `--log-tiles` result |
| `profile_full.py` | Jetson | time, CPU, GPU, memory and disk cost of every step (run with `tegrastats` for power) |
| `trt_lg_bench.py`, `lg_bench.py` | Jetson | time the TensorRT LightGlue engine; one PyTorch LightGlue pair for `nsys profile -t cuda,nvtx` |
| `pipeline_trace.py`, `visualize_trace.py` | Jetson, Mac | record and draw every stage for one success and one failure; see `walkthrough/PIPELINE-WALKTHROUGH.md` |
| `check_engine.py`, `test_images.py` | Jetson | engine sanity, real-image check |
| `archive/` | — | superseded experiment scripts (`profile_match.py`, `run_speed.sh`, `run_speed2.sh`, `compare_precision.py`, `test_batched_lightglue.py`); see `archive/README.md` |

## Environment

- **Workstation:** `~/venv-export` (inherits `sat_env_vision`'s CUDA torch, plus onnx, onnxruntime, onnxscript, shapely, `numpy<2.2`, kornia 0.7.1, LightGlue). No sudo.
- **Jetson:** versions are listed under *What runs on the Jetson*. `trtexec` is at `/usr/src/tensorrt/bin/trtexec`. pycuda 2022.2.2 is used by `e2e_jetson.py`. 838 GB free disk. No internet, so packages are downloaded elsewhere and copied over.
- **Mac → Jetson:** VS Code needs Local Network permission (System Settings → Privacy & Security). Or route through the workstation: `ssh -J soham@172.24.58.65 argus@<jetson-ip>`. The Jetson's address changes (currently 172.26.11.80); run `hostname -I` on it to check.

## Known issues

1. **Pick the right checkpoint.** Tested: full-size `checkpoints_faithful_lora_v2` (the default; the repo's own OD code names it the current best model), nano v2 `nano_train/checkpoints_v2` (34.5% top-1), AstroLoc-tiny `nano_train/checkpoints_faithful_dynamic` (backup). Nano v1 (`nano_train/checkpoints/`) scores 3% top-1. The other full-size checkpoints, including `checkpoints_faithful_dynamic_v2` from the latest round, are untested.
2. **LoRA checkpoints saved before commit `cc75c2b` (nano v2 among them) need their weight names remapped.** That commit renames LoRA weights (`blocks.N.` → `blocks.N.block.`). Our scripts remap the names on load, and so does the repo's own loader from `d92f5bc` on.
3. **TensorRT needs a static batch and opset 17.** Export with `--static-batch --batch 1 --opset 17`.
4. **FP16 retriever rejected.** 2× faster but cosine 0.877 vs FP32 — enough to retrieve wrong tiles. (LightGlue in FP16 is fine.)
5. **Preprocessing is outside the engine** — resize to 224, ImageNet-normalize before calling it.
6. **Database folders are `{year}_{zoom}`** — `2021_09` is year 2021, zoom 9, not a month.
7. **The Jetson's PyTorch needs numpy ≥ 1.23**; the system has 1.21.5, so `pydeps` carries its own numpy.
8. **LightGlue's `__init__.py` imports every extractor**, and one needs torchvision. The copy in `pydeps` imports only `LightGlue` and `SIFT`.
9. **Square images.** The team uses square images, so the test frames (1024 × 1024) match the flight input; the Argus sensor is 4608 × 2592. A separate retrieval-only evaluation (`/mnt/sdc1/astroloc/reference_db/argus_geotiff_eval`) supports a centre square crop: R@1 71.1% against 55.0% squashing the whole frame.
10. **Snow / winter imagery fails to match (paused, come back to later).** Northern GeoTIFF zones (~40–48° N) fix 0–7 of 13 frames against 12–13 in the south, because their Landsat mosaics include snow-covered and winter scenes while the database tiles look like summer. A real Argus photo over snow in winter would hit the same problem. Options: a winter database (the other years 2018–2020 may differ in season; unchecked), or reporting north and south separately. For now we go with the current database.

## Next steps

1. **OD as planned:** ~2 photos per pass, combined over several passes.
2. **Reject wrong fixes:** 3 of 108 astronaut-photo fixes are more than 50 km off; none on Argus-like frames.
3. **Coordinate error for OD:** error models for OD weighting are in progress (`od/make_dataset.py` on master).
4. **Speed, if needed:** see *What would make it faster*.
