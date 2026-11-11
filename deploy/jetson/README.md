# Argus Localization on the Jetson

Everything needed to run, measure and extend AstroLoc localization on the Jetson Orin Nano: the scripts, the commands, where the data lives, and the working conventions. For agents, `CLAUDE.md` holds the rules; this file holds the how.

- **`RUNBOOK.md`** — every result so far, with dates, and the reasoning behind each design choice.
- **`overview/START-HERE.md`** — the whole system on one page: each part, which are neural networks, which are off the shelf.
- **`walkthrough/PIPELINE-WALKTHROUGH.pdf`** — one photo through every stage, with pictures. Start there if the pipeline is new to you.

---

## 1. What it does

A photo of Earth goes in; a location comes out, with no GPS.

```
photo → resize 224 → retriever (TensorRT) → search 7,602 map tiles → top 10 tiles
      → SIFT points → LightGlue (TensorRT) + MAGSAC, tile by tile → stop at the first tile with ≥ 30 inliers (fix)
      → tie points (pixel ↔ lat/lon) + photo footprint → orbit determination (OD)
```

- **Retriever:** AstroLoc full-size, checkpoint `faithful_lora_v2` (DINOv2-base + SALAD, 106.5M parameters, 2,048-number fingerprint). It runs as a TensorRT FP32 engine.
- **Database:** every 2021 zoom-9 map tile (7,602, whole Earth). Each tile is stored at 4 rotations.
- **Matcher:** the `argus-localization` repo's own `SiftLightGlueMatcher` and `Georeferencer`, unmodified.

**Current setup and results**

The setup is: top 10 tiles, search on the GPU, tile features precomputed and packed, LightGlue as a TensorRT FP16 engine (1,024 points), and stop at the first tile with ≥ 30 inliers. Jetson in 15 W mode:

| | 200 astronaut photos | 208 Argus-like frames (Landsat) |
|---|---|---|
| Time per photo (mean) | **0.45 s** | **0.44 s** |
| Fix rate | 54% | 60% |
| Error of a fix | 15.6 km median (footprint centre) | **0.7 km median at the tie points** (6.5 km by footprint centre) |
| Fixes more than 50 km off | 3 of 108 | 0 of 124 |
| Energy / power | ~3.3 J per photo, ~7.6 W while running | same |
| Peak board memory | ~3.9 GB of 7.6 GB | same |

- **Tie-point error** is the error OD sees: each matched point vs the true ground point of its pixel. The footprint-centre error extrapolates from the matched ~60 km tile to the corners of the whole photo and reads much higher.
- **Localization → OD:** on a simulated pass with frames rendered from the Landsat GeoTIFFs, real Jetson tie points into FSW-Payload's Python OD give a **1.4 km** median orbit error (`RUNBOOK.md` → Localization → OD).
- Northern (snow / winter) GeoTIFF frames fix far less often than southern ones; see §7.

---

## 2. What's in this folder

| File | Runs on | What it does |
|---|---|---|
| `export_onnx.py` | workstation | PyTorch checkpoint → ONNX, with a check that the ONNX output matches PyTorch |
| `e2e_prepare.py` | workstation | builds the zoom-9 fingerprint database and a 200-photo test set with ground truth |
| `e2e_jetson.py` | Jetson | retrieval-only test with the TensorRT engine, compared with the workstation |
| `e2e_match.py` | both | **the full flow** (retriever + search + matcher). All options in §6 |
| `lightglue_trt.py` | workstation, Jetson | LightGlue rebuilt as a fixed-size network: ONNX export and check, and the TensorRT runtime used by `--lightglue-trt` |
| `trt_lg_bench.py` | Jetson | times the TensorRT LightGlue engine |
| `run_stop.sh`, `simulate_stop.py` | Jetson, any | timed stop-rule runs, and replaying stop thresholds offline from a `--log-tiles` run |
| `prepare_geotiff_set.py` | workstation | builds the 208 Argus-like frames from the Landsat GeoTIFFs |
| `grade_tie_points.py` | workstation | exact tie-point error on those frames (truth from each GeoTIFF's geotransform) |
| `od/simulate_pass.py` | Mac / any | simulated Argus pass → simulated or real tie points → FSW-Payload Python OD, compared with truth |
| `od/render_pass.py` | workstation | renders a pass's frames from the GeoTIFFs (true perspective views) as a test set |
| `od/check_tie_points.py` | Mac / any | exact tie-point error on a rendered pass |
| `od/runs/` | — | results of the OD runs in `RUNBOOK.md` |
| `overview/` | — | `START-HERE.md` (one-page overview of the system) and the AstroLoc paper summary |
| `precompute_tile_feats.py` | both | computes SIFT features for every tile once (one `.npz` per tile) |
| `pack_tile_feats.py` | Jetson | packs those 7,602 files into one memory-mapped set (`--tile-pack`) |
| `run_match_full.sh` | Jetson | runs `e2e_match.py` with `tegrastats` logging; first argument = run tag |
| `run_speed.sh`, `run_speed2.sh` | Jetson | the comparison batches recorded in `RUNBOOK.md` |
| `compare_topk.py` | Jetson | puts runs side by side: accuracy, time, memory, power |
| `profile_full.py` | Jetson | time, CPU, GPU, memory and disk cost of every step |
| `lg_bench.py` | Jetson | one LightGlue pair repeated, for `nsys` |
| `profile_match.py` | Jetson | older per-stage timing of the matcher (before tile features were stored) |
| `test_batched_lightglue.py` | either | checks batched LightGlue against one tile at a time |
| `pipeline_trace.py`, `visualize_trace.py` | Jetson, Mac | record and draw every stage for one photo (`walkthrough/`) |
| `check_engine.py`, `compare_precision.py`, `test_images.py` | Jetson | engine sanity check, FP32 vs FP16, real-image check |
| `two_frame/*.py` | workstation, Mac, Jetson | finding real same-pass photo pairs for OD (§5.8) |
| `config.sh`, `sync.sh` | Mac | your settings, and copying scripts to the workstation or Jetson |
| `jetson_folder_README.md` | — | copy of the label file inside the Jetson's shared folder |

---

## 3. Machines and access

**Lab workstation** (`172.24.58.65`, 2× RTX 4090, Ubuntu)
- You need the CMU VPN when off campus: `ssh <you>@172.24.58.65`.
- **Rules:** no personal data, no downloaded executables (pip into your own env is fine), **never reboot**, **no sudo**, and never interrupt other people's jobs.
- Check `nvidia-smi` and use a free GPU with `CUDA_VISIBLE_DEVICES` (lately GPU 0 has someone else's job, so use 1).
- The home disk is ~97% full, so delete large scratch files when you're done.

**Jetson Orin Nano 8GB** (shared account `argus`, JetPack 6.0, TensorRT 8.6.2, 15 W mode)
- Reach it through the workstation: `ssh -J <you>@172.24.58.65 argus@172.26.11.80`.
- On campus you can also SSH to it directly.
- **Its address changes** when the network changes. If SSH times out, someone at the lab runs `hostname -I` on it. Its second address (`172.27.x.x`) is blocked for VPN users by CMU's firewall.
- **No internet.** Download files elsewhere and stream them over SSH (§5.10).
- No sudo. Install nothing system-wide. Use the private Python folder (§4).
- VS Code on a Mac may need Local Network permission to reach it (System Settings → Privacy & Security); Terminal works.

### 3.1 Let your agent connect (one-time, per person)

An agent can only run commands over SSH if no password is needed. Set up a key once. **You** type the passwords; never give them to the agent.

```bash
# on your Mac
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519               # make a key if you don't have one
ssh-copy-id <you>@172.24.58.65                                  # asks your workstation password once
ssh-copy-id -o ProxyJump=<you>@172.24.58.65 argus@172.26.11.80 # asks the argus (Jetson) password once
```

Add short names to `~/.ssh/config`:
```
Host argus-ws
  HostName 172.24.58.65
  User <you>
Host argus-jetson
  HostName 172.26.11.80
  User argus
  ProxyJump argus-ws
```

Check that both work without a prompt (connect to the VPN first if you're off campus):
```bash
ssh -o BatchMode=yes argus-ws hostname && ssh -o BatchMode=yes argus-jetson hostname
```

How the agent runs things:
- **One command:** `ssh -o BatchMode=yes argus-jetson 'cd ~/Desktop/astroloc && ls'`
- **A multi-line script:** write it to a local file, then `ssh argus-jetson 'bash -s' < script.sh`.
- **A long run:** start it detached so it survives the connection, and have it write a "done" file; check for that file later.
```bash
ssh argus-jetson 'cd ~/Desktop/astroloc && nohup setsid ./my_run.sh > /dev/null 2>&1 &'
```
- **Going through the workstation** (`ProxyJump`) also avoids the macOS Local Network block on the Jetson's network, so it works from VS Code too.
- **If the Jetson stops answering:** it may have restarted or got a new address. Check it at the lab with `hostname -I`, then update `HostName` and `config.sh`.

### 3.2 What's already on the Jetson (shared by everyone)

There is one shared account (`argus`) and one working folder, `~/Desktop/astroloc/`, with no per-person copies. The folder has its own `README.md` with this list (a copy is `jetson_folder_README.md` here), and an empty `users/` folder for personal work. **Treat everything listed here as shared.** Use it, don't move or delete it, and put your own work where the conventions below say.

| In `~/Desktop/astroloc/` | What it is | Size |
|---|---|---|
| `astroloc_full_fp32.trt` | **current retriever engine** (full-size `faithful_lora_v2`, FP32) | 410 MB |
| `lightglue_sift_k1024_fp16.trt` | **current matcher engine** (LightGlue, SIFT weights, 1,024 points, FP16) | — |
| `astroloc_full_static.onnx` | its ONNX source | 410 MB |
| `nano_v2_fp32.trt`, `nano_v2_static.onnx` | small model (nano v2) engine and ONNX | 110 MB each |
| `tiles/2021_09/` | the 7,602 zoom-9 tile images (their filenames carry the corner coordinates) | 720 MB |
| `tile_feats_2021_09/` | SIFT features per tile, one `.npz` each | 3.0 GB |
| `tile_feats_pack/` | **the same features packed into one set (current)** | 2.2 GB |
| `e2e_alps_full/` | **the 200-photo test set + full-size fingerprint database**, plus every run's results, logs and `tegrastats` files (`*_<tag>.*`) | 263 MB |
| `e2e_alps_v2/` | the same test set built with nano v2 | 81 MB |
| `e2e_pairs/` | the 38 same-pass photo pairs (its database files link to `e2e_alps_full/`) | 9 MB |
| `e2e_geotiff/` | the 208 Argus-like frames, plus their runs' results | ~15 MB |
| `e2e_pass_eu/` | 29 frames rendered along a simulated pass (Libya → Alps), for OD | 12 MB |
| `trace/` | recorded stages for the walkthrough figures | 4 MB |
| `pydeps/`, `wheels/` | private Python packages (PyTorch, LightGlue, …) and their installers | 2.0 GB, 1.0 GB |
| `repo/` | the 9 argus-localization files the matcher needs | — |
| `*.py`, `run_*.sh` | **copies of this package's scripts** (the shared versions) | — |
| `samples/` | a few test images used by `test_images.py` | 3 MB |
| older, superseded: `nano_fp32.trt`, `nano_static.onnx`, `nano_fp16.trt` (nano v1), `astroloc_fp32.trt`, `astroloc_static.onnx` (oldest full-size checkpoint), `e2e_alps/`, `*_output.npy` | kept for reference; not used by anything current | — |
| from a teammate's earlier retrieval runs: `jetson_run.sh`, `dump_vecs.py`, `run2_v1/`, `run2_v2/` | not part of this package; leave them alone | — |

Elsewhere on the Jetson:
- `~/.cache/torch/hub/checkpoints/sift_lightglue_v0-1_arxiv.pth` holds the LightGlue weights.
- The disk has ~836 GB free.

**Working on the shared Jetson:**
- **Outputs:** use your own run tag or output name (e.g. `run_match_full.sh sg_test …`, `--out match_results_sg_test.json`), so nobody's files get overwritten.
- **Your own experiments:** put new scripts or data in `~/Desktop/astroloc/users/<your-name>/` and run them from there. Scripts there can import `e2e_match.py`'s helpers if you copy it alongside.
- **Shared scripts:** update the top-level scripts only from this package (`./sync.sh jetson`), and tell the team when you do, because everyone runs those copies.
- **Before timing anything,** check that nobody else is running: `pgrep -af "python3|trtexec"` and a few seconds of `tegrastats`. Two jobs at once distort both results.
- **Deleting:** only remove files you created. The superseded files above can be removed only after the team agrees.

---

## 4. Environments

**Workstation, one-time setup in your own home**
```bash
git clone https://github.com/cmu-argus-2/argus-localization.git ~/argus-localization
git clone https://github.com/serizba/salad.git ~/argus-localization/third_party/salad
git -C ~/argus-localization/third_party/salad checkout 6aede13
# Python env: inherits the team's CUDA PyTorch from the conda env, adds our packages
/home/argus/miniconda3/envs/sat_env_vision/bin/python -m venv --system-site-packages ~/venv-export
~/venv-export/bin/pip install onnx onnxruntime onnxscript shapely "numpy<2.2" kornia==0.7.1
~/venv-export/bin/pip install --no-deps git+https://github.com/cvg/LightGlue.git
git -C ~/argus-localization checkout jetson-deployment   # this folder is deploy/jetson/ on that branch
```
- Run everything with `~/venv-export/bin/python`, from `~/argus-localization`. Plain `python3` has no PyTorch.
- Scripts were tested at repo commit `cc75c2b`. The matcher and georeferencing files are unchanged up to `476ee54`.
- LightGlue downloads its SIFT weights on first use, to `~/.cache/torch/hub/checkpoints/`.

**Jetson, already installed under the `argus` account**

| | |
|---|---|
| System | Python 3.10, TensorRT 8.6.2, pycuda 2022.2.2, numpy 1.21.5, OpenCV 4.8, `trtexec` at `/usr/src/tensorrt/bin/trtexec` |
| `~/Desktop/astroloc/pydeps/` | PyTorch 2.4.0 (NVIDIA's JetPack 6.0 wheel), numpy 1.26.4, kornia 0.7.1, LightGlue, and PyTorch's small dependencies. Installed with `pip3 install --no-deps --target`; nothing system-wide changed |
| Use it with | `PYTHONPATH=~/Desktop/astroloc/pydeps python3 ...` (needed for anything that uses PyTorch or LightGlue) |
| LightGlue weights | `~/.cache/torch/hub/checkpoints/sift_lightglue_v0-1_arxiv.pth` |
| `pydeps/lightglue/__init__.py` | trimmed to import only `LightGlue` and `SIFT`, because the full version needs torchvision |
| `~/Desktop/astroloc/repo/` | the 9 repo files the matcher needs (matchers, georeference, core types/interfaces, the tile loader) |

The installer wheels are kept in `~/Desktop/astroloc/wheels/`. To rebuild from scratch, see `RUNBOOK.md` → Reproduce → step 3.

---

## 5. Workflows

**Where things live.** Paths on the workstation are under `/mnt/sdc1/astroloc/`; paths on the Jetson are under `~/Desktop/astroloc/`.

| What | Workstation | Jetson |
|---|---|---|
| Full-size checkpoint (current) | `reference_db/astroloc_train/checkpoints_faithful_lora_v2/final.pt` | — |
| Nano v2 checkpoint | `reference_db/nano_train/checkpoints_v2/final.pt` | — |
| Map tiles, 2021 zoom 9 | `data/database/2021_09/` (folders are `{year}_{zoom}`) | `tiles/2021_09/` (720 MB) |
| Astronaut photos (17,763) | `data/queries/` | — |
| Simulated Argus frames | `/mnt/sda2/geotiffs/` (4609 × 2593 Landsat) | — |
| GeoTIFF evaluation results | `reference_db/argus_geotiff_eval/` | — |
| TensorRT engines | — | `astroloc_full_fp32.trt`, `nano_v2_fp32.trt` |
| 200-photo test set | `/home/soham/e2e_alps_full/` (readable only to its Unix group; otherwise rebuild with §5.2) | `e2e_alps_full/` |
| Tile features | — | `tile_feats_2021_09/` (per tile), `tile_feats_pack/` (packed) |

### 5.1 Convert a checkpoint to a TensorRT engine
```bash
# workstation, from ~/argus-localization
CKPT=/mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora_v2/final.pt
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/export_onnx.py --checkpoint $CKPT \
    --output ~/astroloc_full_static.onnx --opset 17 --batch 1 --static-batch
# Jetson, from ~/Desktop/astroloc (copy the .onnx over first, §5.10)
/usr/src/tensorrt/bin/trtexec --onnx=astroloc_full_static.onnx --saveEngine=astroloc_full_fp32.trt
```
- `--opset 17`: TensorRT 8.6 reads no higher.
- `--static-batch`: SALAD's Sinkhorn layer needs a fixed batch size.
- Keep FP32; FP16 was tested and degrades fingerprints.
- Without a GPU, add `--device cpu`.

### 5.2 Build a fingerprint database and test set
```bash
# workstation
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_prepare.py --checkpoint $CKPT --out ~/e2e_alps_full
```
This writes `db_vecs.npy` (7,602 × 4 × 2048), `db_tiles.json`, `queries/`, `queries.json` (true centre and correct tiles), and the workstation's own fingerprints and results. Copy the folder to the Jetson.

### 5.3 Retrieval only, on the Jetson
```bash
cd ~/Desktop/astroloc
python3 e2e_jetson.py --engine astroloc_full_fp32.trt --data e2e_alps_full
```

### 5.4 Tile features (once per tile set)
```bash
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../precompute_tile_feats.py --tiles-dir ../tiles/2021_09 --out ../tile_feats_2021_09   # ~16 min
cd .. && python3 pack_tile_feats.py --data e2e_alps_full --feats tile_feats_2021_09 --out tile_feats_pack                         # ~1 min
```

### 5.5 Full flow (the current setup)
```bash
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../e2e_match.py --data ../e2e_alps_full \
    --tiles-dir ../tiles/2021_09 --engine ../astroloc_full_fp32.trt \
    --top-k 10 --tile-pack ../tile_feats_pack --search-on-gpu \
    --lightglue-trt ../lightglue_sift_k1024_fp16.trt --stop-inliers 30 --out match_results_mine.json
```
- To log power and memory alongside: `cd ~/Desktop/astroloc && ./run_match_full.sh mine --top-k 10 --tile-pack ../tile_feats_pack --search-on-gpu --lightglue-trt ../lightglue_sift_k1024_fp16.trt --stop-inliers 30`. Output: `e2e_alps_full/match_results_mine.json`, `match_run_mine.log`, `tegrastats_match_mine.log`. For another test set, prefix `D=../e2e_geotiff`.
- **LightGlue TensorRT engine** (once): `python deploy/jetson/lightglue_trt.py export --out ~/lightglue_sift_k1024.onnx` (workstation, from the repo root; checks against PyTorch), then on the Jetson `/usr/src/tensorrt/bin/trtexec --onnx=lightglue_sift_k1024.onnx --saveEngine=lightglue_sift_k1024_fp16.trt --fp16`.
- On the workstation, the same script runs without `--engine` (it uses PyTorch fingerprints), with `--tile-feats` or no tile options.

### 5.6 Compare setups
Make a baseline run and your new run **on the same day**, then:
```bash
cd ~/Desktop/astroloc && python3 compare_topk.py --data e2e_alps_full --tags <baseline_tag> <new_tag>
```
It prints accuracy, time, memory, power and energy for each run, and lists fixes lost or gained and answers changed against the first tag.

### 5.7 Profiling
```bash
cd ~/Desktop/astroloc/repo
tegrastats --interval 200 --logfile ../prof_tegra.log &       # stop with: tegrastats --stop
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../profile_full.py --photos 50 --lightglue-mp --out profile_mine.json
PYTHONPATH=~/Desktop/astroloc/pydeps nsys profile -t cuda,nvtx -o ../lg_bench python3 ../lg_bench.py
nsys stats --report cuda_gpu_kern_sum ../lg_bench.nsys-rep
```
- On a Jetson the GPU has no separate memory; `nvidia-smi` doesn't apply. Use `tegrastats` for GPU load, RAM and power (`VDD_IN` is the module's total input power).
- PyTorch's own profiler reports no GPU activity on this Jetson build, so use `nsys` for GPU kernel time.

### 5.8 Real two-frame pairs for OD
```bash
# workstation, from ~/argus-localization (CPU only)
~/venv-export/bin/python deploy/jetson/two_frame/find_pairs.py --out-dir ~/pairs --sample 40
# Mac, in this folder (needs internet; NASA's site can take ~1 min per page; resumable)
source config.sh && scp $WS:pairs/pairs_heldout_sample40.json two_frame/
cd two_frame && python3 fetch_nasa_times.py pairs_heldout_sample40.json && python3 check_same_pass.py pairs_heldout_sample40.json
scp pairs_heldout_sample40.json $WS:pairs/ && cd ..
# workstation: build a test folder from the confirmed pairs, then copy it to the Jetson (§5.10)
~/venv-export/bin/python deploy/jetson/two_frame/prepare_pair_set.py --pairs ~/pairs/pairs_heldout_sample40.json \
    --db-tiles ~/e2e_alps_full/db_tiles.json --out ~/e2e_pairs
# Jetson
cd ~/Desktop/astroloc/e2e_pairs && ln -sf ../e2e_alps_full/db_vecs.npy . && ln -sf ../e2e_alps_full/db_tiles.json .
cd ../repo && PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../e2e_match.py --data ../e2e_pairs --tiles-dir ../tiles/2021_09 \
    --engine ../astroloc_full_fp32.trt --top-k 10 --lightglue-mp --tile-pack ../tile_feats_pack --search-on-gpu --out pair_results.json
cd .. && python3 two_frame/summarize_pairs.py e2e_pairs/pairs.json e2e_pairs/pair_results.json
```
- Capture times come from NASA's Gateway to Astronaut Photography: `eol.jsc.nasa.gov/SearchPhotos/photo.pl?mission=ISS065&roll=E&frame=207004`. Each page also gives the ISS nadir point and altitude, which serve as ground truth for OD.
- Results so far, in `two_frame/`: 38 of 40 sampled pairs are the same pass (12–147 s apart), and 8 got fixes on both photos.

### 5.8b Localization → OD on a simulated pass
```bash
# Mac or any machine with casadi + scipy and an FSW-Payload checkout (FSW_PYTHON_OD=/path/to/FSW-Payload/python_od)
python3 od/simulate_pass.py --perfect --out od/runs/sim_perfect                 # check of the chain
python3 od/simulate_pass.py --seed 0 --out od/runs/sim_seed0                      # simulated localization, measured error stats
python3 od/simulate_pass.py --lat0 31.5 --lon0 13.5 --duration-s 330 --capture-s 10 --poses-only --out od/runs/real_eu
# workstation, from ~/argus-localization: render those frames from the GeoTIFFs
~/venv-export/bin/python deploy/jetson/od/render_pass.py --poses od/runs/real_eu/frames_poses.json --out ~/e2e_pass_eu
# Jetson: copy the folder over, link db_vecs.npy and db_tiles.json from e2e_geotiff/, then
D=../e2e_pass_eu ./run_match_full.sh pass --top-k 10 --tile-pack ../tile_feats_pack --search-on-gpu \
    --lightglue-trt ../lightglue_sift_k1024_fp16.trt --stop-inliers 30 --save-tie-points
# back on the Mac: OD from the real tie points
python3 od/simulate_pass.py --lat0 31.5 --lon0 13.5 --duration-s 330 --capture-s 10 --sigma-rad 0.002 \
    --tie-points-from od/runs/real_eu/match_results_pass.json --out od/runs/real_eu
```

### 5.9 Walkthrough figures
```bash
# Jetson
cd ~/Desktop/astroloc/repo && PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../pipeline_trace.py --out ../trace
# Mac, after copying ~/Desktop/astroloc/trace to walkthrough/pipeline_walkthrough/data (needs matplotlib)
python3 visualize_trace.py --trace walkthrough/pipeline_walkthrough/data --out walkthrough/pipeline_walkthrough
```

### 5.10 Moving files
- Scripts: `./sync.sh workstation` or `./sync.sh jetson`, after setting `WS_USER` in `config.sh`.
- Large files from the workstation to the Jetson: stream through your Mac, so nothing is stored there.
```bash
source config.sh
ssh $WS 'cat ~/astroloc_full_static.onnx' | ssh -J $WS $JETSON 'cat > ~/Desktop/astroloc/astroloc_full_static.onnx'
ssh $WS 'tar -C ~ -cf - e2e_alps_full' | ssh -J $WS $JETSON 'tar -C ~/Desktop/astroloc -xf -'
```

---

## 6. `e2e_match.py` options

| Option | Meaning |
|---|---|
| `--data`, `--tiles-dir` | test folder from `e2e_prepare.py`, and the tile images (their filenames carry the tile corners) |
| `--engine` | TensorRT engine; fingerprints are computed in the same process (Jetson) |
| `--top-k` | candidate tiles checked by the matcher (current: 10) |
| `--min-inliers` | inliers needed for a fix (default 30, the repo's value) |
| `--lightglue-mp` | PyTorch LightGlue in mixed precision (not needed with `--lightglue-trt`) |
| `--max-keypoints` | SIFT points per image (default 1,024; 512 was tested and loses fixes) |
| `--tile-feats DIR` | precomputed per-tile features |
| `--tile-pack DIR` | packed features (current) |
| `--search-on-gpu` | fingerprint search on the GPU (current: on) |
| `--tiles-on-gpu` | preload the whole pack into GPU memory (tested: no gain, +1.8 GB) |
| `--batch-tiles` | one batched LightGlue call for all tiles (tested: slower) |
| `--lightglue-trt FILE` | LightGlue as a TensorRT engine (current: `lightglue_sift_k1024_fp16.trt`) |
| `--stop-inliers N` | stop at the first tile with ≥ N inliers (current: 30) |
| `--log-tiles` | record every candidate tile's inliers and error (for `simulate_stop.py`) |
| `--save-tie-points` | store each fix's tie points `[u, v, lat, lon]` and the photo size (what OD uses) |
| `--limit N`, `--out` | first N photos only; output file name |

Every fix also logs `tie_error_km` (tie points vs truth from the photo's four corners, within ~0.2 km of exact on the Argus-like frames) next to `match_error_km` (footprint centre); the summary reports both.

---

## 7. Known issues and open questions

- **Image shape:** the team uses square images, so the square 1024 × 1024 test photos match the flight input. The GeoTIFF evaluation also favours a centre square crop over squashing the 4608 × 2592 frame (R@1 71% vs 55%).
- **Training overlap:** a photo counts as "unseen" if it is within 2,000 km of a held-out region. Whether the GAPE training photos overlap the 17k photo set elsewhere is an open question.
- **Wrong fixes:** on astronaut photos 3 of 108 fixes are more than 50 km off, and a high inlier count doesn't rule this out (one 245-inlier fix was 90 km off). None on the Argus-like frames. OD should still filter them.
- **Snow / winter imagery:** northern GeoTIFF zones (~40–48° N) fix 0–7 of 13 frames against 12–13 in the south; their Landsat mosaics include snowy and winter scenes while the database tiles look like summer. Paused for now.
- **OD:** `od/simulate_pass.py` writes FSW-Payload's own OD input files directly. The repo's `integration/batchopt_adapter.py` is not used: it imports GNC-Payload through a hardcoded home-folder path and rescales pixels to 4608 × 2592, which doesn't fit square frames.
- **Checkpoints:** 10 full-size runs exist; `faithful_lora_v2` is the one named current. Nano v1 scores 3% top-1, so don't use it.
- **Loading old LoRA checkpoints:** those saved before commit `cc75c2b` need their weight names remapped. Our scripts and the repo (from `d92f5bc` on) both do this.
