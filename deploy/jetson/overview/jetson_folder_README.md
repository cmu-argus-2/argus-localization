# ~/Desktop/astroloc — shared working folder for Argus localization

There is one shared account (`argus`) and this one folder for everyone. Full setup, commands and conventions are in the `argus-jetson-localization` package (`README.md`, `CLAUDE.md`). This file says what's here.

## Current pipeline (used by the scripts; don't move or delete)

| Item | What it is |
|---|---|
| `astroloc_full_fp32.trt` | current retriever engine: full-size AstroLoc `faithful_lora_v2`, TensorRT FP32 |
| `astroloc_full_static.onnx` | its ONNX source |
| `nano_v2_fp32.trt`, `nano_v2_static.onnx` | small model (nano v2) engine and ONNX |
| `tiles/2021_09/` | 7,602 zoom-9 map tiles (filenames carry the corner coordinates) |
| `tile_feats_pack/` | current: SIFT features of every tile, packed into one set |
| `tile_feats_2021_09/` | the same features, one `.npz` per tile (input to the pack) |
| `e2e_alps_full/` | 200-photo test set + full-size fingerprint database, plus every run's results and logs (`*_<tag>.*`) |
| `e2e_alps_v2/` | the same test set built with nano v2 |
| `e2e_pairs/` | 38 real same-pass photo pairs for two-frame OD (database files link to `e2e_alps_full/`) |
| `trace/` | recorded stages behind the pipeline walkthrough figures |
| `pydeps/`, `wheels/` | private Python packages (PyTorch 2.4, numpy 1.26, kornia, LightGlue) and their installers. Use them with `PYTHONPATH=~/Desktop/astroloc/pydeps` |
| `repo/` | the 9 argus-localization files the matcher needs |
| `*.py`, `run_*.sh` | shared copies of the package scripts. Update them only from the package (`sync.sh`) and tell the team |
| `samples/` | test images for `test_images.py` |
| `users/` | **your own scripts and data go in `users/<your-name>/`** |

The LightGlue weights live in `~/.cache/torch/hub/checkpoints/`.

## Superseded (kept for reference; not used by the current pipeline)

`nano_fp32.trt`, `nano_static.onnx`, `nano_fp16.trt`, `nano_fp16_output.npy`, `nano_fp32_output.npy` (nano v1 and the FP16 test); `astroloc_fp32.trt`, `astroloc_static.onnx` (oldest full-size checkpoint); `e2e_alps/` (nano v1 test set). Some teammate scripts still read these, so remove them only after the team agrees.

## A teammate's earlier retrieval runs

`jetson_run.sh`, `dump_vecs.py`, `run2_v1/`, `run2_v2/`. These are not part of the package; leave them alone.

## Rules for this shared folder

- Use your own run tag or output name for every run, so nobody's results get overwritten.
- Before timing anything, check nobody else is running: `pgrep -af "python3|trtexec"` and a few seconds of `tegrastats`.
- Never sudo, reboot or install system-wide. New Python packages go in `pydeps/` (`pip3 install --no-deps --target`).
- Delete only what you created.
