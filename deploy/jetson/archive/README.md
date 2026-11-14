# Archive

Scripts behind past experiments. They are not part of the current setup, and are kept so the results in `../RUNBOOK.md` (Experiment log) can be reproduced.

| File | What it was for |
|---|---|
| `profile_match.py` | per-stage matcher timing before tile features were stored (superseded by `../profile_full.py`) |
| `run_speed.sh`, `run_speed2.sh` | comparison batches: half precision, 512 points, CPU fixes, preloaded tile features, batched LightGlue |
| `compare_precision.py` | retriever engine FP32 vs FP16 (FP16 rejected) |
| `test_batched_lightglue.py` | batched PyTorch LightGlue vs one tile at a time (batched was slower) |

On the Jetson these run from `~/Desktop/astroloc/`, next to `e2e_match.py` (the shared folder is flat).
