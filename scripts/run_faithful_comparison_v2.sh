#!/bin/bash
# v2 of run_faithful_comparison.sh: rebuilds the shared training-data cache
# with build_full_data.py's fix (L_MUM clustering/quadruplets now draw from
# the FULL eligible reference-tile pool, ~391,890 tiles, instead of only the
# ~52,844 tiles that happened to win a query match -- see that script's
# comment). Region scoping was already global (whole GAPE+EarthLoc pool minus
# the 6 eval regions) in build_full_data.py, not the 6-coastal-region
# TRAIN_REGIONS subset astroloc/README.md describes -- that subset belongs to
# an older, unused-by-this-path data pipeline, so no region-scoping change
# was needed here.
#
# Unlike the original orchestrator (sequential on GPU1 alone), this runs
# LoRA+static on GPU0 and full-FT+dynamic on GPU1 truly in parallel, once the
# shared data cache is built. Both GPUs were fully idle when this was written
# (2026-09-22) -- re-check `nvidia-smi` before relying on that if this is
# resumed later.
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python

echo "=== [$(date)] Building shared training data v2 (full-tile-pool clustering) ==="
CUDA_VISIBLE_DEVICES=0 $PY -u -m astroloc.training.build_full_data --device cuda:0 \
  > logs/build_full_data_v2.log 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "DATA BUILD FAILED (exit $status), see logs/build_full_data_v2.log"
  exit 1
fi
echo "=== [$(date)] Data build complete, launching both runs in parallel ==="

CUDA_VISIBLE_DEVICES=0 $PY -u -m astroloc.training.train_faithful \
  --use-lora --device cuda:0 \
  --checkpoint-dir /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora_v2 \
  --wandb-run-name faithful-lora-static-v2-fulltiles \
  > logs/train_faithful_lora_v2.log 2>&1 &
LORA_PID=$!
echo "LoRA+static (GPU0) started, pid=$LORA_PID"

CUDA_VISIBLE_DEVICES=1 $PY -u -m astroloc.training.train_faithful \
  --dynamic-batching --recluster-every-steps 5000 --device cuda:0 \
  --checkpoint-dir /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_dynamic_v2 \
  --wandb-run-name faithful-fullft-dynamic-v2-fulltiles \
  > logs/train_faithful_dynamic_v2.log 2>&1 &
DYNAMIC_PID=$!
echo "Full-FT+dynamic (GPU1) started, pid=$DYNAMIC_PID"

wait $LORA_PID
echo "[$(date)] LoRA+static v2 exited with $?"
wait $DYNAMIC_PID
echo "[$(date)] Full-FT+dynamic v2 exited with $?"

echo "=== [$(date)] BOTH FAITHFUL V2 RUNS DONE ==="
