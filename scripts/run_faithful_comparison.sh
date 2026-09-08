#!/bin/bash
# Orchestrates the controlled AstroLoc comparison agreed with the user
# 2026-08-26: build the shared paper-scale data once, then run the two
# training runs (LoRA+static, full-FT+dynamic) that differ ONLY in fine-tune
# method and clustering mode. Serialized entirely on GPU1 -- GPU0 has a
# heavier (18% util) footprint from another user's (sunac) job and is left
# alone; GPU1's footprint from that same job is idle (0% util) but still
# present, so this is the safer of the two, not a guaranteed-clean GPU.
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python
export CUDA_VISIBLE_DEVICES=1

echo "=== Building shared training data (query pool, tile pool, pairs, split, initial clustering) ==="
$PY -u -m astroloc.training.build_full_data --device cuda:0 \
  > logs/build_full_data.log 2>&1
status=$?

if [ $status -ne 0 ]; then
  echo "DATA BUILD FAILED (exit $status), see logs/build_full_data.log" >> logs/build_full_data.log
  exit 1
fi

echo "=== Data build complete, running LoRA+static (current best proven) ==="
$PY -u -m astroloc.training.train_faithful \
  --use-lora \
  --checkpoint-dir /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora \
  --wandb-run-name faithful-lora-static \
  > logs/train_faithful_lora.log 2>&1
echo "LoRA+static run exited with $?"

echo "=== Running full-FT+dynamic (wholly faithful) ==="
$PY -u -m astroloc.training.train_faithful \
  --dynamic-batching --recluster-every-steps 5000 \
  --checkpoint-dir /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_dynamic \
  --wandb-run-name faithful-fullft-dynamic \
  > logs/train_faithful_dynamic.log 2>&1
echo "Full-FT+dynamic run exited with $?"

echo "BOTH FAITHFUL RUNS DONE"
