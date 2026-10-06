#!/bin/bash
# Waits for the nano faithful-dynamic run to finish, then evaluates it on the
# Argus GeoTIFF frames and on the 6-region ISS benchmark (same full-DB eval as
# the big faithful models, so the numbers are directly comparable).
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CKPT=/mnt/sdc1/astroloc/reference_db/nano_train/checkpoints_faithful_dynamic/final.pt

until [ -f "$CKPT" ] && ! pgrep -f "train_faithful.*checkpoints_faithful_dynamic" > /dev/null; do sleep 60; done
echo "=== [$(date)] training done, GeoTIFF eval ==="
$PY -u -m astroloc.eval.evaluate_argus_geotiffs --run-name nano_faithful_dynamic --checkpoint "$CKPT" --device cuda:0
echo "=== [$(date)] ISS 6-region eval ==="
$PY -u -m astroloc.eval.evaluate --run-name nano_faithful_dynamic --checkpoint "$CKPT" --device cuda:0 --no-wandb
echo "=== [$(date)] ALL DONE ==="
