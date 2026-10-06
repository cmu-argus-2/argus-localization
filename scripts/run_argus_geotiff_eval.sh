#!/bin/bash
# Runs astroloc/eval/evaluate_argus_geotiffs.py (Landsat-8 Argus-format frames
# vs global 2021 zoom 8-10 reference DB) for each retriever, sequentially.
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
A=/mnt/sdc1/astroloc/reference_db/astroloc_train
N=/mnt/sdc1/astroloc/reference_db/nano_train

run() {
  echo "=== [$(date)] $1 ==="
  $PY -u -m astroloc.eval.evaluate_argus_geotiffs --run-name "$1" ${2:+--checkpoint "$2"} --device cuda:0
}

run faithful_lora_v2 $A/checkpoints_faithful_lora_v2/final.pt
run faithful_dynamic_v2 $A/checkpoints_faithful_dynamic_v2/final.pt
run nano_v2 $N/checkpoints_v2/final.pt
run pretrained_baseline ""
echo "=== [$(date)] ALL DONE ==="
