#!/bin/bash
# Evaluates the two v2 faithful checkpoints (full-tile-pool clustering fix,
# see run_faithful_comparison_v2.sh) on the 6 benchmark regions, in parallel
# across both GPUs -- both were idle when this was launched (2026-09-24).
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python

CUDA_VISIBLE_DEVICES=0 $PY -u -m astroloc.eval.evaluate \
  --checkpoint /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora_v2/final.pt \
  --run-name faithful_lora_v2 --device cuda:0 \
  > logs/eval_faithful_lora_v2.log 2>&1 &
LORA_PID=$!
echo "faithful_lora_v2 eval (GPU0) started, pid=$LORA_PID"

CUDA_VISIBLE_DEVICES=1 $PY -u -m astroloc.eval.evaluate \
  --checkpoint /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_dynamic_v2/final.pt \
  --run-name faithful_dynamic_v2 --device cuda:0 \
  > logs/eval_faithful_dynamic_v2.log 2>&1 &
DYN_PID=$!
echo "faithful_dynamic_v2 eval (GPU1) started, pid=$DYN_PID"

wait $LORA_PID
echo "[$(date)] faithful_lora_v2 eval exited with $?"
wait $DYN_PID
echo "[$(date)] faithful_dynamic_v2 eval exited with $?"

echo "=== [$(date)] BOTH V2 EVALS DONE ==="
