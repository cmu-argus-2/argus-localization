#!/bin/bash
# Waits for the LoRA+static faithful training run to finish, then evaluates
# both faithful checkpoints (retrieval recall@k, 6 regions) sequentially on
# GPU1 -- same serialization constraint as the training runs (GPU0 left
# alone, sunac's job).
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python
export CUDA_VISIBLE_DEVICES=1
LORA_PID=218043

while kill -0 "$LORA_PID" 2>/dev/null; do
  sleep 30
done
echo "LoRA training finished (or was already done), starting eval"

$PY -u -m astroloc.eval.evaluate \
  --checkpoint /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_dynamic/final.pt \
  --run-name faithful_dynamic --device cuda:0 \
  > logs/eval_faithful_dynamic.log 2>&1
echo "faithful_dynamic eval exited with $?"

$PY -u -m astroloc.eval.evaluate \
  --checkpoint /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora/final.pt \
  --run-name faithful_lora --device cuda:0 \
  > logs/eval_faithful_lora.log 2>&1
echo "faithful_lora eval exited with $?"

echo "BOTH FAITHFUL EVALS DONE"
