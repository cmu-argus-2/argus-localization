#!/bin/bash
# Collects the full-pipeline error dataset (scripts/collect_error_dataset.py)
# for the 3 models the user asked for -- faithful_lora_v2, faithful_dynamic_v2,
# nano_v2 -- across all 6 regions, 300 queries/region each. One output file
# per model (separate processes writing to the same file could interleave
# partial lines), merged afterward. Both GPUs were idle when launched.
set -uo pipefail
cd /home/pvijayba/argus-localization
PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python
OUT_DIR=output/error_dataset
N_PER_REGION=300

$PY -u scripts/collect_error_dataset.py --models faithful_lora_v2 --n-per-region $N_PER_REGION \
  --device cuda:0 --out $OUT_DIR/faithful_lora_v2.jsonl \
  > logs/collect_errors_faithful_lora_v2.log 2>&1 &
P1=$!
echo "faithful_lora_v2 (GPU0) started, pid=$P1"

$PY -u scripts/collect_error_dataset.py --models faithful_dynamic_v2 --n-per-region $N_PER_REGION \
  --device cuda:1 --out $OUT_DIR/faithful_dynamic_v2.jsonl \
  > logs/collect_errors_faithful_dynamic_v2.log 2>&1 &
P2=$!
echo "faithful_dynamic_v2 (GPU1) started, pid=$P2"

wait $P1
echo "[$(date)] faithful_lora_v2 collection exited with $?"
wait $P2
echo "[$(date)] faithful_dynamic_v2 collection exited with $?"

$PY -u scripts/collect_error_dataset.py --models nano_v2 --n-per-region $N_PER_REGION \
  --device cuda:0 --out $OUT_DIR/nano_v2.jsonl \
  > logs/collect_errors_nano_v2.log 2>&1
echo "[$(date)] nano_v2 collection exited with $?"

cat $OUT_DIR/faithful_lora_v2.jsonl $OUT_DIR/faithful_dynamic_v2.jsonl $OUT_DIR/nano_v2.jsonl > $OUT_DIR/combined.jsonl
echo "=== [$(date)] ALL DONE, combined dataset at $OUT_DIR/combined.jsonl ==="
