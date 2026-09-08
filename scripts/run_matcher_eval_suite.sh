#!/bin/bash
# Runs scripts/evaluate_astroloc_matched.py (SIFT-LightGlue matcher attached to a
# DinoV2SaladRetriever) for all 4 trained retriever variants across all 6 eval
# regions. One-off orchestration script, not meant to be a permanent CLI surface.
set -uo pipefail
cd /home/pvijayba/argus-localization

PY=/home/pvijayba/miniconda3/envs/pri_env/bin/python
export CUDA_VISIBLE_DEVICES=1

ASTROLOC_CACHE=/mnt/sdc1/astroloc/reference_db/astroloc_train
NANO_CACHE=/mnt/sdc1/astroloc/reference_db/nano_train

regions=("Alps" "Texas" "Toshka Lakes" "Amazon" "Napa" "Gobi")

run_one() {
  local model_name="$1" checkpoint="$2" region="$3" db_cache="$4"
  echo "### MODEL=$model_name REGION=$region ###"
  $PY -u scripts/evaluate_astroloc_matched.py \
    --region "$region" \
    --checkpoint "$checkpoint" \
    --db-cache "$db_cache" \
    --num-queries 50
  echo "### END MODEL=$model_name REGION=$region ###"
}

for region in "${regions[@]}"; do
  region_us="${region// /_}"
  if [[ "$region" == "Alps" || "$region" == "Texas" || "$region" == "Toshka Lakes" ]]; then
    lora_cache="$ASTROLOC_CACHE/eval_cache/lora_finetuned_a/$region_us"
  else
    lora_cache="$ASTROLOC_CACHE/eval_cache/lora_finetuned_b/$region_us"
  fi
  run_one "astroloc_dynamic" "$ASTROLOC_CACHE/checkpoints_dynamic/final.pt" "$region" \
    "$ASTROLOC_CACHE/eval_cache/dynamic_finetuned/$region_us"
  run_one "astroloc_lora" "$ASTROLOC_CACHE/checkpoints_lora/final.pt" "$region" "$lora_cache"
  run_one "nano_dynamic" "$NANO_CACHE/checkpoints/final.pt" "$region" \
    "$NANO_CACHE/eval_cache/nano-dinov2s-dynamic-512d/$region_us"
  run_one "nano_lora" "$NANO_CACHE/checkpoints_v2/final.pt" "$region" \
    "$NANO_CACHE/eval_cache/nano-v2-multizoom-lora-dynamic/$region_us"
done

echo "ALL DONE"
