#!/bin/bash
# Stop-at-first-good-tile test. Step 1 (done): --log-tiles runs, replayed offline by simulate_stop.py.
# Step 2: timed runs, baseline (all 10 tiles) vs --stop-inliers 30, on both test sets.
cd ~/Desktop/astroloc
B="--top-k 10 --tile-pack ../tile_feats_pack --search-on-gpu --lightglue-trt ../lightglue_sift_k1024_fp16.trt"
D=../e2e_geotiff ./run_match_full.sh geo_all10 $B; sleep 15
D=../e2e_geotiff ./run_match_full.sh geo_stop30 $B --stop-inliers 30; sleep 15
./run_match_full.sh all10 $B; sleep 15
./run_match_full.sh stop30 $B --stop-inliers 30
echo done > e2e_alps_full/stop_done.txt
