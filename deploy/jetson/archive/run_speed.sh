#!/bin/bash
cd ~/Desktop/astroloc
F="--tile-feats ../tile_feats_2021_09 --top-k 10"
./run_match_full.sh base10 $F;                                   sleep 15
./run_match_full.sh mp $F --lightglue-mp;                        sleep 15
./run_match_full.sh kp512 $F --max-keypoints 512;                sleep 15
./run_match_full.sh mp_kp512 $F --lightglue-mp --max-keypoints 512
echo done > e2e_alps_full/speed_done.txt
