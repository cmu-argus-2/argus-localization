#!/bin/bash
cd ~/Desktop/astroloc
B="--top-k 10 --lightglue-mp"
./run_match_full.sh cur   $B --tile-feats ../tile_feats_2021_09;                                    sleep 15
./run_match_full.sh small $B --tile-pack ../tile_feats_pack --search-on-gpu;                        sleep 15
./run_match_full.sh preload  $B --tile-pack ../tile_feats_pack --search-on-gpu --tiles-on-gpu;         sleep 15
./run_match_full.sh batch $B --tile-pack ../tile_feats_pack --search-on-gpu --batch-tiles
echo done > e2e_alps_full/speed2_done.txt
