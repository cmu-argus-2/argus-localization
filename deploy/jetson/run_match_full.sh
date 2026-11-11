#!/bin/bash
# Full localization flow on the Jetson (TensorRT retriever + search + matcher), with tegrastats logging.
# Lives at ~/Desktop/astroloc/run_match_full.sh on the Jetson.
#   run_match_full.sh                      tile features computed per photo
#   run_match_full.sh pre --tile-feats ../tile_feats_2021_09
# The first argument tags the output files; the rest go to e2e_match.py.
TAG=${1:-}; [ $# -gt 0 ] && shift
SUF=${TAG:+_$TAG}
cd ~/Desktop/astroloc/repo
D=${D:-../e2e_alps_full}   # override for another test set, e.g. D=../e2e_geotiff
rm -f $D/tegrastats_match$SUF.log
tegrastats --interval 1000 --logfile $D/tegrastats_match$SUF.log &
TP=$!
sleep 8
PYTHONPATH=$HOME/Desktop/astroloc/pydeps python3 ../e2e_match.py --data $D --tiles-dir ../tiles/2021_09 \
    --engine ../astroloc_full_fp32.trt --out match_results$SUF.json "$@" > $D/match_run$SUF.log 2>&1
echo "exit code $?" >> $D/match_run$SUF.log
sleep 3
kill $TP
tegrastats --stop >/dev/null 2>&1
