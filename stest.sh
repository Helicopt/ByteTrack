#!/bin/bash
source s0.3.6_py38
#source s0.3.4
ROOT=./
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-6}

RES=YOLOX_outputs/$(basename -s .py $cfg)

OMP_NUM_THREADS=5 spring.submit arun --quotatype=auto -n 1 -x "SH-IDC1-10-142-4-[170]" --ntasks-per-node=1 --gres=gpu:$1 --gpu -p toolchain --job-name=$4 --cpus-per-task=${CPUS_PER_TASK} \
"python -u tools/track.py \
  -f=$cfg \
  -d 1 -b 1 --fp16 -c $3/best_ckpt.pth.tar --fuse --match_thresh 0.7 --mot20 --test-type real1600 $5 \
  2>&1 | tee -a output_logs/test.$T.$(basename $cfg).log "

rm -rf $RES/track_results_1600
cp -r $RES/track_results $RES/track_results_1600

OMP_NUM_THREADS=5 spring.submit arun --quotatype=auto -n 1 -x "SH-IDC1-10-142-4-[170]" --ntasks-per-node=1 --gres=gpu:$1 --gpu -p toolchain --job-name=$4 --cpus-per-task=${CPUS_PER_TASK} \
"python -u tools/track.py \
  -f=$cfg \
  -d 1 -b 1 --fp16 -c $3/best_ckpt.pth.tar --fuse --match_thresh 0.7 --mot20 --test-type real1920 $5 \
  2>&1 | tee -a output_logs/test.$T.$(basename $cfg).log "

rm -rf $RES/track_results_1920
cp -r $RES/track_results $RES/track_results_1920
