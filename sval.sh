#!/bin/bash
source s0.3.6_py38
ROOT=./
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=${CPUS_PER_TASK:-6}

RES=YOLOX_outputs/$(basename -s .py $cfg)

OMP_NUM_THREADS=5 spring.submit arun --quotatype=auto -n 1 --ntasks-per-node=1 --gres=gpu:$1 --gpu -p toolchain --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -u tools/track.py \
  -f=$cfg \
  -d 1 -b 1 --fp16 -c $RES/best_ckpt.pth.tar --fuse --match_thresh 0.7 --mot20 $4 \
  2>&1 | tee -a output_logs/test.$T.$(basename $cfg).log "

