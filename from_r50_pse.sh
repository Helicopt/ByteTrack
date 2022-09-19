#!/bin/bash
source s0.3.6_py38
ROOT=./
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=`echo "$1 * 6" | bc`
bs=`echo "$1 * 4" | bc`

OMP_NUM_THREADS=4 spring.submit run -x SH-IDC1-10-142-4-93 --quotatype=auto -n 1 --ntasks-per-node=1 --gres=gpu:$1 --gpu -p toolchain --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -u tools/train.py \
  -f=$cfg \
  -d $1 -b $bs --fp16 -c YOLOX_outputs/yolox_r50/best_ckpt.pth --resume --pseu $4 \
  2>&1 | tee -a output_logs/train.$T.$(basename $cfg).log "
