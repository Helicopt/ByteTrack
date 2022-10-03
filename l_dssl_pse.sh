#!/bin/bash
source s0.3.6_py38
#source s0.3.4
ROOT=./
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
CPUS_PER_TASK=`echo "$1 * 8" | bc`
bs=`echo "$1 * 6" | bc`

OMP_NUM_THREADS=4 spring.submit arun  --quotatype=auto -n 1 --ntasks-per-node=1 --gres=gpu:$1 --gpu -p toolchain --job-name=$3 --cpus-per-task=${CPUS_PER_TASK} \
"python -u tools/train.py \
  -f=$cfg \
  -d $1 -b $bs --fp16 -o -c YOLOX_outputs/yolox_l_selfdet/epoch_120_ckpt.pth --resume --pseu $4 \
  2>&1 | tee -a output_logs/train.$T.$(basename $cfg).log "

cfg_ft="$(dirname $cfg)/ft_$(basename $cfg)"
cfg_dir="YOLOX_outputs/$(basename -s .py $cfg)"

cp $cfg $cfg_ft

OMP_NUM_THREADS=4 spring.submit arun  --quotatype=auto -n 1 --ntasks-per-node=1 --gres=gpu:1 --gpu -p toolchain --job-name=$3 --cpus-per-task=8 \
"python -u tools/train.py \
  -f=$cfg_ft \
  -d 1 -b 6 --fp16 -o -c $cfg_dir/last_epoch_ckpt.pth.tar --resume $4 \
  2>&1 | tee -a output_logs/train.$T.$(basename $cfg_ft).log "
