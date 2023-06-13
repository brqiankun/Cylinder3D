#! /bin/bash
set -x

name=cylinder_asym_networks
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym.py \
2>&1 | tee ./work/logs_dir/${name}_logs_tee.txt
#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym.py \
#2>&1 | tee ./work/logs_dir/${name}_logs_tee.txt
