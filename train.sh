#!/bin/bash

set -e

cd ~/wyq/DEIM-DEIM
eval "$(conda shell.zsh hook)" || eval "$(conda shell.bash hook)"
conda activate deim

ulimit -n 65536

# 优化多线程性能（避免系统过载）
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# export CUDA_LAUNCH_BLOCKING=1

# GPU配置
export CUDA_VISIBLE_DEVICES=4,5
NUM_GPUS=2

# torchrun --master_port=9940 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/paper_first.yml

# torchrun --master_port=9937 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/dfine.yml

# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/deim/deim_hgnetv2_n_custom.yml

CUDA_VISIBLE_DEVICES=1 python train.py -c configs/yaml/dfine-FDPN.yml
python train.py -c configs/yaml/dfine.yml
echo "✅ Training completed!"
