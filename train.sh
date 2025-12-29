#!/bin/bash

set -e

cd ~/wyq/DEIM-DEIM
eval "$(conda shell.zsh hook)" || eval "$(conda shell.bash hook)"
conda activate deim

ulimit -n 65536

# 优化多线程性能（避免系统过载）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# GPU配置
export CUDA_VISIBLE_DEVICES=3,5
NUM_GPUS=2

torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/dfine/dfine_hgnetv2_n_custom.yml

torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/dfine/dfine_hgnetv2_n_mal_custom.yml

torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/deim/deim_hgnetv2_n_custom.yml

# 都是160轮ok

# CUDA_VISIBLE_DEVICES=0 python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg_test.yml

echo "✅ Training completed!"
