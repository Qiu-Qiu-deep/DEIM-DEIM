export CUDA_VISIBLE_DEVICES=4,5
export OMP_NUM_THREADS=4
export NUM_GPUS=2

# torchrun --master_port=9930 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_hgnetv2_n_custom.yml -r outputs/deim_hgnetv2_n_custom/best_stg2.pth --test-only

# torchrun --master_port=9930 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_hgnetv2_n_custom.yml -r outputs/deim_hgnetv2_n_custom/best_stg2.pth --test-only --eval-wda

# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/my1_daqs.yml -r outputs/my1_daqs/best_stg2.pth --test-only

CUDA_VISIBLE_DEVICES=4 python train.py -c configs/yaml/dfine.yml -r outputs/dfine/best_stg1.pth --test-only --eval-wda # -u val_dataloader.total_batch_size=1 

# CUDA_VISIBLE_DEVICES=4 python train.py -c configs/yaml/dfine.yml -r outputs/dfine/best_stg2.pth --test-only -u val_dataloader.total_batch_size=1