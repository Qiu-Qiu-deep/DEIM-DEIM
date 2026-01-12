export CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=64
export NUM_GPUS=2

# torchrun --master_port=9930 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_hgnetv2_n_custom.yml -r outputs/deim_hgnetv2_n_custom/best_stg2.pth --test-only

# torchrun --master_port=9930 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_hgnetv2_n_custom.yml -r outputs/deim_hgnetv2_n_custom/best_stg2.pth --test-only --eval-wda

torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/deim_r18vd_120e_coco.yml -r outputs/deim_r18vd_120e_custom/best_stg2.pth --test-only --eval-wda
# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/dfine_hgnetv2_n_custom.yml -r outputs/dfine_hgnetv2_n_custom/best_stg2.pth --test-only
# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/dfine_hgnetv2_n_mal_custom.yml -r outputs/dfine_hgnetv2_n_mal_custom/best_stg2.pth --test-only
# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/baseline/rtdetrv2_r18vd_120e_coco.yml -r 