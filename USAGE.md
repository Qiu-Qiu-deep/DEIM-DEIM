
# 训练
```
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml
```

```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node 2 train.py -c configs/deim/deim_hgnetv2_n_custom.yml
```

# 测试
```
python train.py -c configs/test/dfine_hgnetv2_n_visdrone.yml --test-only -r /home/waas/best_stg2.pth
```
