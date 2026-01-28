#!/bin/bash
# Wave-DFINE 完整训练脚本

echo "🌊 开始训练 Wave-DFINE"
echo "========================================"

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1

# 训练配置
CONFIG="configs/baseline/wave_dfine_hgnetv2_n_custom.yml"
BATCH_SIZE=8
EPOCHS=160
SAVE_DIR="outputs/wave_dfine_hybrid_$(date +%Y%m%d_%H%M%S)"

echo "配置文件: $CONFIG"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "保存路径: $SAVE_DIR"
echo "========================================"

# 开始训练
python train.py \
  --config $CONFIG \
  --device 0,1 \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --save-dir $SAVE_DIR \
  --note "Wave-DFINE hybrid architecture" \
  --eval-interval 10 \
  --save-best

echo ""
echo "✅ 训练完成！"
echo "模型保存在: $SAVE_DIR"
echo ""
echo "下一步："
echo "1. 测试模型: bash scripts/test_wave_dfine.sh $SAVE_DIR/best.pth"
echo "2. 对比基线: python tools/compare_results.py"
