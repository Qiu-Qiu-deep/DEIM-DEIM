#!/bin/bash
# Wave-DFINE 快速验证脚本
# 用10 epochs测试模块是否正常工作

echo "🌊 Wave-DFINE 快速验证"
echo "========================================"

# 1. 测试模块
echo "Step 1: 测试Wave模块..."
cd /root/DEIM-DEIM
python engine/extre_module/wave_modules.py

if [ $? -ne 0 ]; then
    echo "❌ 模块测试失败！请检查代码"
    exit 1
fi
echo "✅ 模块测试通过"

# 2. 快速训练验证
echo ""
echo "Step 2: 快速训练验证（10 epochs）..."
python train.py \
  --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
  --device 0,1 \
  --batch-size 4 \
  --epochs 10 \
  --save-dir outputs/wave_dfine_quick_test \
  --note "Quick validation test"

if [ $? -ne 0 ]; then
    echo "❌ 训练失败！请检查配置"
    exit 1
fi

echo ""
echo "✅ 快速验证完成！"
echo "========================================"
echo "下一步："
echo "1. 检查 outputs/wave_dfine_quick_test/train.log"
echo "2. 确认loss正常下降"
echo "3. 如果没问题，运行完整训练："
echo "   bash scripts/train_wave_dfine.sh"
