#!/bin/bash
# Wave-DFINE 消融实验脚本

echo "🔬 Wave-DFINE 消融实验"
echo "========================================"

# 实验配置
EPOCHS=80  # 消融实验用较少epochs
DEVICE="0,1"
BASE_DIR="outputs/ablation_$(date +%Y%m%d)"

# 创建输出目录
mkdir -p $BASE_DIR

# 实验1: DFINE基线（对照组）
echo ""
echo "实验1/6: DFINE基线"
python train.py \
  --config configs/baseline/dfine_hgnetv2_n_custom.yml \
  --device $DEVICE \
  --batch-size 8 \
  --epochs $EPOCHS \
  --save-dir $BASE_DIR/exp1_dfine_baseline \
  --note "Ablation: DFINE baseline"

# 实验2: Wave weight=0.2
echo ""
echo "实验2/6: Wave weight=0.2"
sed 's/wave_weight: 0.5/wave_weight: 0.2/g' configs/baseline/wave_dfine_hgnetv2_n_custom.yml > /tmp/wave_02.yml
python train.py \
  --config /tmp/wave_02.yml \
  --device $DEVICE \
  --batch-size 8 \
  --epochs $EPOCHS \
  --save-dir $BASE_DIR/exp2_wave_w02 \
  --note "Ablation: wave_weight=0.2"

# 实验3: Wave weight=0.5
echo ""
echo "实验3/6: Wave weight=0.5"
python train.py \
  --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
  --device $DEVICE \
  --batch-size 8 \
  --epochs $EPOCHS \
  --save-dir $BASE_DIR/exp3_wave_w05 \
  --note "Ablation: wave_weight=0.5"

# 实验4: Wave weight=0.8
echo ""
echo "实验4/6: Wave weight=0.8"
sed 's/wave_weight: 0.5/wave_weight: 0.8/g' configs/baseline/wave_dfine_hgnetv2_n_custom.yml > /tmp/wave_08.yml
python train.py \
  --config /tmp/wave_08.yml \
  --device $DEVICE \
  --batch-size 8 \
  --epochs $EPOCHS \
  --save-dir $BASE_DIR/exp4_wave_w08 \
  --note "Ablation: wave_weight=0.8"

# 实验5: 纯Wave替换
echo ""
echo "实验5/6: 纯Wave替换"
python train.py \
  --config configs/baseline/wave_dfine_pure_hgnetv2_n_custom.yml \
  --device $DEVICE \
  --batch-size 8 \
  --epochs $EPOCHS \
  --save-dir $BASE_DIR/exp5_pure_wave \
  --note "Ablation: Pure Wave replacement"

# 实验6: Wave + 固定参数
echo ""
echo "实验6/6: Wave固定物理参数"
# 需要修改wave_modules.py中learnable_params=False
python train.py \
  --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
  --device $DEVICE \
  --batch-size 8 \
  --epochs $EPOCHS \
  --save-dir $BASE_DIR/exp6_wave_fixed \
  --note "Ablation: Fixed wave parameters"

echo ""
echo "✅ 消融实验完成！"
echo "========================================"
echo "结果保存在: $BASE_DIR"
echo ""
echo "分析结果："
echo "python tools/analyze_ablation.py --result-dir $BASE_DIR"
