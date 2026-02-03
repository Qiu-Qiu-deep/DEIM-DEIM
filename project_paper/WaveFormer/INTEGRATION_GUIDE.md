# Wave-DFINE 集成指南

## 📚 概述

将WaveFormer的波动传播机制集成到DFINE检测器中，用于解决小麦穗检测的域泛化和小目标检测难题。

## 🎯 三种集成方案

### 方案1：混合架构（推荐先做）⭐
- **文件**: `wave-dfine-n.yaml`
- **模块**: `WaveEnhancedEncoder`
- **特点**: 
  - Transformer + Wave双分支并行
  - 风险最低，可渐进式验证
  - 通过`wave_weight`参数控制Wave贡献度
- **预期**: 
  - AP提升 +0.02~0.05
  - AP_s提升 +0.03~0.08
  - FPS下降 <10%

### 方案2：完全替换
- **文件**: `wave-dfine-n-pure.yaml`
- **模块**: `WaveEncoderBlock`
- **特点**:
  - 完全用Wave替换Transformer
  - 推理速度更快
  - 需要更多调参
- **预期**:
  - AP提升 +0.01~0.03
  - FPS提升 +10~20%
  - 训练可能不稳定

### 方案3：多尺度Wave（后续扩展）
- **模块**: `MultiScaleWaveEncoder`
- **特点**: 
  - P4和P5使用不同波动参数
  - 高频细节 + 全局语义分离建模
  - 最大化性能提升

---

## 🔧 安装步骤

### Step 1: 注册Wave模块到tasks.py

在 `/root/DEIM-DEIM/engine/extre_module/tasks.py` 中修改：

```python
# 1. 在文件顶部导入
from engine.deim.hybrid_encoder import RepNCSPELAN4, ConvNormLayer_fuse, SCDown, CSPLayer, TransformerEncoderBlock
# 👇 添加这行
from engine.extre_module.wave_modules import WaveEnhancedEncoder, WaveEncoderBlock

# 2. 在parse_model函数中（约第219行），在TransformerEncoderBlock处理后添加：
elif m in {TransformerEncoderBlock}:    
    c2 = ch[f]
    args = [c2, *args]
# 👇 添加这几行
elif m in {WaveEnhancedEncoder, WaveEncoderBlock}:
    c2 = ch[f]
    args = [c2, *args]
```

### Step 2: 创建训练配置

在 `/root/DEIM-DEIM/configs/baseline/` 创建：

```yaml
# wave_dfine_hgnetv2_n_custom.yml
includes:
  - ../base/common.yml
  - ../cfg/wave-dfine-n.yaml  # 👈 使用Wave配置
  - ../base/dataloader.yml
  - ../base/optimizer.yml

num_classes: 1
remap_mscoco_category: False
use_focal_loss: True

epoches: 160
lr: 0.0008
batch_size: 8

model_config:
  pretrained_path: ""
```

### Step 3: 测试模块

```bash
# 测试Wave模块是否正常工作
cd /root/DEIM-DEIM
python engine/extre_module/wave_modules.py

# 预期输出：
# ============================================================
# 测试Wave2D模块
# ============================================================
# Wave2D输入: torch.Size([2, 128, 20, 20]), 输出: torch.Size([2, 128, 20, 20])
# ...
# ✅ 所有模块测试通过！
```

### Step 4: 小规模验证

```bash
# 用10 epochs快速验证收敛性
python train.py \
  --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
  --device 0,1 \
  --batch-size 8 \
  --epochs 10 \
  --save-dir outputs/wave_dfine_test

# 观察：
# 1. loss是否正常下降
# 2. 是否有NaN/Inf
# 3. 显存占用是否合理
```

---

## 📊 评估与对比

### 训练完整模型

```bash
# 方案1：混合架构（推荐）
python train.py \
  --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
  --device 0,1 \
  --batch-size 8 \
  --epochs 160 \
  --save-dir outputs/wave_dfine_hybrid

# 方案2：纯Wave（实验性）
python train.py \
  --config configs/baseline/wave_dfine_pure_hgnetv2_n_custom.yml \
  --device 0,1 \
  --batch-size 8 \
  --epochs 160 \
  --save-dir outputs/wave_dfine_pure
```

### 测试与对比

```bash
# 测试DFINE基线
python test.py \
  --config configs/baseline/dfine_hgnetv2_n_custom.yml \
  --checkpoint outputs/dfine_baseline/best.pth

# 测试Wave-DFINE
python test.py \
  --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
  --checkpoint outputs/wave_dfine_hybrid/best.pth
```

### 关键指标对比

| 指标 | DFINE基线 | Wave-DFINE（预期） | 提升幅度 |
|------|----------|------------------|----------|
| AP (Test) | 0.205 | **0.26~0.28** | +27~37% |
| AP_50 | 0.538 | **0.58~0.60** | +8~12% |
| AP_75 | 0.116 | **0.15~0.17** | +29~47% |
| AP_s | 0.039 | **0.08~0.12** | +105~208% 🔥 |
| AP_m | 0.227 | **0.28~0.30** | +23~32% |
| FPS | ~120 | **110~125** | -8~+4% |

---

## 🔍 调试与优化

### 常见问题

#### 1. 训练不收敛
**症状**: loss震荡，AP始终很低

**解决**:
```python
# 在wave_modules.py中调整初始化
self.wave_speed = nn.Parameter(torch.ones(1) * 0.5)  # 降低波速
self.damping = nn.Parameter(torch.ones(1) * 0.3)     # 增大阻尼

# 或在yaml中降低wave_weight
- [-1, WaveEnhancedEncoder, [8, 512, 0.1, "relu", 10000, False, True, 0.2]]
#                                                                        ^^^
```

#### 2. 显存溢出
**症状**: CUDA out of memory

**解决**:
```bash
# 减小batch size
--batch-size 4

# 或使用梯度累积
--grad-accum-steps 2
```

#### 3. 推理速度慢
**症状**: FPS < 100

**解决**:
```python
# 使用更快的DCT实现（已在代码中使用torch.fft.dct）
# 或降低分辨率
self.wave_branch = Wave2D(dim=d_model, res=10)  # 从20降到10
```

---

## 📈 消融实验设计

### 实验组设置

| 实验ID | 配置 | 目的 |
|--------|------|------|
| Exp-0 | DFINE原始 | 基线 |
| Exp-1 | Wave weight=0.2 | 弱融合 |
| Exp-2 | Wave weight=0.5 | 中等融合 |
| Exp-3 | Wave weight=0.8 | 强融合 |
| Exp-4 | 纯Wave | 完全替换 |
| Exp-5 | Wave + 可学习α,c | 自适应参数 |
| Exp-6 | 固定α,c | 物理先验 |

### 快速消融脚本

```bash
#!/bin/bash
# ablation_wave_dfine.sh

for weight in 0.2 0.5 0.8; do
  echo "Training with wave_weight=$weight"
  python train.py \
    --config configs/baseline/wave_dfine_hgnetv2_n_custom.yml \
    --device 0,1 \
    --batch-size 8 \
    --epochs 80 \
    --save-dir outputs/ablation_weight_$weight \
    --note "wave_weight=$weight"
done
```

---

## 🎨 可视化分析

### 频谱分析

```python
# 在wave_modules.py的forward中添加hook
def visualize_frequency_spectrum(self, x, save_path='freq_vis.png'):
    """可视化DCT系数分布"""
    x_freq = self.dct2d(x)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 输入空间域
    axes[0].imshow(x[0, 0].cpu().detach().numpy())
    axes[0].set_title('Spatial Domain')
    
    # 频率域（对数尺度）
    axes[1].imshow(torch.log(torch.abs(x_freq[0, 0]) + 1e-8).cpu().detach().numpy())
    axes[1].set_title('Frequency Domain (DCT)')
    
    # 阻尼后的频率域
    damped = x_freq * torch.exp(-self.damping * t)
    axes[2].imshow(torch.log(torch.abs(damped[0, 0]) + 1e-8).cpu().detach().numpy())
    axes[2].set_title('After Damping')
    
    plt.savefig(save_path)
```

### 域泛化分析

```python
# 分域评估脚本
domains = ['UQ_8', 'ARC_1', 'Terraref_2', ...]
for domain in domains:
    ap = evaluate_on_domain(model, domain)
    print(f"{domain}: AP={ap:.3f}")
```

---

## 📝 论文撰写要点

### Method部分结构

```latex
\subsection{Wave Propagation Operator}
\paragraph{Motivation}
传统Transformer的self-attention存在：
1) 小目标特征被大目标掩盖
2) 高频细节过度平滑
3) 域偏移敏感

\paragraph{Formulation}
引入阻尼波动方程：
$$u(x,y,t) = \mathcal{F}^{-1}\{e^{-\alpha t/2}[\mathcal{F}(u_0)\cos(\omega_d t) + ...]\}$$

\paragraph{Implementation}
在DFINE的Encoder层，使用Wave2D替换/增强...
```

### 实验部分重点

1. **表格1**: 与DFINE基线对比（Test集各域AP）
2. **表格2**: 消融实验（wave_weight影响）
3. **图1**: 频谱可视化（Wave如何保留高频）
4. **图2**: 小目标检测可视化（DFINE vs Wave-DFINE）
5. **图3**: 域泛化曲线（18个子域的AP分布）

---

## ✅ 检查清单

在提交论文前确认：

- [ ] 模块测试通过（运行`python wave_modules.py`）
- [ ] 至少训练3次取平均（随机种子：42, 123, 456）
- [ ] Test集AP提升 >5%
- [ ] AP_s提升 >50%（从0.039→0.06+）
- [ ] 至少2个域的AP提升 >10%
- [ ] FPS下降 <15%
- [ ] 消融实验覆盖wave_weight=[0.2, 0.5, 0.8]
- [ ] 可视化频谱图和检测结果
- [ ] 代码已整理并添加注释

---

## 📧 技术支持

遇到问题可查看：
1. `/root/DEIM-DEIM/engine/extre_module/wave_modules.py` - 模块实现
2. `/root/DEIM-DEIM/configs/cfg/wave-dfine-n.yaml` - 配置文件
3. WaveFormer原论文 - 理论细节

祝实验顺利！🚀
