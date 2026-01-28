# 🌊 Wave-DFINE: 波动传播增强的小麦穗检测器

将WaveFormer的波动传播机制集成到DFINE检测器，用于解决GWHD 2021数据集的**域泛化**和**小目标检测**难题。

---

## ✨ 核心创新

### 问题分析
DFINE在GWHD数据集上的三大瓶颈：
1. **小目标失效**: AP_s=0.039（仅3.9%）
2. **域泛化崩溃**: Val 0.504 → Test 0.205（-59%）
3. **密集场景漏检**: UQ_8域（117个/图）召回率低

### 解决方案
引入**阻尼波动传播**机制（来自WaveFormer）：

```
传统Transformer: e^(-kω²t) → 高频过度衰减（热扩散）
Wave Propagation: e^(-αt/2) · cos(ωt) → 频率解耦 + 振荡保护
```

**优势**：
- ✅ 高频细节保留 → 小目标可见
- ✅ 物理先验建模 → 域无关特征
- ✅ O(N log N)复杂度 → 实时性保证

---

## 🚀 快速开始

### 1️⃣ 测试模块（1分钟）
```bash
cd /root/DEIM-DEIM
python engine/extre_module/wave_modules.py

# 预期输出：
# ✅ 所有模块测试通过！
```

### 2️⃣ 快速验证（2小时）
```bash
bash scripts/quick_test_wave_dfine.sh
```

### 3️⃣ 完整训练（12-24小时）
```bash
bash scripts/train_wave_dfine.sh
```

### 或使用交互式菜单
```bash
bash start_wave_dfine.sh
```

---

## 📊 预期性能

| 指标 | DFINE基线 | Wave-DFINE（预期） | 提升 |
|------|----------|------------------|------|
| **AP (Test)** | 0.205 | **0.26~0.30** | +27~46% |
| **AP_s** | 0.039 | **0.08~0.12** | **+105~208%** 🔥 |
| **AP_75** | 0.116 | **0.15~0.18** | +29~55% |
| **FPS** | ~120 | **110~130** | -8~+8% |

### 关键提升域
- **UQ_11** (小目标36%): AP从0.15→0.28 (+87%)
- **ARC_1** (OOD苏丹): AP从0.08→0.20 (+150%)
- **UQ_8** (密集117/图): AP从0.18→0.30 (+67%)

---

## 🏗️ 架构设计

### 方案1：混合架构（推荐）⭐
```yaml
Encoder:
  - Transformer分支: 全局语义建模
  - Wave分支: 频率感知特征增强
  - 自适应融合: 学习权重平衡
```

**配置**: `configs/cfg/wave-dfine-n.yaml`

### 方案2：完全替换
```yaml
Encoder:
  - WaveEncoderBlock: 用Wave完全替换Transformer
```

**配置**: `configs/cfg/wave-dfine-n-pure.yaml`

---

## 📁 文件结构

```
/root/DEIM-DEIM/
├── engine/extre_module/
│   └── wave_modules.py              ⭐ 核心实现（400行）
├── configs/
│   ├── cfg/
│   │   ├── wave-dfine-n.yaml        # 混合架构
│   │   └── wave-dfine-n-pure.yaml   # 纯Wave
│   └── baseline/
│       └── wave_dfine_hgnetv2_n_custom.yml
├── scripts/
│   ├── quick_test_wave_dfine.sh     # 快速验证
│   ├── train_wave_dfine.sh          # 完整训练
│   └── ablation_wave_dfine.sh       # 消融实验
├── tools/visualization/
│   └── wave_dfine_vis.py            # 可视化工具
└── papers_code/WaveFormer/
    ├── INTEGRATION_GUIDE.md         📖 详细指南（30页）
    └── INTEGRATION_SUMMARY.md       📝 完整总结
```

---

## 🔬 实验流程

### Phase 1: 基础验证（1周）
```bash
# 混合架构
bash scripts/train_wave_dfine.sh

# 纯Wave对比
bash scripts/train_wave_dfine.sh  # 修改配置为pure版本
```

### Phase 2: 消融实验（1周）
```bash
# 6组对比（每组80 epochs）
bash scripts/ablation_wave_dfine.sh
```

实验组：
1. DFINE基线
2. Wave weight=0.2/0.5/0.8
3. 纯Wave替换
4. 固定vs可学习参数

### Phase 3: 可视化（2天）
```python
from tools.visualization.wave_dfine_vis import *

# 频谱分析
visualize_frequency_spectrum(model, image, 'fig/freq.png')

# 检测对比
compare_detection_results(dfine_res, wave_res, imgs, 'fig/comp/')

# 域泛化曲线
plot_domain_generalization_curve(results, 'fig/domain.png')
```

---

## 📝 论文要点

### Method核心论述

**问题**: Transformer自注意力的固有缺陷
- 类热扩散特性导致高频细节丢失
- 基于相似度的建模对域纹理敏感
- O(N²)复杂度限制分辨率

**方案**: 波动传播算子
$$u(x,y,t) = \mathcal{F}^{-1}\{e^{-\alpha t/2}[\mathcal{F}(u_0)\cos(\omega_d t) + ...]\}$$

**优势**:
1. 频率-时间解耦：α与ω独立
2. 振荡保护：cos/sin维持高频
3. 域无关：物理规律泛化性强

### 关键图表

| 图表 | 内容 | 说明 |
|------|------|------|
| **图1** | 架构图 | Transformer + Wave双分支 |
| **图2** | 频谱对比 | Wave保留高频细节 |
| **图3** | 检测可视化 | 小目标/密集场景对比 |
| **表1** | 主实验 | 18域AP对比 |
| **表2** | 消融 | wave_weight影响 |

---

## ⚙️ 核心参数

### Wave2D配置
```python
wave_speed = 1.0    # c: 振荡频率（越大全局性越强）
damping = 0.1       # α: 衰减速度（越大越平滑）
```

**调参建议**：
- 小目标密集 → α=0.05~0.1, c=1.2~1.5
- 嘈杂数据 → α=0.3~0.5, c=0.5~0.8

### 训练配置
```yaml
batch_size: 8
lr: 0.0008
epochs: 160
wave_weight: 0.5  # Wave分支权重
```

---

## 🐛 常见问题

### Q1: 训练NaN
**A**: 增大阻尼系数或降低学习率
```python
self.damping = nn.Parameter(torch.ones(1) * 0.5)  # 从0.1→0.5
lr: 0.0004  # 从0.0008→0.0004
```

### Q2: 性能不升反降
**A**: 降低wave_weight
```yaml
wave_weight: 0.2  # 从0.5→0.2，逐步增强
```

### Q3: 显存溢出
**A**: 减小batch size或使用梯度累积
```bash
--batch-size 4 --grad-accum-steps 2
```

---

## 📚 参考文献

1. WaveFormer: Frequency-Time Decoupled Vision Modeling (AAAI 2026)
2. D-FINE: Redefine DETR... (arXiv 2024)
3. Global Wheat Head Detection 2021 (Plant Phenomics 2024)

---

## ✅ 成功标准

### 最低目标（可发表）
- ✅ Test集AP > 0.25 (+22%)
- ✅ AP_s > 0.07 (+79%)
- ✅ 3个域AP提升>15%

### 理想目标（顶会）
- 🎯 Test集AP > 0.28 (+37%)
- 🎯 AP_s > 0.10 (+156%)
- 🎯 FPS > 110
- 🎯 清晰的消融证明

---

## 🎓 投稿建议

- **首选**: CVPR 2027 Agri-Vision Workshop
- **备选**: ECCV 2026, Plant Phenomics (Q1期刊)
- **强调**: 首次将波动方程引入农业检测

---

## 📧 支持

- 📖 详细指南: [INTEGRATION_GUIDE.md](papers_code/WaveFormer/INTEGRATION_GUIDE.md)
- 📊 完整总结: [INTEGRATION_SUMMARY.md](papers_code/WaveFormer/INTEGRATION_SUMMARY.md)
- 💻 核心代码: [wave_modules.py](engine/extre_module/wave_modules.py)

---

## 🎉 立即开始

```bash
# 一键启动
bash start_wave_dfine.sh

# 或直接测试
python engine/extre_module/wave_modules.py
```

**祝实验顺利！🚀**

---

**创建日期**: 2026-01-23  
**版本**: v1.0  
**状态**: ✅ 已完成集成，可开始训练
