# HyperGraph Enhancement for DFINE

基于Hyper-YOLO (TPAMI 2025) 的超图计算模块，用于捕获目标检测中的高阶特征关联。

## 📖 背景

传统目标检测方法（包括DFINE）主要关注语义特征学习，忽略了特征间的高阶结构信息：
- **PANet**: 只能融合相邻层特征（跨层级不足）
- **Gold-YOLO**: 缺少跨位置特征点交互
- **Transformer**: 仅建模pairwise关系，缺少高阶消息传递

**超图的优势**：
- 超边可连接多个节点（vs 简单图只能连2个）
- 天然适合建模复杂的高阶关联
- 打破grid结构限制，实现跨层级+跨位置的信息流动

---

## 🎯 核心模块

### 1. HyperGraphEnhance (完整版)

**功能**: 多尺度特征的超图增强

**工作流程**:
```python
输入: [P3(80×80), P4(40×40), P5(20×20)]
  ↓ 语义收集 (Semantic Collecting)
统一尺寸: 全部pool到20×20
  ↓ 特征融合
拼接: [B,768,20,20] → 降维: [B,256,20,20]
  ↓ 超图计算 (Hypergraph Computation)
构建超图: 基于距离构建ε-ball超边
高阶传递: V→E→V 两阶段消息聚合
  ↓ 语义散射 (Semantic Scattering)
插值+残差: 恢复原尺寸并融合
  ↓
输出: [P3', P4', P5'] 增强后的特征
```

**参数说明**:
```python
HyperGraphEnhance(
    hidden_dim=256,        # 特征通道数（匹配DFINE）
    threshold=8,           # 距离阈值（N:6, S:8, M:10）
    target_size=20,        # 统一特征尺寸（15/20/25）
    residual_weight=0.5    # 残差权重（0.3~0.7）
)
```
---

## 📁 配置文件

### dfine-s-hyper.yaml (推荐⭐)

**特点**: 在encoder末端添加一个超图模块
```yaml
encoder:
  # ... 原有encoder ...
  - [[16, 19, 22], HyperGraphEnhance, [256, 8, 20, 0.5]] # 超图增强

decoder:
  - [-1, DFINETransformer, {...}]
```

**优势**:
- ✅ 改动最小（只加一行）
- ✅ 保留DFINE原有结构
- ✅ 计算高效（一次超图计算）
- ✅ 易于消融实验

---

## 🚀 使用方法

### 训练

```bash
# 使用完整版超图
python train.py --config configs/cfg/dfine-s-hyper.yaml

# 使用轻量版超图
python train.py --config configs/cfg/dfine-s-hyper-lite.yaml

# 消融实验
python train.py --config configs/cfg/dfine-s-hyper-ablation.yaml
```

### 测试

```bash
python test.py \
  --config configs/cfg/dfine-s-hyper.yaml \
  --checkpoint outputs/dfine-s-hyper/best.pth
```

---

## 🔧 超参数调优

### 1. 距离阈值 (threshold)

控制超图的稀疏程度：
```python
threshold = 6   # 稀疏超图，连接少 → 速度快，可能欠拟合
threshold = 8   # 中等（推荐）
threshold = 10  # 密集超图，连接多 → 速度慢，表达力强
threshold = 12  # 非常密集，可能过拟合
```

**建议**:
- 小模型(N): threshold=6~8
- 中模型(S): threshold=8~10
- 大模型(M/L): threshold=10~12

### 2. 目标尺寸 (target_size)

统一多尺度特征的尺寸：
```python
target_size = 15  # 省内存，细节损失
target_size = 20  # 推荐，平衡
target_size = 25  # 保细节，计算慢
```

**计算量估算**:
- target_size=15: N=675点, 距离矩阵45万
- target_size=20: N=1200点, 距离矩阵144万
- target_size=25: N=1875点, 距离矩阵352万

### 3. 残差权重 (residual_weight)

控制原始特征和超图特征的融合：
```python
residual_weight = 0.3  # 更保守，接近baseline
residual_weight = 0.5  # 推荐
residual_weight = 0.7  # 更激进，依赖超图
```

---

## 📊 预期效果

基于Hyper-YOLO在COCO上的表现：

| 指标 | 预期变化 | 说明 |
|------|----------|------|
| **AP** | +1.5~2.5% | 超图捕获高阶关联 |
| **AP_small** | +2~4% | 跨尺度信息增强小目标 |
| **AP_large** | +0.5~1% | 大目标相对收益较小 |
| **参数量** | +0.5~1M | HyperConv相对轻量 |
| **FLOPs** | +5~10% | 主要来自cdist操作 |
| **推理速度** | -5~10% | TensorRT对cdist优化不足 |

**模型尺寸对比**:
```
DFINE-S:         ~12M params
+ HyperGraph:    ~13M params (+8%)
```

---

## 🔬 消融实验建议

### 实验1: 验证超图有效性
```bash
# Baseline
python train.py --config configs/cfg/dfine-s.yaml

# + HyperGraph
python train.py --config configs/cfg/dfine-s-hyper.yaml
```

**预期**: AP提升1.5~2.5%

### 实验2: 阈值敏感性
```python
# 测试不同阈值: [6, 8, 10, 12]
thresholds = [6, 8, 10, 12]
for t in thresholds:
    # 修改yaml中的threshold参数
    train(threshold=t)
```

### 实验3: 插入位置对比
```yaml
# 方案A: encoder后（推荐）
- [[16, 19, 22], HyperGraphEnhance, [...]]

# 方案B: 每层后（轻量版）
- [-1, HyperGraphEnhanceLite, [...]]

# 方案C: decoder前 + encoder中
- [[5, 6, 7], HyperGraphEnhance, [...]]  # backbone输出
```

### 实验4: 与其他Neck对比
```bash
# vs Gold-YOLO neck
# vs PANet
# vs BiFPN
```

---

## 🐛 常见问题

### Q1: CUDA out of memory
**原因**: 距离矩阵计算占用大量显存

**解决方案**:
1. 减小target_size: 20→15
2. 增大threshold: 减少超边数量
3. 使用梯度检查点:
```python
self.hyper_compute = torch.utils.checkpoint.checkpoint_sequential(...)
```

### Q2: 训练速度慢
**原因**: torch.cdist在GPU上效率不高

**解决方案**:
1. 使用混合精度训练(AMP)
2. 减小batch_size
3. 只在关键epoch使用超图（warmup策略）

### Q3: 精度没有提升
**可能原因**:
1. 阈值设置不当
2. 训练不充分
3. 数据增强过强

**调试建议**:
1. 先在小数据集验证
2. 可视化超图结构
3. 打印特征范数变化

---

## 📚 参考资料

**论文**:
- [Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation](https://arxiv.org/abs/2408.04804)
- IEEE TPAMI 2025

**代码**:
- [官方实现](https://github.com/iMoonLab/Hyper-YOLO)

**核心概念**:
- HGC-SCS框架
- ε-ball超图构建
- 两阶段超图卷积

---

## 📈 进阶优化方向

1. **稀疏超图表示**
```python
# 只存储距离<threshold的边
sparse_H = H[H > 0]  # 节省50%+内存
```

2. **动态阈值**
```python
# 根据特征分布自适应调整
threshold = mean_distance + std_distance
```

3. **多头超图**
```python
# 类似multi-head attention
hypergraphs = [build_hypergraph(x, t) for t in thresholds]
```

4. **与Transformer融合**
```python
# 超图增强 + Self-Attention
x = HyperGraph(x) + Attention(x)
```

---

## ✅ TODO

- [ ] 添加可视化工具（超图结构、特征图对比）
- [ ] 实现TensorRT优化版本
- [ ] 多数据集验证（VOC, Objects365）
- [ ] 与其他检测器集成（YOLOv8, DINO）
- [ ] 设计更高效的超图构建算法

---

## 👥 贡献者

基于Hyper-YOLO (TPAMI 2025) 实现，适配DFINE架构。

**Hyper-YOLO原作者**:
- Yifan Feng, Tsinghua University
- [论文链接](https://arxiv.org/abs/2408.04804)

---

## 📄 License

本实现遵循原项目的开源协议。
