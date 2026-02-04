# HyperGraph Enhancement 模块论文准备清单

## 1. 模块命名建议 ⭐

### 当前名称问题
- `HyperGraphEnhance` 过于通用，未体现穗类检测的特定应用场景
- 缺乏与田间实际问题（密度、遮挡、跨域）的关联

### 推荐命名方案

#### **方案A（推荐）: MSIA (Multi-Scale Interactive Aggregation)**
- **英文全称**: Multi-Scale Interactive Aggregation  
- **中文译名**: 多尺度交互聚合模块
- **缩写**: MSIA
- **优势**:
  - 强调多尺度特征融合（针对穗类多尺度密度变化）
  - "Interactive"体现跨层级跨位置的高阶交互
  - 简洁易读，符合学术命名习惯
- **田间动机**: 田间穗类呈现多尺度分布（近景密集、远景稀疏），需要跨尺度的信息交互来适应不同生长密度

#### 方案B: CLIA (Cross-Level Interactive Aggregation)
- **英文全称**: Cross-Level Interactive Aggregation
- **中文译名**: 跨层级交互聚合模块
- **优势**: 直接对应Hyper-YOLO的"Cross-Level and Cross-Position"核心
- **劣势**: 偏长，不如MSIA简洁

#### 方案C: HOF (High-Order Fusion)
- **英文全称**: High-Order Fusion Module
- **中文译名**: 高阶融合模块  
- **优势**: 直接体现超图的高阶消息传递能力
- **劣势**: 过于抽象，缺少对多尺度问题的明确指向

### LaTeX宏定义（基于方案A）
```latex
\newcommand{\MSIA}{MSIA }
\newcommand{\MSIAfull}{Multi-Scale Interactive Aggregation }
\newcommand{\MSIAcn}{多尺度交互聚合模块 }
```

---

## 2. 田间实际动机与技术映射 🌾

### 核心挑战 → 技术方案映射

#### **挑战1: 多尺度密度差异（Density Variation Across Scales）**

**田间现象**:
- **近景密集场景**（如GWHD的UQ_8: 117.9个/图，Ukyoto_1: 14.3%高遮挡）
  - 穗与穗之间距离极小（<5像素）
  - P3(8stride)高分辨率特征能捕获边界，但感受野不足
  - P5(32stride)感受野大，但分辨率损失边界信息
  
- **远景稀疏场景**（如Terraref_2: 12个/图）
  - 穗之间距离大（>50像素）
  - P3特征含大量冗余背景
  - P5特征丢失小目标细节

**技术解决方案 - MSIA的作用**:
```
语义收集阶段 (Semantic Collecting):
  - 统一[P3, P4, P5]到中间尺度(40×40)
  - 通过AdaptivePool保留各尺度的关键语义
  - 拼接融合: [B,768,40,40] → [B,256,40,40]
  
超图计算阶段 (Hypergraph Computation):
  - 构建ε-ball超边(threshold=6): 连接语义相近的特征点
  - 打破grid结构: 允许P3的边界特征与P5的上下文特征直接交互
  - 两阶段消息传递(V→E→V): 高阶聚合跨尺度信息
  
语义散射阶段 (Semantic Scattering):
  - 将增强后的统一特征分发回[80×80, 40×40, 20×20]
  - 残差连接(weight=0.5): 保留原始特征的同时注入高阶关联
```

**定量支撑**（需补充到论文）:
- 在UQ_8(密集)域: AP提升预计+3.5% (从baseline的XX%到XX%)
- 在Terraref_2(稀疏)域: AP提升预计+2.8%
- 小目标AP_small提升: +2~4% (MSIA增强P3特征)

---

#### **挑战2: 跨域泛化不足（Domain Generalization）**

**田间现象**:
- **18个OOD测试域**: Sudan干旱土壤、Mexico长芒品种、France高密度
- **训练集偏欧洲**: 76%数据来自Europe_1/2/3，颜色/纹理单一
- **PANet的局限**: 只在相邻层融合，无法捕获跨域共性特征

**技术解决方案**:
```
超图跨域建模能力:
  - ε-ball基于语义距离(欧氏距离)而非空间位置
  - Sudan域的"干燥麦穗特征"可与Europe的"绿色背景麦穗"建立超边
  - 高阶传播: 跨域共性特征(如穗的纹理)在语义空间聚合
  
与ASWB的协同:
  - ASWB: 自适应场景密度(波动传播)
  - MSIA: 跨尺度高阶关联(超图聚合)
  - 联合效果: 既适应密度变化，又捕获跨域结构不变性
```

**数据集证据**（需补充实验）:
- WDA (Weighted Domain Accuracy)指标: 预计从baseline的XX%提升到XX%
- Per-domain分析: Sudan/Mexico等极端域AP提升>3%

---

#### **挑战3: 密集场景下的遮挡与重叠（Occlusion in Dense Scenes）**

**田间现象**:
- Ukyoto_1: 14.3%高遮挡率
- 密集场景: 部分麦穗只露出1/3或1/2
- 传统FPN: 被遮挡的穗无法从单一尺度获得完整信息

**技术解决方案**:
```
跨位置高阶交互:
  - 超图构建: 遮挡穗的可见部分与邻近完整穗建立超边
  - 消息传递: 从完整穗传播结构先验到遮挡穗
  - 残差融合: 补全遮挡区域的语义信息
  
与GO-LSD的协同:
  - GO-LSD: 精细化分布细化(定位准确)
  - MSIA: 补全遮挡信息(特征完整)
  - 联合效果: 既定位准确，又特征丰富
```

---

## 3. 可视化需求 📊

### 图1: MSIA架构图（Method部分，高优先级）

**目标**: 清晰展示三阶段框架与超图构建

**设计要求**:
```
三个子模块水平排列:
┌─────────────────────────────────────────────────┐
│ (a) Semantic Collecting                         │
│  P3(80×80) ─┐                                   │
│  P4(40×40) ─┼→ AdaptivePool → Concat → Conv1×1│
│  P5(20×20) ─┘        ↓                          │
│              [B, 256, 40, 40]                   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ (b) Hypergraph Computation                      │
│  ε-ball构建 → Incidence Matrix H                │
│  展示超边连接(不同颜色表示不同超边)            │
│  V→E→V消息传递示意图                            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ (c) Semantic Scattering                         │
│  统一特征 → Interpolate + ResConv → 残差融合    │
│     ↓            ↓            ↓                  │
│  P3'(80×80)  P4'(40×40)  P5'(20×20)            │
└─────────────────────────────────────────────────┘
```

**绘图工具**: 
- PowerPoint/Keynote (初稿)
- draw.io (矢量图)
- TikZ (LaTeX，最终版)

**配色方案**:
- P3特征: 蓝色系（高分辨率）
- P4特征: 绿色系（中间）
- P5特征: 橙色系（大感受野）
- 超边: 红色虚线

---

### 图2: 超图ε-ball构建示意图（Method部分，中优先级）

**目标**: 可视化超边如何连接不同尺度/位置的特征点

**内容**:
```
左侧: 传统Grid结构
- 展示P3/P4/P5的grid feature map
- 标注相邻层级才能交互的限制

右侧: 超图语义空间
- 40×40=1600个特征点散点图
- 圈出3-5个超边(ε-ball)
- 用箭头标注跨尺度连接:
  * P3的边界点 ↔ P5的上下文点
  * 不同位置的语义相似点
```

**数据来源**:
- 使用GWHD的UQ_8域某张图
- 提取encoder输出的[P3, P4, P5]特征
- PCA降维到2D并可视化距离矩阵

**实现脚本** (需创建):
```bash
tools/visualization/hypergraph_ball.py
  --config configs/cfg/dfine-s-hypergraph.yaml
  --checkpoint outputs/dfine-s-hypergraph/best.pth
  --image data/gwhd_2021/images/UQ_8_001.jpg
  --save_path figures/hypergraph_ball.pdf
```

---

### 图3: 密集vs稀疏场景的超图对比（Experiments部分，中优先级）

**目标**: 验证MSIA在不同密度下的自适应能力

**设计**:
```
2×3网格布局:
┌──────────────┬──────────────┬──────────────┐
│  原图        │  密度heatmap │  超边可视化  │
├──────────────┼──────────────┼──────────────┤
│ UQ_8(密集)  │   117.9/图   │  密集连接    │
│ Terraref(疏)│   12/图      │  稀疏连接    │
└──────────────┴──────────────┴──────────────┘

超边可视化:
- 用线条连接被超边连接的特征点
- 线条粗细表示权重(距离越近越粗)
- 密集场景: 局部密集+跨尺度连接
- 稀疏场景: 长距离连接占主导
```

---

### 图4: 特征图对比（Ablation部分，中优先级）

**目标**: 对比Baseline PANet vs. MSIA的特征表示

**内容**:
```
3×4网格:
┌────────┬─────────┬─────────┬─────────┐
│        │  P3     │  P4     │  P5     │
├────────┼─────────┼─────────┼─────────┤
│原图    │  [输入图像]                  │
├────────┼─────────┼─────────┼─────────┤
│Baseline│ [灰度] │ [灰度]  │ [灰度]  │
│PANet   │ 边界弱 │ 中等    │ 模糊    │
├────────┼─────────┼─────────┼─────────┤
│+MSIA   │ [彩色] │ [彩色]  │ [彩色]  │
│        │ 边界强 │ 增强    │ 清晰    │
└────────┴─────────┴─────────┴─────────┘
```

**生成方法**:
```python
# tools/visualization/feature_map_comparison.py
# 使用GradCAM或直接可视化某个通道
# 对比+MSIA前后的P3/P4/P5特征激活
```

---

### 图5 (可选): 跨域泛化性能雷达图

**目标**: 展示MSIA在18个测试域的提升

**内容**:
```
雷达图(18个轴):
- 每个轴代表一个测试域(如Sudan, Mexico, France等)
- Baseline曲线(虚线)
- +MSIA曲线(实线)
- 突出极端域(Sudan, Mexico)的大幅提升
```

---

## 4. 主实验表格 📈

### 表1: GWHD主实验结果（与baseline和ASWB对比）

**目标**: 验证MSIA单独作用和联合ASWB的效果

| Method | AP | AP50 | AP75 | AP_small | WDA | Params | FLOPs |
|--------|-----|------|------|----------|-----|--------|-------|
| DFINE-S (Baseline) | XX.X | XX.X | XX.X | XX.X | XX.X | 12.0M | XX.XG |
| + ASWB | XX.X | XX.X | XX.X | XX.X | XX.X | 13.0M | XX.XG |
| + MSIA | **XX.X** | XX.X | XX.X | **XX.X** | **XX.X** | 13.5M | XX.XG |
| + ASWB + MSIA | **XX.X** | **XX.X** | **XX.X** | **XX.X** | **XX.X** | 14.5M | XX.XG |

**重点指标**:
- AP: 整体检测精度
- AP_small: 小目标提升（验证多尺度融合）
- WDA: 跨域泛化能力（18个测试域加权平均）

**预期结果**:
- +MSIA: AP提升1.5~2.0%，AP_small提升2.5~3.5%
- +ASWB+MSIA: 联合提升3~4%

---

### 表2: DRPD验证实验（可选，如果DRPD实验完成）

| Method | AP | AP50 | AP_GSD7 | AP_GSD12 | AP_GSD20 | Params |
|--------|-----|------|---------|----------|----------|--------|
| Baseline | XX.X | XX.X | XX.X | XX.X | XX.X | 12.0M |
| + MSIA | **XX.X** | **XX.X** | **XX.X** | **XX.X** | **XX.X** | 13.5M |

**说明**: 验证MSIA在不同高度/尺度(GSD)的稻穗检测中的泛化能力

---

### 表3: Per-Domain分析（补充材料或正文）

**目标**: 详细展示MSIA在18个测试域的提升

| Domain | Density | Occlusion | Baseline AP | +MSIA AP | Δ AP |
|--------|---------|-----------|-------------|----------|------|
| **Dense Domains** |  |  |  |  |  |
| UQ_8 | 117.9 | 12.5% | XX.X | **XX.X** | **+X.X** |
| Ukyoto_1 | 89.3 | 14.3% | XX.X | **XX.X** | **+X.X** |
| ... | ... | ... | ... | ... | ... |
| **Sparse Domains** |  |  |  |  |  |
| Terraref_2 | 12.0 | 2.1% | XX.X | **XX.X** | **+X.X** |
| ... | ... | ... | ... | ... | ... |
| **Extreme OOD** |  |  |  |  |  |
| Sudan (arid) | 45.6 | 8.7% | XX.X | **XX.X** | **+X.X** |
| Mexico (long awn) | 52.3 | 9.2% | XX.X | **XX.X** | **+X.X** |

**分析重点**:
- 密集域: 验证跨尺度融合+遮挡处理
- 稀疏域: 验证小目标检测增强
- 极端OOD域: 验证跨域泛化能力

---

## 5. 消融实验 🔬

### 消融1: 超图阈值(threshold)敏感性分析

**当前配置**: threshold=6 (基于Hyper-YOLO-N的轻量设计)

**实验设置**:

| threshold | 语义 | 超边密度 | 计算复杂度 | 预期AP |
|-----------|------|----------|-----------|--------|
| 4 | 极稀疏 | 低 | 低 | XX.X (可能欠拟合) |
| 6 | 稀疏（推荐） | 中等 | 中等 | **XX.X** |
| 8 | 中等密集 | 高 | 高 | XX.X |
| 10 | 密集 | 很高 | 很高 | XX.X (可能过拟合) |

**技术原理**:
- **threshold↓**: 超边只连接最近邻，局部建模强，但缺乏全局交互
- **threshold↑**: 超边连接更多远距离点，全局交互强，但噪声增加

**田间映射**:
- 密集场景(UQ_8): 推荐threshold=6~8（需要局部精细建模）
- 稀疏场景(Terraref): 推荐threshold=8~10（需要长距离关联）

**实验命令**:
```bash
# 修改configs/cfg/dfine-s-hypergraph.yaml的threshold参数
for t in 4 6 8 10; do
  python train.py --config configs/cfg/dfine-s-hypergraph.yaml \
    --override encoder.-1.2=$t \
    --output_dir outputs/ablation_threshold_$t
done
```

---

### 消融2: 残差权重(residual_weight)分析

**当前配置**: residual_weight=0.5

**实验设置**:

| residual_weight | 语义 | 原始特征占比 | 超图特征占比 | 预期AP |
|-----------------|------|--------------|--------------|--------|
| 0.3 | 保守 | 70% | 30% | XX.X |
| 0.5 | 平衡（推荐） | 50% | 50% | **XX.X** |
| 0.7 | 激进 | 30% | 70% | XX.X |
| 1.0 | 纯超图 | 0% | 100% | XX.X (不稳定) |

**技术原理**:
- **weight↓**: 更依赖原始PANet特征，超图作为辅助增强
- **weight↑**: 更信任超图高阶信息，可能丢失原始局部细节

**联合消融**: threshold × residual_weight

| Config | threshold | weight | AP | AP_small | 训练稳定性 |
|--------|-----------|--------|-----|----------|-----------|
| A | 6 | 0.5 | **XX.X** | **XX.X** | ✅ 稳定 |
| B | 6 | 0.7 | XX.X | XX.X | ⚠️ 需warmup |
| C | 8 | 0.5 | XX.X | XX.X | ✅ 稳定 |
| D | 8 | 0.7 | XX.X | XX.X | ⚠️ 需warmup |

**推荐配置**:
- **DFINE-N/S**: threshold=6, weight=0.5 (平衡)
- **DFINE-M/L**: threshold=8, weight=0.7 (更强表达)

---

### 消融3: 统一尺寸(target_size)分析

**当前配置**: target_size=40 (使用P4中间尺度)

**实验设置**:

| target_size | 特征点数 | 距离矩阵大小 | 内存占用 | AP | 推理速度 |
|-------------|----------|--------------|----------|-----|---------|
| 20 (P5) | 1,200 | 1.44M | 低 | XX.X | 快 |
| 40 (P4) | 4,800 | 23.04M | 中等 | **XX.X** | 中等 |
| 80 (P3) | 19,200 | 368.64M | 极高 | XX.X | 慢 |

**技术权衡**:
- **target_size=20**: 省内存，但P3细节损失严重（小目标AP↓）
- **target_size=40**: 平衡，保留P3部分细节，P5上采样质量可接受
- **target_size=80**: 保细节，但计算爆炸（距离矩阵23M→369M，增长16倍）

**推荐策略**:
- 如果GPU内存<16GB: target_size=20
- 如果GPU内存≥24GB: target_size=40（推荐）

---

### 消融4: 模块有效性验证（与其他Neck对比）

**目标**: 验证MSIA相比其他多尺度融合方法的优势

| Neck Type | 跨层级 | 跨位置 | 高阶 | AP | AP_small | Params |
|-----------|--------|--------|------|-----|----------|--------|
| PANet (YOLOv8) | ✅ 相邻 | ❌ | ❌ | XX.X | XX.X | 12.0M |
| BiFPN | ✅ 加权 | ❌ | ❌ | XX.X | XX.X | 12.5M |
| Gold-YOLO | ✅ 全连接 | ❌ | ❌ | XX.X | XX.X | 13.2M |
| MSIA (Ours) | ✅ 全连接 | ✅ | ✅ | **XX.X** | **XX.X** | 13.5M |

**核心差异**:
- PANet: 只能U→D→U，信息流受限
- Gold-YOLO: Gather-Distribute打破层级限制，但仍是grid结构
- MSIA: 超图打破grid，允许任意位置的语义相似点交互

---

## 6. 论文叙事结构 📝

### Introduction中的引入（~2段）

**第一段 - 挑战描述**:
```
穗类检测面临多尺度密度差异：田间图像中，近景呈现密集分布（如GWHD的UQ_8域
达到117.9个麦穗/图），远景则极为稀疏（Terraref_2域仅12个/图），密度跨度达
9.8倍。传统的特征金字塔网络（FPN）及其变体（如PANet）虽能融合多尺度信息，
但其相邻层级融合的设计限制了跨尺度长距离关联的捕获。在密集场景中，P3高分
辨率特征能够捕捉边界细节，但感受野不足以理解整体布局；而P5大感受野特征虽
能把握全局分布，但分辨率损失导致边界模糊。这种尺度间信息割裂的问题在处理
极端密度变化时尤为突出，影响了模型在真实田间环境中的泛化能力。
```

**第二段 - MSIA引入**:
```
为解决上述问题，本文引入\MSIA（\MSIAfull），一种基于超图计算的多尺度特征
增强模块。不同于传统的网格结构特征融合，\MSIA通过将多尺度特征映射到统一的
语义空间，并构建基于语义距离的超图结构，实现了跨层级、跨位置的高阶信息传递。
具体而言，\MSIA包含三个阶段：(1)语义收集：将P3/P4/P5特征统一到中间尺度；
(2)超图计算：基于ε-ball构建超边，通过两阶段消息传递（V→E→V）捕获高阶关联；
(3)语义散射：将增强后的特征分发回原始尺度。这种设计使得密集场景中的边界细节
特征能够与大感受野上下文特征直接交互，同时在稀疏场景中实现长距离语义关联，
从而有效应对多尺度密度变化带来的挑战。
```

---

### Method部分结构

#### 3.X MSIA: Multi-Scale Interactive Aggregation

**3.X.1 动机（Motivation）**:
```
传统Neck的局限性分析：
- PANet: 只能相邻层交互，信息流路径长(P3→P4→P5需2跳)
- Gold-YOLO: Gather-Distribute统一语义，但缺少跨位置交互
- 穗类检测的特殊需求：密集场景需要P3细节+P5上下文，稀疏场景需要跨位置聚合

引入超图的理由：
- 超边可连接2+节点，天然适合多尺度融合
- 基于语义距离构建，打破grid空间限制
- 高阶消息传递能力强于pairwise attention
```

**3.X.2 整体框架（Overall Framework）**:
```
描述HGC-SCS三阶段：
1. Semantic Collecting: 多尺度统一策略
2. Hypergraph Computation: ε-ball构建+两阶段卷积
3. Semantic Scattering: 残差分发机制

公式:
\begin{equation}
\left\{
\begin{aligned}
    & \mathbf{X}_{mixed} = \text{Concat}(\text{Pool}(P_3), \text{Pool}(P_4), \text{Pool}(P_5)) \\
    & \mathbf{X}_{hyper} = \text{HyperConv}(\mathbf{X}_{mixed}, \mathbf{H}) \\
    & P'_i = P_i + \alpha \cdot \text{Interp}(\mathbf{X}_{hyper})
\end{aligned}
\right.
\end{equation}
```

**3.X.3 超图构建（Hypergraph Construction）**:
```
详细描述:
- ε-ball定义: ball(v, ε) = {u | ||x_u - x_v|| < ε}
- 距离计算: 欧氏距离 vs 余弦距离
- 阈值选择: threshold=6 (基于DFINE-S的参数量权衡)
- 稀疏化策略: 只存储ε内的边(节省内存)

附图: 图2 超图ε-ball构建示意图
```

**3.X.4 超图卷积（Hypergraph Convolution）**:
```
两阶段消息传递公式:
\begin{equation}
\left\{
\begin{aligned}
    & \mathbf{x}_e = \frac{1}{|\mathcal{N}_v(e)|} \sum_{v \in \mathcal{N}_v(e)} \mathbf{x}_v \mathbf{\Theta} \\
    & \mathbf{x}'_v = \mathbf{x}_v + \frac{1}{|\mathcal{N}_e(v)|} \sum_{e \in \mathcal{N}_e(v)} \mathbf{x}_e
\end{aligned}
\right.
\end{equation}

矩阵形式:
\text{HyperConv}(\mathbf{X}, \mathbf{H}) = \mathbf{X} + \mathbf{D}_v^{-1} \mathbf{H} \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta}
```

**3.X.5 与ASWB的协同（Synergy with ASWB）**:
```
两个模块的互补性:
- ASWB: 自适应场景密度(波动传播) → 时域动态
- MSIA: 跨尺度高阶关联(超图聚合) → 空域结构
- 联合作用: 既适应密度变化，又捕获结构不变性

Pipeline:
Backbone → ASWB(P5层) → Neck → MSIA([P3,P4,P5]) → Decoder
```

---

### Experiments部分结构

#### 4.X MSIA的实验验证

**4.X.1 主实验结果（表1）**:
```
对比Baseline, +ASWB, +MSIA, +ASWB+MSIA
重点分析:
- AP整体提升
- AP_small大幅提升(验证多尺度融合)
- WDA提升(验证跨域泛化)
```

**4.X.2 Per-Domain分析（表3）**:
```
详细分析18个测试域:
- 密集域(UQ_8, Ukyoto_1): 提升归因于跨尺度融合
- 稀疏域(Terraref_2): 提升归因于长距离关联
- 极端OOD域(Sudan, Mexico): 提升归因于超图结构不变性
```

**4.X.3 消融实验**:
```
4.X.3.1 超图阈值敏感性(表4)
4.X.3.2 残差权重分析(表5)
4.X.3.3 统一尺寸对比(表6)
4.X.3.4 与其他Neck对比(表7)
```

**4.X.4 可视化分析**:
```
图3: 密集vs稀疏场景的超图对比
图4: 特征图对比(Baseline vs +MSIA)
```

---

## 7. 代码补充清单 💻

### 7.1 可视化脚本

**脚本1**: `tools/visualization/hypergraph_ball.py`
```python
# 功能: 可视化超图ε-ball构建过程
# 输入: checkpoint, image, config
# 输出: 超边连接示意图(PDF)
# 实现要点:
#   - 提取[P3,P4,P5]特征
#   - PCA降维到2D
#   - 计算距离矩阵并绘制超边
```

**脚本2**: `tools/visualization/feature_map_comparison.py`
```python
# 功能: 对比+MSIA前后的特征图
# 输入: baseline_ckpt, msia_ckpt, image
# 输出: 3×4网格对比图
# 实现要点:
#   - hook提取P3/P4/P5特征
#   - 选择响应最强的通道
#   - 生成heatmap覆盖原图
```

**脚本3**: `tools/visualization/density_heatmap.py`
```python
# 功能: 绘制密度与超边数量的关系
# 输入: 测试集所有图像
# 输出: 散点图(密度 vs 超边数)
# 实现要点:
#   - 统计每张图的目标密度
#   - 计算超图的平均超边数
#   - 拟合密度-超边关系曲线
```

---

### 7.2 消融实验配置

**配置1**: `configs/ablation/hypergraph_threshold.yaml`
```yaml
# 修改threshold参数
encoder:
  - [[16, 19, 22], HyperGraphEnhance, [256, {threshold}, 40, 0.5]]
  # threshold遍历: [4, 6, 8, 10]
```

**配置2**: `configs/ablation/hypergraph_residual.yaml`
```yaml
# 修改residual_weight参数
encoder:
  - [[16, 19, 22], HyperGraphEnhance, [256, 6, 40, {weight}]]
  # weight遍历: [0.3, 0.5, 0.7, 1.0]
```

**配置3**: `configs/ablation/hypergraph_target_size.yaml`
```yaml
# 修改target_size参数
encoder:
  - [[16, 19, 22], HyperGraphEnhance, [256, 6, {size}, 0.5]]
  # size遍历: [20, 40, 80]
```

---

### 7.3 评估脚本

**脚本**: `tools/eval_per_domain.py`
```python
# 功能: 计算18个测试域的per-domain AP
# 输入: checkpoint, GWHD test set
# 输出: CSV表格(domain, density, AP, AP50, ...)
# 实现要点:
#   - 读取stage_occlusion_stats.json
#   - 按domain分组评估
#   - 计算WDA加权平均
```

---

## 8. 文献补充清单 📚

### 核心参考文献（必须引用）

1. **Hyper-YOLO原论文**:
   ```bibtex
   @article{feng2025hyper,
     title={Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation},
     author={Feng, Yifan and Huang, Jiangang and Du, Shaoyi and Ying, Shihui and Yong, Jun-Hai and Li, Yipeng and Ding, Guiguang and Ji, Rongrong and Gao, Yue},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
     year={2025}
   }
   ```

2. **超图神经网络基础**:
   - HGNN (AAAI 2019): 超图卷积基础理论
   - HGNN+ (TNNLS 2022): 改进的超图卷积

3. **多尺度特征融合**:
   - FPN (CVPR 2017): 特征金字塔网络
   - PANet (CVPR 2018): 路径聚合网络
   - BiFPN (CVPR 2020): 双向特征金字塔
   - Gold-YOLO (NeurIPS 2023): Gather-Distribute机制

4. **目标检测中的高阶关联**:
   - Relation Networks (CVPR 2018): 关系建模
   - Self-Attention in Detection (ECCV 2020): 注意力机制

---

## 9. 优先级与时间线 ⏰

### 高优先级（Week 1-2，实验阶段）

**必须完成**:
- ✅ 确定模块命名: MSIA
- ⏳ 训练完成: dfine-s-hypergraph.yaml on GWHD
- ⏳ 主实验结果: 表1 (Baseline vs +ASWB vs +MSIA vs Joint)
- ⏳ 消融实验1: threshold敏感性 [4,6,8,10]
- ⏳ 消融实验2: residual_weight [0.3,0.5,0.7]

**可视化**:
- ⏳ 图1: MSIA架构图（三阶段框架）
- ⏳ 图3: 密集vs稀疏场景超图对比

---

### 中优先级（Week 3，写作阶段）

**实验补充**:
- ⏳ Per-domain分析: 表3 (18个测试域详细结果)
- ⏳ 与其他Neck对比: 表7 (PANet/BiFPN/Gold-YOLO)

**可视化**:
- ⏳ 图2: 超图ε-ball构建示意图
- ⏳ 图4: 特征图对比

**论文写作**:
- ⏳ Introduction中引入MSIA (2段)
- ⏳ Method部分撰写 (3.X节，5页)
- ⏳ Experiments结果分析 (4.X节，3页)

---

### 低优先级（Week 4，润色阶段）

**可选实验**:
- ⏳ DRPD验证: 表2 (如果数据集准备好)
- ⏳ 统一尺寸消融: 表6 [20,40,80]
- ⏳ 跨域泛化雷达图: 图5

**代码整理**:
- ⏳ 可视化脚本完善
- ⏳ 消融配置文件规范化
- ⏳ README更新

---

## 10. 关键问题（需要你的反馈）❓

### 问题1: 实验状态确认
- **Q1.1**: dfine-s-hypergraph在GWHD上训练完成了吗？当前AP是多少？
- **Q1.2**: 是否有baseline的对比结果（dfine-s.yaml）？
- **Q1.3**: ASWB单独实验（dfine-s-wave.yaml）完成了吗？

### 问题2: 消融实验范围
- **Q2.1**: 4个消融配置[threshold×weight: (6,0.5), (6,0.7), (8,0.5), (8,0.7)]是否足够？
  - 我的建议: 增加threshold=4和10的极端情况，共6个配置
- **Q2.2**: 是否有GPU资源训练target_size=[20,40,80]的消融？
  - 如果内存不足，可以只做[20,40]

### 问题3: DRPD数据集
- **Q3.1**: DRPD数据集准备好了吗（yolo2coco转换完成）？
- **Q3.2**: 是否需要在DRPD上验证MSIA的泛化能力？
  - 建议: 如果时间允许，至少做一次DRPD实验（表2）

### 问题4: 可视化偏好
- **Q4.1**: 绘图工具偏好？PowerPoint初稿 + TikZ最终版 可以吗？
- **Q4.2**: 是否有现成的样例图像（UQ_8密集 + Terraref稀疏）？
- **Q4.3**: 特征图可视化需要哪个层？建议P4(40×40)最清晰

### 问题5: 命名确认
- **Q5.1**: 对MSIA命名满意吗？如果不满意，方案B(CLIA)或方案C(HOF)也可以
- **Q5.2**: LaTeX宏需要调整吗？（如\MSIA → \MSIAblock）

### 问题6: 论文投稿目标
- **Q6.1**: 目标期刊/会议？TPAMI/TIP/CVPR/ECCV？
- **Q6.2**: Deadline紧急程度？决定实验优先级
- **Q6.3**: 页数限制？决定可选内容取舍

### 问题7: 第三个改进
- **Q7.1**: 第三个改进是什么？也需要提前了解以规划整体叙事
- **Q7.2**: 三个改进的联合实验（ASWB+MSIA+Improvement3）计划了吗？

---

## 11. 下一步行动建议 🎯

### 推荐工作流程

**选项A: 继续技术理解（推荐）**
```
现在 → 介绍第三个改进（Improvement 3）
     → 建立完整技术图景（ASWB + MSIA + ???）
     → 规划三者的协同叙事
然后 → 回答TODO.md的关键问题
     → 明确实验状态和资源
最后 → 并行执行实验和写作
```

**选项B: 立即开始实验**
```
如果主实验已完成:
  → 执行消融实验（threshold, residual_weight）
  → 生成可视化图表
  → 边实验边写Method和Experiments部分

如果主实验未完成:
  → 先训练dfine-s-hypergraph
  → 我可以同时准备Method部分的文本
```

**选项C: 先完成ASWB部分**
```
如果三个改进相对独立:
  → 先完成ASWB的实验和写作
  → 再完成MSIA的实验和写作
  → 最后完成Improvement 3
  → 最后写联合实验部分

优势: 各个击破，每个模块独立成章
劣势: 可能缺少整体叙事的连贯性
```

---

## 📌 总结

MSIA模块本质上是**将Hyper-YOLO的HGC-SCS框架应用到穗类检测的多尺度密度变化场景**。核心贡献在于：

1. **技术创新**: 超图高阶建模 + 跨尺度跨位置交互
2. **应用创新**: 针对穗类检测的密度差异、遮挡、跨域泛化问题
3. **工程优化**: 轻量化设计(threshold=6, target_size=40)适配DFINE-S

完成论文的关键路径：
1. **实验**: 主实验+消融(threshold, weight) → 2周
2. **可视化**: 架构图+超图示意+特征对比 → 1周
3. **写作**: Method(5页) + Experiments(3页) → 1周

**请先告诉我：你想继续介绍第三个改进，还是先回答上述关键问题？**
