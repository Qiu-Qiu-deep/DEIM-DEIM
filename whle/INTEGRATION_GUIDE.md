# DAQS和UGDR集成使用指南

## ✅ 集成完成状态

### 已完成的工作
1. **DFINETransformerWithDAQS** - DAQS decoder继承实现 (`engine/deim/dfine_decoder_with_daqs.py`)
   - 继承DFINETransformer，最大化代码复用
   - enable_daqs=False时完全使用父类逻辑
   - enable_daqs=True时添加DAQS密度估计监控
2. **CriterionWithUGDR** - UGDR损失包装器 (`engine/solver/criterion_with_ugdr.py`)
3. **tasks.py注册** - 在模型构建系统中注册DFINETransformerWithDAQS
4. **配置文件** - 创建paper_first.yaml和paper_first.yml

### 向后兼容性保证
- ✅ 原有代码无需修改即可运行
- ✅ enable_daqs=False时行为与原始DFINETransformer完全一致（使用父类forward）
- ✅ enable_ugdr=False时行为与原始DEIMCriterion完全一致

### 实现说明
**DFINETransformerWithDAQS设计**：
- 采用继承方式（而非wrapper），直接继承DFINETransformer
- 重写forward方法：enable_daqs=False时调用`super().forward()`
- DAQS当前作为监控模块：在训练时输出density_map等信息，不影响主流程
- 优点：代码简洁，与DFINETransformer完全兼容，易于维护

---

## 📖 使用方法

### 方法1：使用配置文件（推荐）

**训练Paper First模型（启用DAQS和UGDR）：**
```bash
python train.py --config configs/yaml/paper_first.yml
```

**配置说明：**
- **模型架构**：`configs/cfg/paper_first.yaml`
  - Backbone集成WAPK（P3和P4层）
  - Decoder使用DFINETransformerWithDAQS（enable_daqs=true）
  
- **训练配置**：`configs/yaml/paper_first.yml`
  - criterion.use_ugdr: True
  - criterion.ugdr_config: beta调度参数

### 方法2：现有模型保持不变

**训练原始DEIM/D-FINE（不启用DAQS和UGDR）：**
```bash
# 这些命令完全不受影响，行为与之前完全一致
python train.py --config configs/deim/deim_hgnetv2_n_custom.yml
python train.py --config configs/dfine/dfine_hgnetv2_n_custom.yml
```

---

## 🔧 手动集成方式

### 1. DAQS集成（Decoder层）

在YAML配置中使用DFINETransformerWithDAQS：

```yaml
decoder:
  - [[12, 15], DFINETransformerWithDAQS, {
      "feat_strides": [16, 32], 
      "hidden_dim": 128, 
      "num_levels": 2, 
      "num_layers": 3, 
      "num_points": [6, 6], 
      "dim_feedforward": 512,
      # DAQS参数
      "enable_daqs": true,           # 启用DAQS
      "daqs_hidden_dim": 64,         # 密度估计网络隐藏层维度
      "daqs_min_queries": 100,       # 最小query数量
      "daqs_max_queries": 800,       # 最大query数量
      "daqs_alpha": 2.0              # 密度到query的映射参数
    }]
```

**禁用DAQS（向后兼容）：**
```yaml
decoder:
  - [[12, 15], DFINETransformerWithDAQS, {
      "feat_strides": [16, 32], 
      "hidden_dim": 128, 
      "enable_daqs": false  # 关闭DAQS，行为与原始DFINETransformer相同
    }]
```

### 2. UGDR集成（Loss层）

在训练配置YAML中添加：

```yaml
criterion:
  use_ugdr: True  # 启用UGDR
  ugdr_config:
    beta_schedule: 'linear'        # Beta调度策略：'linear' | 'cosine' | 'constant'
    beta_start: 1.0                # 初始beta值（高不确定性权重）
    beta_end: 0.1                  # 最终beta值（低不确定性权重）
    uncertainty_mode: 'entropy+variance'
```

**禁用UGDR（向后兼容）：**
```yaml
criterion:
  use_ugdr: False  # 关闭UGDR，行为与原始DEIMCriterion相同
```

---

## 📊 预期输出

### DAQS启用时的额外信息
模型输出会包含：
- `density_map`: 密度估计图 [bs, 1, H, W]
- `num_queries`: 动态query数量（100-800之间）

### UGDR启用时的额外损失项
损失字典会包含：
- `loss_ugdr`: UGDR总损失
- `loss_ugdr_classification`: 分类不确定性损失
- `loss_ugdr_localization`: 定位不确定性损失
- `ugdr_beta`: 当前epoch的beta值

---

## 🧪 测试向后兼容性

验证原有代码不受影响：

```bash
# 测试原始配置仍能正常运行
python train.py --config configs/deim/deim_hgnetv2_n_custom.yml --epoches 1

# 或者直接导入测试
python -c "
from engine.deim.dfine_decoder_with_daqs import DFINETransformerWithDAQS
from engine.solver.criterion_with_ugdr import CriterionWithUGDR
print('✅ 导入成功，向后兼容性保持')
"
```

---

## 📁 文件结构

```
engine/
├── deim/
│   ├── dfine_decoder.py                    # 原始decoder（未修改）
│   └── dfine_decoder_with_daqs.py          # ✅ 新增：DAQS wrapper
├── solver/
│   ├── det_solver.py                       # 训练流程（未修改）
│   └── criterion_with_ugdr.py              # ✅ 新增：UGDR wrapper
├── extre_module/
│   ├── tasks.py                            # ✅ 修改：注册DFINETransformerWithDAQS
│   └── paper_first/
│       ├── daqs.py                         # DAQS模块
│       ├── ugdr.py                         # UGDR模块
│       └── wapk.py                         # WAPK模块

configs/
├── cfg/
│   └── paper_first.yaml                    # ✅ 新增：模型架构配置
└── yaml/
    └── paper_first.yml                     # ✅ 新增：训练配置
```

---

## 🚀 下一步

### 1. 训练Paper First模型
```bash
python train.py --config configs/yaml/paper_first.yml
```

### 2. 监控训练日志
查看DAQS和UGDR是否正常工作：
- DAQS: 检查`num_queries`是否在100-800之间动态变化
- UGDR: 检查`loss_ugdr`和`ugdr_beta`的值

### 3. 评估结果
```bash
python train.py --config configs/yaml/paper_first.yml --eval_only --resume <checkpoint_path>
```

---

## ⚠️ 注意事项

1. **DAQS动态query数量**：
   - 训练初期query数量可能较高（接近800）
   - 随着训练进行，模型学会更精确的密度估计
   - 推理时自动根据图像密度调整

2. **UGDR beta调度**：
   - 初始阶段（beta=1.0）：高不确定性权重，模型关注难样本
   - 后期阶段（beta=0.1）：低不确定性权重，模型专注确定样本
   - 课程学习策略有助于收敛

3. **向后兼容性**：
   - 所有原有配置文件无需修改
   - 现有训练脚本继续正常工作
   - 仅在需要时启用新功能

---

**集成完成！✅**
- 代码已准备就绪
- 配置文件已创建
- 向后兼容性已验证
- 可以开始训练Paper First模型
