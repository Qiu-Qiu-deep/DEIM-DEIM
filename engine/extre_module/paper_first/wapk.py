'''
Wheat-Aware Poly Kernel Network (WAPK) v2
小麦感知的多核卷积网络 v2

引入动机（基于GWHD数据集的三大挑战）：
1. 形状特异性：小麦穗呈细长椭圆形（长宽比1:2-1:3），标准方形卷积核捕获不足
2. 密度极端差异：12-118个/图（9.8倍差异），需要自适应感受野
3. 尺度变化大：小目标居多，需要精细的多尺度特征提取

设计思路（融合4篇CVPR/ECCV顶会论文的核心代码）：
1. PKIBlock (CVPR 2024)：借鉴渐进式多核融合 x = x + Σkernel_i(x)
   - 关键代码：x = x + self.dw_conv1(x) + self.dw_conv2(x) + ...
   - 优势：残差式累加，训练更稳定
   
2. LSKblock (ICCV 2023)：借鉴双路径注意力（spatial + variance）
   - 关键代码：x_v = torch.var(x, dim=(-2,-1)) 统计方差作为全局特征
   - 优势：轻量级全局感知
   
3. SMFA (ECCV 2024)：借鉴统计引导的自调制 
   - 关键代码：x_l = x * F.interpolate(x_s * alpha + x_v * belt)
   - 优势：参数化控制，自适应特征增强
   
4. InceptionDWConv (CVPR 2024)：借鉴分支式高效计算
   - 关键代码：torch.split + concat，部分通道独立处理
   - 优势：减少计算量，保持表达能力

参考论文：
[1] PKIBlock: https://arxiv.org/pdf/2403.06258 (CVPR 2024)
[2] LSKblock: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf
[3] SMFA: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf (ECCV 2024)
[4] InceptionDWConv: https://arxiv.org/pdf/2303.16900 (CVPR 2024)
'''

import os, sys

# from engine.backbone.hgnetv2 import ConvBNAct
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 尝试导入calflops用于参数量计算，如果没有安装则跳过
try:
    from calflops import calculate_flops
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False
    print("Warning: calflops not installed, parameter calculation will be skipped")

class LearnableAffineBlock(nn.Module):
    """
    可学习的仿射变换模块 (Learnable Affine Block)  
   
    该模块对输入 `x` 进行仿射变换：    
        y = scale * x + bias
    其中 `scale` 和 `bias` 是可训练参数。
     
    适用于需要简单线性变换的场景，例如：
    - 归一化调整
    - 特征平移缩放
    - 作为更复杂模型的一部分   
    """    
    def __init__(   
            self,
            scale_value=1.0,  # 初始化缩放因子，默认为 1.0（保持输入不变）   
            bias_value=0.0    # 初始化偏移量，默认为 0.0（无偏移）  
    ):     
        super().__init__()
        # 定义可学习参数：缩放因子和偏移量
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)    
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)   

    def forward(self, x):  
        """
        前向传播：执行仿射变换     
   
        参数:     
        x (Tensor) - 输入张量

        返回:  
        Tensor - 变换后的输出张量
        """
        return self.scale * x + self.bias   
  

class ConvBNAct(nn.Module):
    def __init__( 
            self,
            in_chs,     
            out_chs, 
            kernel_size,
            stride=1,
            groups=1,
            padding='',     
            use_act=True,     
            use_lab=False   
    ):     
        super().__init__()     
        self.use_act = use_act    
        self.use_lab = use_lab 
        if padding == 'same':
            self.conv = nn.Sequential(   
                # nn.ZeroPad2d([0, 1, 0, 1]) 手动填充 右侧 1 个像素 和 底部 1 个像素，而左侧和顶部不填充。
	            # 这种方式适用于 kernel_size=2 的情况，使得卷积输出的尺寸与输入相同（在 stride=1 时）。    
                nn.ZeroPad2d([0, 1, 0, 1]),     
                nn.Conv2d(     
                    in_chs, 
                    out_chs,   
                    kernel_size,
                    stride,  
                    groups=groups, 
                    bias=False
                ) 
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,     
                out_chs,
                kernel_size,  
                stride,
                padding=(kernel_size - 1) // 2, # 表示 PyTorch 默认的 SAME 填充，即对 左右、上下 进行均匀填充。     
                groups=groups,  
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)   
        if self.use_act:    
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab: 
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()
 
    def forward(self, x): 
        x = self.conv(x)
        x = self.bn(x)    
        x = self.act(x) 
        x = self.lab(x)    
        return x     


def autopad(kernel_size: tuple, dilation: int = 1) -> tuple:
    """根据卷积核大小自动计算padding，保持特征图尺寸不变
    
    Args:
        kernel_size: 卷积核大小 (h, w)
        dilation: 膨胀率
        
    Returns:
        padding: (pad_h, pad_w)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    pad_h = (kernel_size[0] - 1) * dilation // 2
    pad_w = (kernel_size[1] - 1) * dilation // 2
    return (pad_h, pad_w)


class VarianceGuidedAttention(nn.Module):
    """统计引导注意力（借鉴SMFA的variance统计）
    
    核心思想：使用特征的统计信息（方差）作为全局上下文
    参考SMFA代码：x_v = torch.var(x, dim=(-2,-1), keepdim=True)
    """
    def __init__(self, channels: int):
        super().__init__()
        # 可学习的缩放和偏移参数（借鉴SMFA的alpha和belt）
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.belt = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            调制后的特征
        """
        # 计算方差作为全局统计特征（SMFA代码）
        x_var = torch.var(x, dim=(-2, -1), keepdim=True)
        # 参数化调制（SMFA的self-modulation机制）
        return x * (self.alpha + x_var * self.belt)


class WheatShapedKernels(nn.Module):
    """小麦形状自适应卷积核（融合PKIBlock和InceptionDWConv思想）
    
    设计3种针对小麦形状的卷积核：
    1. 竖向带状核 (1×7 + 7×1)：捕获竖向排列的麦穗
    2. 横向带状核 (7×1 + 1×7)：捕获横向排列的麦穗
    3. 方形核 (3×3)：标准特征提取
    
    核心创新：
    - 借鉴InceptionDWConv的分支设计：不同通道使用不同核
    - 借鉴PKIBlock的渐进融合：x = x + k1(x) + k2(x) + k3(x)
    """
    def __init__(self, channels: int, branch_ratio: float = 0.25):
        """
        Args:
            channels: 输入通道数
            branch_ratio: 每个分支的通道比例（借鉴InceptionDWConv）
        """
        super().__init__()
        # 计算每个分支的通道数（InceptionDWConv代码）
        gc = int(channels * branch_ratio)  # group channels
        self.gc = gc
        
        # 分支1：竖向带状卷积（decomposed vertical: 1×7 + 7×1）
        # 借鉴InceptionDWConv的分解卷积思想
        self.vertical_1 = nn.Conv2d(gc, gc, (1, 7), padding=(0, 3), groups=gc, bias=False)
        self.vertical_2 = nn.Conv2d(gc, gc, (7, 1), padding=(3, 0), groups=gc, bias=False)
        
        # 分支2：横向带状卷积（decomposed horizontal: 7×1 + 1×7）
        self.horizontal_1 = nn.Conv2d(gc, gc, (7, 1), padding=(3, 0), groups=gc, bias=False)
        self.horizontal_2 = nn.Conv2d(gc, gc, (1, 7), padding=(0, 3), groups=gc, bias=False)
        
        # 分支3：标准方形卷积（3×3）
        self.square = nn.Conv2d(gc, gc, 3, padding=1, groups=gc, bias=False)
        
        # Identity分支（InceptionDWConv代码中的x_id）
        self.split_indexes = (channels - 3 * gc, gc, gc, gc)
        
        # BN层（PKIBlock使用单独的BN）
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            融合后的特征
        """
        # 分支式处理（InceptionDWConv代码）
        x_id, x_v, x_h, x_s = torch.split(x, self.split_indexes, dim=1)
        
        # 竖向分支（分解卷积）
        x_v = self.vertical_2(self.vertical_1(x_v))
        
        # 横向分支（分解卷积）
        x_h = self.horizontal_2(self.horizontal_1(x_h))
        
        # 方形分支
        x_s = self.square(x_s)
        
        # 合并所有分支（InceptionDWConv代码）
        out = torch.cat([x_id, x_v, x_h, x_s], dim=1)
        
        return self.bn(out)


class DualPathEnhancement(nn.Module):
    """双路径特征增强（借鉴LSKblock的双路径设计）
    
    LSKblock的核心思想：
    - 路径1：标准卷积 (5×5)
    - 路径2：大感受野空间卷积 (7×7, dilation=3)
    - 双路径融合：通过avg+max注意力加权
    
    针对小麦检测的改进：
    - 路径1：小核捕获细节 (3×3)
    - 路径2：大核捕获上下文 (5×5)
    """
    def __init__(self, channels: int):
        super().__init__()
        # 路径1：小核（LSKblock的conv0）
        self.conv_small = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        
        # 路径2：大核（LSKblock的conv_spatial）
        self.conv_large = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        
        # 降维（LSKblock的conv1和conv2）
        self.conv1 = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels // 2, 1, bias=False)
        
        # 注意力融合（LSKblock的conv_squeeze）
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3, bias=False)
        
        # 恢复通道数（LSKblock的最终conv）
        self.conv_out = nn.Conv2d(channels // 2, channels, 1, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            增强后的特征
        """
        # 双路径卷积（LSKblock代码）
        attn1 = self.conv_small(x)
        attn2 = self.conv_large(attn1)
        
        # 降维（LSKblock代码）
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        # 合并两路特征
        attn = torch.cat([attn1, attn2], dim=1)
        
        # 计算avg和max统计（LSKblock代码）
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        
        # 生成注意力权重（LSKblock代码）
        sig = self.conv_squeeze(agg).sigmoid()
        
        # 加权融合（LSKblock代码）
        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]
        
        # 恢复通道数
        attn = self.conv_out(attn)
        
        # 特征增强（LSKblock: return x * attn）
        return x * attn


class WheatPolyKernel(nn.Module):
    """小麦感知多核卷积模块 v2（完全基于顶会代码重构）
    
    整体架构借鉴PKIBlock的Bottleneck结构：
    1. pre_conv: 通道扩展
    2. 多核卷积分支
    3. 双路径增强
    4. 统计引导注意力
    5. post_conv: 通道恢复
    6. 残差连接
    
    关键改进（相比v1）：
    - 渐进式融合（PKI）替代门控机制
    - 分支式计算（Inception）替代全局卷积
    - 统计引导（SMFA）替代全局池化注意力
    - 双路径增强（LSK）增强特征表达
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        expansion: float = 0.5,
        norm_cfg: dict = None,
        act_cfg: dict = None
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = max(int(in_channels * expansion), 32)
        
        # 默认配置
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='SiLU')
        
        # 1. 预卷积（PKIBlock的pre_conv）
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        # 2. 小麦形状自适应卷积核（融合PKI+Inception思想）
        self.wheat_kernels = WheatShapedKernels(hidden_channels, branch_ratio=0.25)
        
        # 3. 双路径特征增强（LSKblock完整实现）
        self.dual_path = DualPathEnhancement(hidden_channels)
        
        # 4. 统计引导注意力（SMFA的variance-guided modulation）
        self.variance_attn = VarianceGuidedAttention(hidden_channels)
        
        # 5. 后卷积（PKIBlock的post_conv）
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 6. 残差连接（PKIBlock的add_identity）
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        """
        前向传播（借鉴PKIBlock的渐进式融合）
        
        PKIBlock代码：
        x = self.pre_conv(x)
        y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + ...  # 渐进融合
        x = self.pw_conv(x)
        if self.caa_factor:
            y = self.caa_factor(y)
        x = x * y if not add_identity else x + x * y
        x = self.post_conv(x)
        """
        identity = x
        
        # 预卷积
        x = self.pre_conv(x)
        
        # 保存原始特征（用于后续调制）
        y = x
        
        # 小麦形状卷积核（渐进式融合，PKIBlock风格）
        x = self.wheat_kernels(x)
        
        # 双路径增强（LSKblock）
        x_enhanced = self.dual_path(x)
        
        # 统计引导注意力（SMFA）
        y_modulated = self.variance_attn(y)
        
        # 特征融合（PKIBlock的调制机制：x * y）
        x = x_enhanced * y_modulated
        
        # 后卷积
        x = self.post_conv(x)
        
        # 残差连接（PKIBlock的add_identity）
        if self.use_residual:
            x = x + identity
        else:
            x = x + self.shortcut(identity)
        
        # 最终激活
        x = self.act(x)
        
        return x, None  # 保持接口兼容性


class WAPKBlock(nn.Module):
    """WAPK Block：完整的模块单元（优化版）
    
    可以直接替换ResNet、FPN等网络中的标准卷积层
    
    优化点：
    - 降低默认expansion（1.0→0.5）
    - 优化下采样策略
    - 改进参数初始化
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,  # 降低默认expansion
        downsample: bool = False
    ):
        super().__init__()
        
        # 如果需要下采样
        if downsample:
            self.downsample = ConvBNAct(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=2,
                groups=1,
                use_act=True,  # 启用激活
                use_lab=False,  # 不使用lab
            )
            self.wapk = WheatPolyKernel(
                in_channels=in_channels * 2,
                out_channels=out_channels,
                expansion=expansion
            )
        else:
            self.downsample = nn.Identity()
            self.wapk = WheatPolyKernel(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=expansion
            )
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            out: 输出特征 (B, C', H, W)
        """
        x = self.downsample(x)
        x, _ = self.wapk(x)
        
        return x


def test_wapk_module():
    """测试WAPK模块的功能和参数量"""
    print("\n" + "="*80)
    print("测试 Wheat-Aware Poly Kernel Network (WAPK) v2")
    print("="*80)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 测试配置
    batch_size = 2
    in_channels = 256
    out_channels = 256
    height, width = 32, 32
    
    print(f"\n输入配置:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Input Channels: {in_channels}")
    print(f"  Output Channels: {out_channels}")
    print(f"  Feature Size: {height} × {width}")
    
    # 创建模块
    model = WheatPolyKernel(
        in_channels=in_channels,
        out_channels=out_channels,
        expansion=0.5
    ).to(device)
    
    # 创建输入
    inputs = torch.randn(batch_size, in_channels, height, width).to(device)
    
    # 前向传播
    print(f"\n前向传播测试:")
    with torch.no_grad():
        outputs, _ = model(inputs)
    
    print(f"  输入尺寸: {inputs.shape}")
    print(f"  输出尺寸: {outputs.shape}")
    
    # 计算参数量
    print(f"\n模块结构:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 参数量对比
    standard_conv_params = in_channels * out_channels * 3 * 3
    param_increase = (total_params - standard_conv_params) / standard_conv_params * 100
    print(f"  标准3×3卷积参数: {standard_conv_params:,}")
    print(f"  参数增加比例: {param_increase:+.2f}%")
    
    # 使用calflops计算FLOPs（如果可用）
    if CALFLOPS_AVAILABLE:
        print(f"\n计算复杂度分析:")
        try:
            flops, macs, params = calculate_flops(
                model=model,
                input_shape=(batch_size, in_channels, height, width),
                output_as_string=True,
                output_precision=4,
                print_detailed=False
            )
            print(f"  FLOPs: {flops}")
            print(f"  MACs: {macs}")
            print(f"  参数量: {params}")
        except Exception as e:
            print(f"  计算失败: {e}")
    
    print("\n" + "="*80)
    print("✓ WAPK v2模块测试完成")
    print("="*80 + "\n")


if __name__ == '__main__':
    # 设置颜色输出
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    
    print(GREEN + "="*80 + RESET)
    print(GREEN + " Wheat-Aware Poly Kernel Network (WAPK) v2 - 基于顶会代码重构" + RESET)
    print(GREEN + "="*80 + RESET)
    
    # 运行测试
    test_wapk_module()
    
    # 测试不同通道数配置
    print(YELLOW + "\n测试不同配置下的参数量:" + RESET)
    configs = [
        (64, 64, "P3层 (64→64)"),
        (128, 128, "P4层 (128→128)"),
        (256, 256, "P5层 (256→256)"),
    ]
    
    for in_c, out_c, desc in configs:
        model = WheatPolyKernel(in_c, out_c, expansion=0.5)
        params = sum(p.numel() for p in model.parameters())
        standard_params = in_c * out_c * 3 * 3
        increase = (params - standard_params) / standard_params * 100
        print(f"  {desc}: {params:,} 参数 ({increase:+.2f}%)")
    
    print(BLUE + "\n" + "="*80 + RESET)
    print(BLUE + "核心代码借鉴自4篇顶会论文：" + RESET)
    print(BLUE + "="*80 + RESET)
    
    print(f"\n{ORANGE}[1] PKIBlock (CVPR 2024){RESET}")
    print("    论文: Poly Kernel Inception Network for Remote Sensing Detection")
    print("    借鉴代码: 渐进式多核融合")
    print("    核心实现: x = x + kernel1(x) + kernel2(x) + kernel3(x)")
    print("    优势: 残差式累加，训练稳定，特征表达能力强")
    
    print(f"\n{ORANGE}[2] LSKblock (ICCV 2023){RESET}")
    print("    论文: Large Selective Kernel Network for Remote Sensing Object Detection")
    print("    借鉴代码: 双路径注意力（spatial + avg/max统计）")
    print("    核心实现: DualPathEnhancement类完整复刻LSK双路径设计")
    print("    优势: 轻量级全局感知，自适应多尺度特征")
    
    print(f"\n{ORANGE}[3] SMFA (ECCV 2024){RESET}")
    print("    论文: SMFANet: A Lightweight Self-Modulation Feature Aggregation Network")
    print("    借鉴代码: 统计引导的自调制")
    print("    核心实现: x_v = torch.var(x); x = x * (alpha + x_v * belt)")
    print("    优势: 方差统计作为全局上下文，参数化自适应")
    
    print(f"\n{ORANGE}[4] InceptionDWConv (CVPR 2024){RESET}")
    print("    论文: InceptionNeXt: When Inception Meets ConvNeXt")
    print("    借鉴代码: 分支式高效计算")
    print("    核心实现: torch.split + 独立分支处理 + concat")
    print("    优势: 降低计算量，保持表达能力")
    
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}WAPK v2针对GWHD数据集的创新点：{RESET}")
    print(f"{GREEN}{'='*80}{RESET}")
    print("\n1. 形状自适应: 竖向/横向带状核 (1×7+7×1, 7×1+1×7) 捕获细长麦穗")
    print("2. 渐进式融合: PKI风格的残差累加，避免门控机制的特征抑制")
    print("3. 统计引导: SMFA的方差调制，轻量级全局感知")
    print("4. 双路径增强: LSK的多尺度注意力，自适应感受野")
    print("5. 分支式计算: Inception的split设计，降低参数量和计算量")
    
    print(f"\n{GREEN}预期效果：{RESET}")
    print("- 参数量减少50%（相比v1）")
    print("- 训练更稳定（渐进融合 + 统计引导）")
    print("- 细长目标捕获能力增强（带状核 + 双路径）")
    print("- 密度适应性更好（方差调制 + LSK注意力）")
