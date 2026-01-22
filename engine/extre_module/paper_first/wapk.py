'''
Wheat-Aware Poly Kernel Network (WAPK)
小麦感知的多核卷积网络

引入动机：
针对GWHD数据集的小麦穗形状特异性问题（细长椭圆形，长宽比1:2-1:3），
标准方形卷积核（3×3, 5×5）对细长目标捕获不足。本模块设计椭圆形卷积核，
动态选择最适合当前场景的核，增强小目标特征提取能力。

参考论文：
1. PKIBlock (CVPR 2024): "Poly Kernel Inception Network for Remote Sensing Detection"
   - 借鉴多核卷积的思想，但针对小麦形状设计椭圆核
   - 论文链接：https://arxiv.org/pdf/2403.06258
2. LSKblock (ICCV 2023): "Large Selective Kernel Network for Remote Sensing Object Detection"  
   - 借鉴自适应核选择的注意力机制
   - Github：https://github.com/zcablii/Large-Selective-Kernel-Network
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


class EllipticalDepthwiseConv(nn.Module):
    """椭圆形深度可分离卷积
    
    针对小麦穗的细长椭圆形状设计，使用非对称卷积核捕获椭圆形特征
    """
    def __init__(self, channels: int, kernel_size: tuple, dilation: int = 1):
        """
        Args:
            channels: 输入输出通道数
            kernel_size: 卷积核大小 (h, w)，如 (3, 5) 表示竖向椭圆
            dilation: 膨胀率，用于增大感受野
        """
        super().__init__()
        self.kernel_size = kernel_size
        padding = autopad(kernel_size, dilation)
        
        # 深度可分离卷积：每个通道独立卷积
        self.dwconv = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=channels,  # 深度可分离
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return self.bn(self.dwconv(x))


class WheatPolyKernel(nn.Module):
    """小麦感知的多核卷积模块
    
    设计4个不同形状的卷积核，通过注意力机制自适应选择：
    1. 竖向椭圆核 (3×5)：捕获竖向排列的麦穗
    2. 横向椭圆核 (5×3)：捕获横向排列的麦穗  
    3. 细长椭圆核 (3×7)：捕获更细长的麦穗
    4. 标准方形核 (3×3)：保留标准特征提取能力
    
    参考PKIBlock的多核Inception结构，但针对小麦形状优化核设计
    """
    def __init__(
        self, 
        in_channels: int,
        out_channels: Optional[int] = None,
        expansion: float = 1.0,
        norm_cfg: dict = None,
        act_cfg: dict = None
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数，默认与输入相同
            expansion: 隐藏层通道扩展系数
            norm_cfg: 归一化配置
            act_cfg: 激活函数配置
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * expansion)
        
        # 默认配置
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='SiLU')
        
        # 1. 预卷积：通道调整
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        # 2. 四个椭圆形/标准卷积核（深度可分离）
        # 参考PKIBlock的多核结构，但设计针对小麦的椭圆核
        self.kernel_vertical = EllipticalDepthwiseConv(hidden_channels, (3, 5))    # 竖向椭圆
        self.kernel_horizontal = EllipticalDepthwiseConv(hidden_channels, (5, 3))  # 横向椭圆
        self.kernel_elongated = EllipticalDepthwiseConv(hidden_channels, (3, 7))   # 细长椭圆
        self.kernel_square = EllipticalDepthwiseConv(hidden_channels, (3, 3))      # 标准方形
        
        # 3. 逐点卷积：融合多核特征
        self.pw_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        # 4. 核选择注意力（参考LSKblock的注意力机制）
        # 使用全局平均池化和1x1卷积生成4个核的权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 -> (B, C, 1, 1)
            nn.Conv2d(hidden_channels, 4, 1),  # 生成4个权重
            nn.Softmax(dim=1)  # 归一化权重
        )
        
        # 5. 后卷积：通道恢复
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 残差连接
        self.use_residual = (in_channels == out_channels)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            out: 输出特征 (B, C', H, W)
            weights: 四个核的选择权重 (B, 4, 1, 1)，用于可视化
        """
        identity = x
        
        # 预卷积
        x = self.pre_conv(x)  # (B, C_hidden, H, W)
        
        # 四个核的卷积输出
        feat_vertical = self.kernel_vertical(x)      # 竖向椭圆
        feat_horizontal = self.kernel_horizontal(x)  # 横向椭圆
        feat_elongated = self.kernel_elongated(x)    # 细长椭圆
        feat_square = self.kernel_square(x)          # 标准方形
        
        # 逐点卷积
        x = self.pw_conv(x)
        
        # 核选择注意力
        weights = self.attention(x)  # (B, 4, 1, 1)
        
        # 加权融合四个核的输出（参考PKIBlock的相加融合）
        x_fused = (
            feat_vertical * weights[:, 0:1, :, :] +
            feat_horizontal * weights[:, 1:2, :, :] +
            feat_elongated * weights[:, 2:3, :, :] +
            feat_square * weights[:, 3:4, :, :]
        )
        
        # 特征调制：x * x_fused（参考PKIBlock的门控机制）
        x = x * x_fused
        
        # 后卷积
        x = self.post_conv(x)
        
        # 残差连接
        if self.use_residual:
            x = x + identity
            
        return x, weights


class WAPKBlock(nn.Module):
    """WAPK Block：完整的模块单元
    
    可以直接替换ResNet、FPN等网络中的标准卷积层
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 1.0,
        downsample: bool = False
    ):
        super().__init__()
        
        
        # 如果需要下采样
        if downsample:
            self.downsample = ConvBNAct(
                in_channels,   # 输入通道数  
                in_channels*2,   # 维持通道数不变    
                kernel_size=3,  
                stride=2,  # 采用 stride=2 进行下采样   
                groups=1,  # 分组卷积（深度卷积） 
                use_act=False,  # 关闭激活函数   
                use_lab=True, # 是否使用可学习的仿射变换模块  
            )
            self.wapk = WheatPolyKernel(
            in_channels=in_channels*2,
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
    print("测试 Wheat-Aware Poly Kernel Network (WAPK)")
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
        expansion=1.0
    ).to(device)
    
    # 创建输入
    inputs = torch.randn(batch_size, in_channels, height, width).to(device)
    
    # 前向传播
    print(f"\n前向传播测试:")
    with torch.no_grad():
        outputs, weights = model(inputs)
    
    print(f"  输入尺寸: {inputs.shape}")
    print(f"  输出尺寸: {outputs.shape}")
    print(f"  核权重尺寸: {weights.shape}")
    
    # 打印核选择权重
    print(f"\n核选择权重（第一个样本）:")
    w = weights[0].squeeze().cpu().numpy()
    kernel_names = ['竖向椭圆(3×5)', '横向椭圆(5×3)', '细长椭圆(3×7)', '标准方形(3×3)']
    for i, (name, weight) in enumerate(zip(kernel_names, w)):
        print(f"  {name}: {weight:.4f}")
    
    # 计算参数量
    print(f"\n模块结构:")
    print(model)
    
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
    print("✓ WAPK模块测试完成")
    print("="*80 + "\n")


if __name__ == '__main__':
    # 设置颜色输出
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    
    print(GREEN + "="*80 + RESET)
    print(GREEN + " Wheat-Aware Poly Kernel Network (WAPK) - 独立测试" + RESET)
    print(GREEN + "="*80 + RESET)
    
    # 运行测试
    test_wapk_module()
    
    # 测试不同通道数配置
    print(YELLOW + "\n测试不同配置下的参数量增加:" + RESET)
    configs = [
        (64, 64, "P3层 (64→64)"),
        (128, 128, "P4层 (128→128)"),
        (256, 256, "P5层 (256→256)"),
    ]
    
    for in_c, out_c, desc in configs:
        model = WheatPolyKernel(in_c, out_c)
        params = sum(p.numel() for p in model.parameters())
        standard_params = in_c * out_c * 3 * 3
        increase = (params - standard_params) / standard_params * 100
        print(f"  {desc}: {params:,} 参数 ({increase:+.2f}%)")
    
    print(BLUE + "\n论文引用说明:" + RESET)
    print("  [1] PKIBlock (CVPR 2024) - 多核卷积结构设计")
    print("  [2] LSKblock (ICCV 2023) - 自适应核选择注意力")
    print("  本模块针对小麦形状特异性优化，设计椭圆形卷积核")
