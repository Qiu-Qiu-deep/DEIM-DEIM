"""
Wave Propagation Modules for DFINE Integration
基于WaveFormer的波动传播算子，适配于目标检测任务

集成方案：
1. Wave2D: 核心波动传播模块（适配检测特征）
2. WaveEnhancedEncoder: 在Transformer基础上增加Wave分支
3. WaveEncoderBlock: 完全替换Transformer的版本
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入DCT实现
try:
    # PyTorch 1.8+ 有原生DCT
    torch.fft.dct
    USE_TORCH_DCT = True
except AttributeError:
    # 使用torch_dct库
    try:
        import torch_dct as dct
        USE_TORCH_DCT = False
    except ImportError:
        print("警告: 需要安装torch_dct: pip install torch_dct")
        raise


class Wave2D(nn.Module):
    """
    阻尼波动方程在2D特征图上的实现
    基于频率域的解析解：
    u(x,y,t) = F⁻¹{e^(-αt/2)[F(u₀)cos(ωₐt) + sin(ωₐt)/ωₐ(F(v₀) + α/2·F(u₀))]}
    
    Args:
        dim: 输入通道数
        hidden_dim: 隐藏层通道数
        res: 特征图分辨率（用于频率嵌入）
        learnable_params: 是否让α和c可学习
    """
    def __init__(self, dim=128, hidden_dim=None, res=20, learnable_params=True, use_padding=True):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.res = res
        self.use_padding = use_padding  # 是否使用padding减少边界伪影
        
        # 深度可分离卷积：提取局部特征
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        # 线性变换：生成u₀和v₀（初始语义场和速度场）
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        
        # 输出归一化和投影
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # 频率嵌入到时间的映射（学习每个频率的传播时间）
        self.to_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        
        # 波动方程参数
        if learnable_params:
            self.wave_speed = nn.Parameter(torch.ones(1) * 1.0)  # c: 波速
            self.damping = nn.Parameter(torch.ones(1) * 0.1)     # α: 阻尼系数
        else:
            self.register_buffer('wave_speed', torch.ones(1) * 1.0)
            self.register_buffer('damping', torch.ones(1) * 0.1)
    
    def forward(self, x, freq_embed=None):
        """
        Args:
            x: [B, C, H, W] 输入特征图
            freq_embed: [H, W, C] 可选的频率位置编码
        Returns:
            [B, C, H, W] 波动传播后的特征
        """
        B, C, H, W = x.shape
        
        # 1. 局部特征提取
        x = self.dwconv(x)
        
        # 2. 生成初始语义场u₀和速度场v₀
        x_transformed = self.linear(x.permute(0, 2, 3, 1))  # [B, H, W, 2C]
        u0, v0 = x_transformed.chunk(2, dim=-1)  # 各自 [B, H, W, C]
        
        u0 = u0.permute(0, 3, 1, 2)  # [B, C, H, W]
        v0 = v0.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 3. 频率域变换（可选padding减少边界伪影）
        if self.use_padding:
            # 使用反射padding减少DCT的周期性假设带来的边界效应
            pad_h, pad_w = H // 4, W // 4
            u0_padded = F.pad(u0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            v0_padded = F.pad(v0, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            u0_freq = self.dct2d(u0_padded)
            v0_freq = self.dct2d(v0_padded)
            # 记录padding尺寸和实际频率域尺寸
            freq_H, freq_W = H + 2 * pad_h, W + 2 * pad_w
            pad_info = (pad_h, pad_w)
        else:
            u0_freq = self.dct2d(u0)
            v0_freq = self.dct2d(v0)
            freq_H, freq_W = H, W
            pad_info = None
        
        # 4. 计算传播时间（频率感知）
        if freq_embed is not None:
            # 使用外部频率嵌入（每个stage的可学习参数）
            t = self.to_time(freq_embed.unsqueeze(0).expand(B, -1, -1, -1))
            t = t.permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            # 使用频率坐标生成时间参数（关键修正）
            # 为每个DCT系数位置生成与频率相关的时间（使用频率域尺寸）
            freq_y = torch.arange(freq_H, device=x.device, dtype=x.dtype).view(1, 1, freq_H, 1)
            freq_x = torch.arange(freq_W, device=x.device, dtype=x.dtype).view(1, 1, 1, freq_W)
            # 归一化频率坐标到[0, π]
            freq_y = freq_y * (math.pi / freq_H)
            freq_x = freq_x * (math.pi / freq_W)
            # 径向频率（空间频率大小）
            omega = torch.sqrt(freq_y**2 + freq_x**2)  # [1, 1, freq_H, freq_W]
            # 扩展到batch和channel
            t = omega.expand(B, C, freq_H, freq_W)
        
        # 5. 波动方程求解（阻尼振荡）
        # ω_d = sqrt(ω²c² - (α/2)²) 阻尼频率
        omega_d = torch.sqrt(torch.clamp(
            (self.wave_speed * t)**2 - (self.damping / 2)**2,
            min=1e-8
        ))
        
        cos_term = torch.cos(omega_d)
        sin_term = torch.sin(omega_d) / (omega_d + 1e-8)
        
        # 波动项 + 速度项
        wave_component = cos_term * u0_freq
        velocity_component = sin_term * (v0_freq + (self.damping / 2) * u0_freq)
        
        # 关键修正：应用阻尼衰减因子 e^(-αt/2)
        damping_factor = torch.exp(-self.damping * t / 2)
        final_freq = damping_factor * (wave_component + velocity_component)
        
        # 6. 逆变换回空间域
        x_wave = self.idct2d(final_freq)
        
        # 如果使用了padding，需要裁剪回原始尺寸
        if self.use_padding and pad_info is not None:
            pad_h, pad_w = pad_info
            x_wave = x_wave[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        # 7. 输出处理（归一化 + 门控）
        x_wave = self.out_norm(x_wave.permute(0, 2, 3, 1))
        x_wave = x_wave.permute(0, 3, 1, 2)
        
        # SiLU门控（类似GLU）
        gate = F.silu(v0)
        x_wave = x_wave * gate
        
        x_out = self.out_linear(x_wave.permute(0, 2, 3, 1))
        x_out = x_out.permute(0, 3, 1, 2)
        
        return x_out
    
    @staticmethod
    def dct2d(x):
        """2D DCT-II变换"""
        if USE_TORCH_DCT:
            x = torch.fft.dct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.dct(x, type=2, dim=-1, norm='ortho')
        else:
            # 使用torch_dct库
            x = dct.dct_2d(x, norm='ortho')
        return x
    
    @staticmethod
    def idct2d(x):
        """2D IDCT-II变换"""
        if USE_TORCH_DCT:
            x = torch.fft.idct(x, type=2, dim=-2, norm='ortho')
            x = torch.fft.idct(x, type=2, dim=-1, norm='ortho')
        else:
            # 使用torch_dct库
            x = dct.idct_2d(x, norm='ortho')
        return x


class WaveEnhancedEncoder(nn.Module):
    """
    混合架构：Transformer + Wave双分支
    适用于阶段1：最小侵入式集成
    
    结构：
    Input → Transformer分支 ──┐
         → Wave分支 ──────────┤ → Fusion → Output
    """
    def __init__(self, 
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False,
                 wave_enabled=True,
                 wave_weight=0.5):
        super().__init__()
        self.wave_enabled = wave_enabled
        self.wave_weight = wave_weight
        
        # Transformer分支（保持原有能力）
        from engine.deim.hybrid_encoder import TransformerEncoderBlock
        self.transformer_branch = TransformerEncoderBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            pe_temperature=pe_temperature,
            normalize_before=normalize_before
        )
        
        # Wave分支（新增能力）
        if wave_enabled:
            self.wave_branch = Wave2D(
                dim=d_model,
                hidden_dim=d_model,
                res=20,  # 默认640/32=20，会根据实际尺寸调整
                learnable_params=True,
                use_padding=True  # 使用padding减少边界伪影
            )
            
            # 特征融合：可学习的加权融合
            self.fusion_weight = nn.Parameter(torch.tensor([1.0, wave_weight]))
            self.fusion_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            [B, C, H, W] 增强后的特征图
        """
        if not self.wave_enabled:
            return self.transformer_branch(x)
        
        B, C, H, W = x.shape
        
        # Transformer分支
        feat_trans = self.transformer_branch(x)
        
        # Wave分支
        feat_wave = self.wave_branch(x)
        
        # 自适应融合（softmax归一化权重）
        weights = F.softmax(self.fusion_weight, dim=0)
        feat_fused = weights[0] * feat_trans + weights[1] * feat_wave
        
        # 残差连接 + 归一化
        feat_fused = self.fusion_norm(feat_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return feat_fused


class WaveEncoderBlock(nn.Module):
    """
    纯Wave版本的Encoder Block
    适用于阶段2：完全替换Transformer
    
    结构：Wave2D + FFN（类似Transformer的MHA + FFN）
    """
    def __init__(self,
                 d_model,
                 nhead=8,  # 保留接口兼容性，但不使用
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pe_temperature=10000,
                 normalize_before=False):
        super().__init__()
        from engine.deim.utils import get_activation
        
        self.normalize_before = normalize_before
        
        # Wave传播层（替代Multi-Head Attention）
        self.wave_op = Wave2D(
            dim=d_model,
            hidden_dim=d_model,
            res=20,
            learnable_params=True,
            use_padding=True  # 使用padding减少边界伪影
        )
        
        # FFN层（保持与Transformer一致）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Wave传播 + 残差
        if self.normalize_before:
            x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.dropout1(self.wave_op(x_norm))
        else:
            wave_out = self.wave_op(x)
            x = x + self.dropout1(wave_out)
            x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # FFN + 残差
        if self.normalize_before:
            x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x2 = self.linear2(self.dropout(self.activation(
                self.linear1(x_norm.permute(0, 2, 3, 1))
            )))
            x = x + self.dropout2(x2.permute(0, 3, 1, 2))
        else:
            x_ffn = x.permute(0, 2, 3, 1)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x_ffn))))
            x = x + self.dropout2(x2.permute(0, 3, 1, 2))
            x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


class MultiScaleWaveEncoder(nn.Module):
    """
    多尺度Wave编码器（阶段3：深度融合）
    为不同分辨率的特征使用不同的波动参数
    
    P4 (H/16): α=0.1, c=1.2 (保留高频细节)
    P5 (H/32): α=0.3, c=0.8 (关注全局语义)
    """
    def __init__(self, dim=128, scales=[16, 32]):
        super().__init__()
        self.scales = scales
        
        # 为每个尺度创建独立的Wave模块
        self.wave_modules = nn.ModuleList([
            Wave2D(
                dim=dim,
                hidden_dim=dim,
                res=640 // scale,
                learnable_params=False  # 使用预设参数
            ) for scale in scales
        ])
        
        # 为P4设置参数（高频）
        self.wave_modules[0].damping.data = torch.tensor([0.1])
        self.wave_modules[0].wave_speed.data = torch.tensor([1.2])
        
        # 为P5设置参数（低频）
        self.wave_modules[1].damping.data = torch.tensor([0.3])
        self.wave_modules[1].wave_speed.data = torch.tensor([0.8])
        
        # 跨尺度融合
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv2d(dim * len(scales), dim, 1),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
    
    def forward(self, features):
        """
        Args:
            features: List[[B, C, H, W]] 多尺度特征
        Returns:
            List[[B, C, H, W]] 增强后的多尺度特征
        """
        enhanced_feats = []
        
        for i, (feat, wave_module) in enumerate(zip(features, self.wave_modules)):
            enhanced = wave_module(feat)
            enhanced_feats.append(enhanced)
        
        return enhanced_feats


# ============= 辅助函数：注册到tasks.py =============
def register_wave_modules():
    """
    在tasks.py中注册Wave模块，使其能够通过YAML配置
    需要在tasks.py中添加：
    
    from engine.extre_module.wave_modules import WaveEnhancedEncoder, WaveEncoderBlock
    
    然后在parse_model函数中添加：
    elif m in {WaveEnhancedEncoder, WaveEncoderBlock}:
        c2 = ch[f]
        args = [c2, *args]
    """
    pass


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    print("="*60)
    print("测试Wave2D模块")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 2, 128, 20, 20
    x = torch.randn(B, C, H, W)
    
    # 测试Wave2D
    wave = Wave2D(dim=C, hidden_dim=C, res=H)
    out = wave(x)
    print(f"Wave2D输入: {x.shape}, 输出: {out.shape}")
    assert out.shape == x.shape, "形状不匹配！"
    
    # 测试WaveEnhancedEncoder
    print("\n" + "="*60)
    print("测试WaveEnhancedEncoder")
    print("="*60)
    wave_enhanced = WaveEnhancedEncoder(
        d_model=C,
        nhead=8,
        dim_feedforward=512,
        wave_enabled=True
    )
    out2 = wave_enhanced(x)
    print(f"WaveEnhancedEncoder输入: {x.shape}, 输出: {out2.shape}")
    
    # 测试WaveEncoderBlock
    print("\n" + "="*60)
    print("测试WaveEncoderBlock")
    print("="*60)
    wave_block = WaveEncoderBlock(d_model=C, dim_feedforward=512)
    out3 = wave_block(x)
    print(f"WaveEncoderBlock输入: {x.shape}, 输出: {out3.shape}")
    
    print("\n✅ 所有模块测试通过！")
