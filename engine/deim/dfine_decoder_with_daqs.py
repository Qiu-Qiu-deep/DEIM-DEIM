"""
D-FINE Decoder with DAQS (Density-Adaptive Query Sampling) Integration
向后兼容的继承实现，enable_daqs=False时行为与原始DFINETransformer完全一致

设计思路：
1. 继承DFINETransformer，重写_get_decoder_input方法
2. enable_daqs=False: 调用父类方法（完全一致行为）
3. enable_daqs=True: 使用DAQS动态生成queries

关键修改点：
- _get_decoder_input: 原本通过top-k选择queries，改为DAQS动态生成
- forward: 在输出中添加DAQS的density_map等信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from .dfine_decoder import DFINETransformer
from ..extre_module.paper_first.daqs import DAQS
from ..core import register


@register()
class DFINETransformerWithDAQS(DFINETransformer):
    """
    D-FINE Transformer Decoder with optional DAQS support
    
    核心设计原则：
    1. 继承DFINETransformer：复用所有基础功能
    2. 向后兼容：enable_daqs=False时，完全使用父类方法
    3. 可选DAQS：enable_daqs=True时，用DAQS替代top-k query selection
    
    Args:
        enable_daqs: 是否启用DAQS（默认False保持向后兼容）
        daqs_hidden_dim: DAQS的隐藏层维度
        daqs_min_queries: DAQS最小query数量
        daqs_max_queries: DAQS最大query数量
        daqs_alpha: DAQS的alpha参数（query数量 = alpha * 预测数量）
        其他参数: 与DFINETransformer相同
    """
    
    def __init__(
        self,
        # DFINETransformer的所有参数
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='default',
        reg_max=32,
        reg_scale=4.,
        layer_scale=1,
        mlp_act='relu',
        # DAQS特有参数
        enable_daqs: bool = False,
        daqs_hidden_dim: int = 64,
        daqs_min_queries: int = 100,
        daqs_max_queries: int = 800,
        daqs_alpha: float = 2.0,
    ):
        # 调用父类初始化
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
        )
        
        self.enable_daqs = enable_daqs
        
        # 如果启用DAQS，创建DAQS模块
        if enable_daqs:
            # DAQS输入是投影后的特征，通道数都是hidden_dim
            in_channels_list = [hidden_dim] * min(len(feat_channels), num_levels)
            
            self.daqs = DAQS(
                in_channels_list=in_channels_list,
                embed_dim=hidden_dim,
                hidden_dim=daqs_hidden_dim,
                min_queries=daqs_min_queries,
                max_queries=daqs_max_queries,
                alpha=daqs_alpha
            )
        else:
            self.daqs = None
    
    def forward(self, feats: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """
        简化方案A：DAQS最小化集成
        
        训练阶段：
          - 主检测器：固定300 queries（标准流程）
          - DAQS：并行训练（有梯度），输出density_map供criterion计算损失
          - 关键：DAQS不改变decoder的query生成逻辑
        
        推理阶段：
          - DAQS预测密度 → 计算动态query数量
          - 临时修改num_queries → 调用父类forward
          - 父类的_get_decoder_input会用新的num_queries做top-k
        
        这样DAQS通过密度损失学会预测，但不干扰训练稳定性
        """
        if not self.enable_daqs:
            return super().forward(feats, targets)
        
        # ========== 训练阶段 ==========
        if self.training:
            # 1. 调用父类标准forward（固定queries）
            output = super().forward(feats, targets)
            
            # 2. DAQS并行前向传播（带梯度，用于密度损失）
            proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats[:self.num_levels])]
            _, _, density_map, predicted_count, _ = self.daqs(proj_feats)
            
            # 3. 添加到输出，供criterion计算密度损失
            output['daqs_density_map'] = density_map
            output['daqs_predicted_count'] = predicted_count
            
            return output
        
        # ========== 推理阶段 ==========
        else:
            # 1. DAQS预测密度
            proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats[:self.num_levels])]
            _, _, density_map, predicted_count, dynamic_num_queries = self.daqs(proj_feats)
            
            # 2. 临时修改num_queries（dynamic_num_queries已经是int）
            original_num_queries = self.num_queries
            self.num_queries = dynamic_num_queries
            
            # 3. 调用父类forward（会自动用新的num_queries）
            try:
                output = super().forward(feats, targets)
            finally:
                self.num_queries = original_num_queries
            
            # 4. 添加DAQS信息（便于调试和监控）
            output['daqs_density_map'] = density_map
            output['daqs_predicted_count'] = predicted_count
            output['daqs_num_queries'] = torch.tensor([dynamic_num_queries] * density_map.shape[0])
            # 确保实际query数量与输出一致
            assert output['pred_logits'].shape[1] == dynamic_num_queries, \
                f"Query数量不一致: pred_logits={output['pred_logits'].shape[1]}, dynamic_num_queries={dynamic_num_queries}"
            
            return output
