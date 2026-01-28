"""
D-FINE Decoder with DAQS (Density-Adaptive Query Sampling) Integration
核心改进：让DAQS真正参与检测流程

设计思路（V2 - 修复版）：
1. 继承DFINETransformer，重写_get_decoder_input方法
2. enable_daqs=False: 调用父类方法（完全一致行为）
3. enable_daqs=True: 用DAQS生成的queries替代top-k选择

关键修改点：
- _get_decoder_input: DAQS生成queries，替代原始的encoder top-k
- forward: 保持标准流程，DAQS自然融入
- 训练和推理使用相同逻辑（避免train/test不一致）

V2改进：
- 修复：DAQS生成的queries真正用于decoder训练
- 修复：训练时DAQS接收检测任务的梯度反馈
- 优化：简化forward逻辑，去除train/test分支
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
    
    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):
        """重写decoder输入生成（DAQS核心集成点）
        
        原始流程：encoder输出 → top-k选择 → decoder输入
        DAQS流程：encoder输出 + 投影特征 → DAQS动态生成 → decoder输入
        
        关键改进：
        1. DAQS生成的queries直接用于decoder（不再是旁观者）
        2. 保留encoder的bbox预测用于辅助损失
        3. 训练和推理使用相同逻辑
        """
        if not self.enable_daqs:
            # 禁用DAQS时，调用父类方法
            return super()._get_decoder_input(
                memory, spatial_shapes, denoising_logits, denoising_bbox_unact
            )
        
        # ========== DAQS启用流程 ==========
        bs = memory.shape[0]
        device = memory.device
        
        # 1. 生成anchors（父类逻辑，用于encoder的bbox预测）
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        
        if bs > 1:
            anchors = anchors.repeat(bs, 1, 1)
        
        memory = valid_mask.to(memory.dtype) * memory
        
        # 2. Encoder输出（用于辅助损失，保持与原始DFINE一致）
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)
        
        # 3. **关键：DAQS生成动态queries**
        # 获取投影后的特征（与DAQS的输入一致）
        # 注意：这里需要从原始feats获取，但_get_decoder_input没有feats参数
        # 解决方案：在forward中缓存proj_feats
        if not hasattr(self, '_cached_proj_feats'):
            raise RuntimeError("DAQS需要投影特征，请确保forward中正确缓存")
        
        proj_feats = self._cached_proj_feats
        daqs_queries, daqs_query_pos, density_map, predicted_count, num_queries = self.daqs(proj_feats)
        
        # 缓存DAQS输出供forward使用
        self._cached_density_map = density_map
        self._cached_predicted_count = predicted_count
        self._cached_num_queries = num_queries
        
        # 4. 生成query的初始bbox（使用DAQS的位置）
        # daqs_query_pos包含位置信息，转换为bbox格式
        # 这里简化处理：从query_pos提取位置，生成小的初始bbox
        # query_pos是(B, Q, embed_dim)，我们需要(B, Q, 4)的bbox
        
        # 使用encoder的bbox head预测初始bbox（基于DAQS queries）
        enc_topk_bbox_unact = self.enc_bbox_head(daqs_queries)  # (B, Q, 4)
        
        # 5. 训练时需要encoder的top-k用于辅助损失
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            # 从encoder输出中选择top-k（用于辅助损失）
            # 使用原始的num_queries（配置中的值）
            _, topk_ind = torch.topk(
                enc_outputs_logits.max(-1).values, 
                self.num_queries,  # 使用配置的固定值
                dim=-1
            )
            
            enc_topk_anchors = anchors.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, 4)
            )
            enc_topk_logits = enc_outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_logits.shape[-1])
            )
            enc_topk_bbox_from_encoder = self.enc_bbox_head(
                output_memory.gather(
                    dim=1,
                    index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
                )
            ) + enc_topk_anchors
            
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_from_encoder)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)
        
        # 6. 获取query内容（基于DAQS）
        if self.learn_query_content:
            # 如果配置为学习query内容，使用DAQS生成的queries
            content = daqs_queries
        else:
            # 否则使用DAQS queries（已经包含内容）
            content = daqs_queries.detach()
        
        # bbox不detach，需要梯度
        # enc_topk_bbox_unact不detach（DAQS生成的）
        
        # 7. 处理denoising（去噪训练）
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits
    
    def forward(self, feats: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """
        V2改进：统一训练和推理流程
        
        核心改变：
        1. DAQS生成的queries直接送入decoder
        2. 训练时DAQS接收检测loss的梯度
        3. 去除train/test分支，逻辑更清晰
        
        V2.1修复：
        - 修复denoising的attn_mask维度问题
        - 先运行DAQS获取动态query数量，再调用父类forward
        """
        if not self.enable_daqs:
            return super().forward(feats, targets)
        
        # ========== 预处理：缓存投影特征 ==========
        # DAQS需要投影后的特征，在这里缓存供_get_decoder_input使用
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats[:self.num_levels])]
        
        # 处理额外的levels
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))
        
        self._cached_proj_feats = proj_feats
        
        # ========== V2.1关键修复：提前运行DAQS获取动态query数量 ==========
        # 这样父类的denoising生成才能使用正确的query数量
        _, _, density_map, predicted_count, dynamic_num_queries = self.daqs(proj_feats)
        
        # 临时修改num_queries（让denoising使用正确的数量）
        original_num_queries = self.num_queries
        self.num_queries = dynamic_num_queries
        
        # ========== 调用父类forward ==========
        # _get_decoder_input会被调用，使用我们重写的版本
        try:
            output = super().forward(feats, targets)
        finally:
            # 确保恢复原始值
            self.num_queries = original_num_queries
        
        # ========== 添加DAQS输出到结果 ==========
        if hasattr(self, '_cached_density_map'):
            output['daqs_density_map'] = self._cached_density_map
            output['daqs_predicted_count'] = self._cached_predicted_count
            if not self.training:
                # 推理时记录实际使用的query数量
                output['daqs_num_queries'] = torch.tensor(
                    [self._cached_num_queries] * self._cached_density_map.shape[0],
                    device=self._cached_density_map.device
                )
        
        # 清理缓存
        if hasattr(self, '_cached_proj_feats'):
            delattr(self, '_cached_proj_feats')
        if hasattr(self, '_cached_density_map'):
            delattr(self, '_cached_density_map')
            delattr(self, '_cached_predicted_count')
            delattr(self, '_cached_num_queries')
        
        return output
