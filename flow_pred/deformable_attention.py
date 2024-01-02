import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from opencood.utils.mmcv_utils import constant_init, xavier_init
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
# from multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32 as MultiScaleDeformableAttnFunction
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch # 使用pytorch版本， cuda算子在部分机器上编译出问题
from mmdet.models.utils import LearnedPositionalEncoding
from opencood.models.sub_modules.flownet import FlowPred

# SelfAttn 和 CrossAttn 可以通用
class DeforAttn(nn.Module):
    def __init__(self, embed_dims, num_heads=1, num_points=4, dropout=0.1, max_num_levels=2):
        super().__init__()

        self.max_num_levels = max_num_levels  # number_level, 
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, max_num_levels*num_heads*num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, max_num_levels*num_heads * num_points)

        self.dropout = nn.Dropout(dropout)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
    
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.max_num_levels, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self, query, query_pos, value, reference_points, spatial_shapes, *args):
        """
        Args:
            query: [1, H*W, C]
            query_pos: [1, H*W, C]
            value: [N, H*W, C]
            reference_points: [1, H*W, N, 2]
            spatial_shapes: [N, 2]
            # todo: 这里对于多个Bs没有很好兼容,需要改一下
        Returns:
            _type_: _description_
        """
        identity = query

        query = query + query_pos  # query_pos是每次都加吗,是的

        N, num_value, C = value.shape
        
        value = self.value_proj(value)
        # 跟之前单batch不能通用
        value = value.unsqueeze(0).contiguous().view(N, num_value, self.num_heads,-1) # [N, H*W, n_head, C//n_head]

        bs, num_query, C = query.shape
        
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.max_num_levels, self.num_points, 2)  # 
        # sampling_offsets = sampling_offsets[:, :, :, :N, :, :].contiguous() # [1, H*W, n_head, 2, n_point, 2]
        
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(bs, num_query,
                                                self.num_heads,
                                                self.max_num_levels,
                                                self.num_points)
        # attention_weights = attention_weights[:, :, :, :N, :]
         
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                self.num_heads,
                                                1,
                                                self.num_points).contiguous() # [B, H*W, n_head, 1, n_point]

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # [N, 2] 换了hw一下方位?
        # [1, H*W, N, 2]-> [1, H*W, 1, N, 1, 2] + [1, H*W, n_head, N, n_point, 2]
        sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        
        # output = MultiScaleDeformableAttnFunction.apply(
        #         value, spatial_shapes, torch.tensor([0], device=query.device), sampling_locations, attention_weights)
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights) # [1, h*w, c]
        
        output = self.output_proj(output)

        return self.dropout(output) + identity

class FFN(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 ffn_drop=0.1,
                 add_identity=True,
                 dropout_layer=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = dropout_layer if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity = None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class Block(nn.Module):
    def __init__(self, embed_dims, num_heads=1, num_points=4, dropout=0.1, max_num_levels=2, cfgs = ["self_attn", "norm", "cross_attn", "norm", "ffn", "norm"]) -> None:
        super().__init__()
        # 没有历史信息，第一层或许可以不要self_attn; 问题不大
        # self_attn,norm,cross_attn, norm, ffn, norm
        # 1) 后来的一种主流做法是，先只有cross_attn, 在self_attn； 问题不大，两个小的设计点都可以尝试, 配置生成
        # 2) surround_occ 是只有一层，后面都是卷积

        block_layers = nn.ModuleList()
        for cfg in cfgs:
            if cfg == "self_attn":
                self_attn = DeforAttn(embed_dims, num_heads, num_points, dropout, max_num_levels=1)
                block_layers.append(self_attn)
            elif cfg == "ffn":
                ffn = FFN(embed_dims, feedforward_channels = embed_dims*4)
                block_layers.append(ffn)
            elif cfg == "cross_attn":
                cross_attn = DeforAttn(embed_dims, num_heads, num_points, dropout, max_num_levels=max_num_levels)
                block_layers.append(cross_attn)
            elif cfg == "norm":
                block_layers.append(nn.LayerNorm(embed_dims))
        self.cfgs = cfgs
        self.block_layers = block_layers
    
    def forward(self, query, query_pos, value, ref_2d, spatial_shapes):
        """_summary_

        Args:
            query: [1, H*W, C]
            query_pos: [1, H*W, C]
            value: [N, H*W, C]
            ref_2d: [1, H*W, N, 2]
            spatial_shapes: [N, 2]

        Returns:
            _type_: _description_
        """
        for layer_type,  layer in zip(self.cfgs, self.block_layers):
            if layer_type == "self_attn":
                query = layer(query, query_pos, query, ref_2d[:, :, :1, :], spatial_shapes[:1, :])
            elif layer_type == "ffn" or layer_type == "norm":
                query = layer(query)
            elif layer_type == "cross_attn":
                query = layer(query, query_pos, value, ref_2d, spatial_shapes[:1, :])
        return query, query_pos, value, ref_2d, spatial_shapes