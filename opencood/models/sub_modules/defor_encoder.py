# modified by Junjie Wang
# from BEVformer 

# 采用最小化的写法, 参照vis的写法
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from opencood.utils.mmcv_utils import constant_init, xavier_init
# from multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32 as MultiScaleDeformableAttnFunction
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch # 使用pytorch版本， cuda算子在部分机器上编译出问题
from mmdet.models.utils import LearnedPositionalEncoding
from flow_pred.CNN_flow_pred2 import FlowEncoderDecoder

def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False, return_mask=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    out = F.grid_sample(src, grid, padding_mode=padding_mode, align_corners=align_corners)
    # use the grid to generate a mask
    if return_mask:
        mask = (grid > 1) | (grid < -1)
        mask = mask[:, :, :, 0] | mask[:, :, :, 1]
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2)
        return out, mask
    else:
        return out
    
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

        Returns:
            _type_: _description_
        """
        identity = query

        query = query + query_pos  # query_pos是每次都加吗,是的

        N, num_value, C = value.shape
        
        value = self.value_proj(value)
        value = value.unsqueeze(0).contiguous().view(1, N*num_value, self.num_heads,-1) # [1, N*H*W, n_head, C//n_head]

        bs, num_query, C = query.shape
        
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.max_num_levels, self.num_points, 2)  # 
        sampling_offsets = sampling_offsets[:, :, :, :N, :, :].contiguous() # [1, H*W, n_head, 2, n_point, 2]
        
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(bs, num_query,
                                                self.num_heads,
                                                self.max_num_levels,
                                                self.num_points)
        attention_weights = attention_weights[:, :, :, :N, :]
         
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, N*self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                self.num_heads,
                                                N,
                                                self.num_points).contiguous() # [1, H*W, n_head, 2, n_point]

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
                query = layer(query, query_pos, value, ref_2d, spatial_shapes)
        return query, query_pos, value, ref_2d, spatial_shapes


class DeforEncoder(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()

        block_cfgs = model_cfg["block_cfgs"]
        for block_cfg in block_cfgs:
            self.blocks.append(Block(*block_cfg))

        # self.blocks = nn.Sequential(blocks) # Sequential只支持单输入，需自己解包

        self.bev_h = model_cfg["bev_h"] # 100
        self.bev_w = model_cfg["bev_w"] # 252
        self.embed_dims = model_cfg["embed_dims"]  # 384 按照原来的大小设置
        self.max_num_features = model_cfg["max_num_features"]

        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
        self.positional_encoding = LearnedPositionalEncoding(        
            num_feats=self.embed_dims//2,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w)
        
        self.feature_embeds = nn.Parameter(
            torch.Tensor(self.max_num_features, self.embed_dims))
        
        if "calibrate" in model_cfg:
            self.calibrate = model_cfg["calibrate"]
        else:
            self.calibrate = False
        if self.calibrate:
            self.flow_pred = FlowEncoderDecoder(8, 128) # todo, magic number

    
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA. Copied from BEVformer.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)  # 先w, 再H
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    @staticmethod
    def create_mesh_grid(bs, H, W, device):
        ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        -H//2, H//2-1, H, dtype=torch.float32),
                    torch.linspace(
                        -W//2, W//2-1, W, dtype=torch.float32)
                )
        ref_2d = torch.cat((ref_x.unsqueeze(-1), ref_y.unsqueeze(-1)), -1).unsqueeze(0)
        ref_2d = ref_2d.repeat(bs, 1, 1, 1) # bs, H, W, 2
        return ref_2d.to(device)  

    @staticmethod
    def applyTransform(points, T):
        # points: bs*N*2
        # T: bs*2*3
        bs, N, _ = points.shape
        T = T.float() # float64 -> float32 
        T_pad = torch.Tensor([0, 0, 1]).float().to(points).unsqueeze(0).repeat(bs, 1, 1) # 1*3 -> 1,1,3 -> N, 1, 3
        T = torch.cat([T, T_pad], dim=1)
        
        ones = torch.ones((bs, N, 1)).to(points)
        P = torch.cat((points, ones), dim=-1)
        
        projected_points = (T@(P.transpose(1, 2))).transpose(1, 2)[:, :, :2] # (bs, 3, 2) @ (bs, 3, N)
        return projected_points
    
    def project_flow(self, flow, T_forward, T_backward):
        # flow: Bs, H, W, 2
        # T_forward: bs*2*3
        # T_backward: bs*2*3, affine grid level
        bs, H, W, _ = flow.shape
        mesh_grid = self.create_mesh_grid(bs, H, W, flow.device)  # torch: bs, H, W, 2
        flow = flow + mesh_grid
        # Step1: project the content
        flow = self.applyTransform(flow.view(bs, H*W, 2), T_forward).reshape(bs, H, W, 2).permute(0, 3, 1, 2) # --> bs, 2, H, W
        # Step 2: affine grid
        affined_flow, mask = warp_affine_simple(flow, T_backward, [H, W], return_mask=True)
        affined_flow = affined_flow.permute(0, 2, 3, 1) # bs, 2, H, W ->  bs, H, W, 2
        affined_flow = affined_flow - mesh_grid 
        affined_flow[mask] = 0
        return affined_flow
    
    def forward(self, x, record_len, pairwise_t_matrix, pairwise_t_matrix_forward, adjacent_flows, time_delay):
        # 这里生成ref points
        # 可以融合多个级别,不同大小的feature map, 这里只取最后的一个
           
        split_x = self.regroup(x, record_len)
        C, H, W = split_x[0].shape[1:]
        
        # 预测参考点
        if self.calibrate:
            # 预测参考点
            flows = self.flow_pred(adjacent_flows, time_delay)  # [bs, H, W, 2]
            # project flow; 投影了之后flow里面的数值不会变； 怎么投
            # map: 同样使用warp_affine_simple； 里面的每个值二维， 以自己为中心旋转
            flows = self.project_flow(flows, pairwise_t_matrix_forward[:, 1, 0, :, :], pairwise_t_matrix[:, 0, 1, :, :])  # [bs, H, W, 2], forward: inf->ego; backward: ego->inf
    
        out = []
        for b, xx in enumerate(split_x):
            # input: xx: N, C, H, W; 其中N可能变化
            N = xx.shape[0] # N is dynamic

            bev_queries = self.bev_embedding.weight.to(xx.dtype)  # H*W, C
            bev_queries = bev_queries.unsqueeze(0) #  [1, H*W, C]
            bev_mask = torch.zeros((1, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(xx.dtype)
            bev_pos = self.positional_encoding(bev_mask).to(xx.dtype) # [1, num_feats*2, h, w]
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1) # [1, C, h*w]->[1, h*w, C]

            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            i = 0  # ego
            xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))

            ref_2d = self.get_reference_points(
                H, W, dim='2d', bs=1, device=split_x[0].device, dtype=split_x[0].dtype)

            if self.calibrate and N > 1:
                # ref_2d: 1, H*W, 1, 2
                flow = flows[b].view(self.bev_h*self.bev_w, 2).unsqueeze(0).unsqueeze(2) # H, W, 2 -> [1, H*W, 1, 2]
                ref_2d_calibrate = ref_2d + flow  # b, 
                ref_2d = torch.cat([ref_2d, ref_2d_calibrate], dim=2)
            else:
                ref_2d = ref_2d.repeat(1, 1, N,1) # (1, H*W, N, 2)
            # 使用真值参考点 offsets = transformed_points - points
            # if offsets is not None and offset_masks is not None and N > 1:
            #     offset = offsets[b].view(self.bev_h*self.bev_w, 2).unsqueeze(0).unsqueeze(2)  # [1, H*W, 1, 2]
            #     offset_mask = offset_masks[b].view(self.bev_h*self.bev_w) # [H*W] 为True的地方遮掩
            #     offset_mask_index = offset_mask.nonzero().squeeze(-1)
            #     ref_2d_ = ref_2d + offset
            #     ref_2d_[:, offset_mask_index, :, :] = -1e6
            #     ref_2d = torch.cat([ref_2d, ref_2d_], dim=2)
            # else:
            #     ref_2d = ref_2d.repeat(1, 1, N,1)

            xx += self.feature_embeds[:N, :, None, None]  # [N, C, H, W] + [N, C]-> [N, C, 1, 1]

            spatial_shapes = [(H, W)] * N  # the values has two levels with the same shape
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=xx.device) # N*2

            xx = xx.view(N, C, -1).permute(0, 2, 1) # N, H*W, C

            # h = self.att(xx[:1, ...], xx, xx, ref_2d, spatial_shapes)  # 切片保持维度
            # 
            for _, block in enumerate(self.blocks):
                bev_queries, bev_pos, xx, ref_2d, spatial_shapes = block(bev_queries, bev_pos, xx, ref_2d, spatial_shapes)  # [1, h*w, C]
            
            bev_queries = bev_queries.permute(0, 2, 1).view(1, C, H, W)  # 就是这个问题，其他的不行也是因为我没有permute
            out.append(bev_queries)
       
        return torch.cat(out, dim=0)

    