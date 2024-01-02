import numpy as np
import torch
import torch.nn as nn
import math
from mmdet.models.utils import LearnedPositionalEncoding
from .deformable_attention import Block

class SineTimeEncoding(nn.Module):
    """Time encoding with sine and cosine functions.
    """

    def __init__(self,
                 num_feats,
                 temperature=10000):
        super(SineTimeEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature

    def forward(self, time, device, shape=None):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        bs = time.shape[0]
        time = time.view(bs, 1, 1) * torch.ones(shape).to(device)
        
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = time.unsqueeze(-1) / dim_t
     
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        pos_x = torch.concat(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3)

        return pos_x

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class CNNFlowPred(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # 共三个block, stride来降维的
        self.conv1_1 = convrelu(input_channels, 32)
        self.conv1_2 = convrelu(32, 32)
        self.conv1_3 = convrelu(32, 32)

        self.conv2_1 = convrelu(32, 64, stride=2)
        self.conv2_2 = convrelu(64, 64)
        self.conv2_3 = convrelu(64, 64)

        self.conv3_1 = convrelu(64, 128, stride=2)
        self.conv3_2 = convrelu(128, 128)
        self.conv3_3 = convrelu(128, 128)

        self.deconv1 = deconv(128, 64)  # upscale

        self.out1 = convrelu(128, 128)
        self.out2 = nn.Conv2d(128, 2, 3, padding=1)

    def forward(self, x):
        # x: B, C, 200, 504

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        block1_out = self.conv1_3(x)

        #
        x = self.conv2_1(block1_out)
        x = self.conv2_2(x)
        block2_out = self.conv2_3(x)

        x = self.conv3_1(block2_out)
        x = self.conv3_2(x)
        block3_out = self.conv3_3(x)

        upscale = self.deconv1(block3_out)

        x = torch.cat((block2_out, upscale), dim=1)  # B, 128, H, C

        # x = self.out1(x)
        # x = self.out2(x)

        return x

# 在原来的CNN基础上加上decoder
class FlowEncoderDecoder(nn.Module):

    def __init__(self, input_channels, embed_dims):
        super().__init__()
        self.encoder = CNNFlowPred(input_channels=input_channels)

        self.bev_h = 100
        self.bev_w = 252

        self.positional_encoding = LearnedPositionalEncoding(        
            num_feats=embed_dims//2,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w)
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, embed_dims)
        self.temporal_encoding = SineTimeEncoding(embed_dims)
        
        # decoder block
        self.block1 = Block(embed_dims, num_heads=4, num_points=8, dropout=0.1, max_num_levels=1, 
                            cfgs = ["self_attn", "norm", "cross_attn", "norm", "ffn", "norm"])
        self.block2 = Block(embed_dims, num_heads=4, num_points=8, dropout=0.1, max_num_levels=1, 
                            cfgs = ["self_attn", "norm", "cross_attn", "norm", "ffn", "norm"])
        self.block3 = Block(embed_dims, num_heads=4, num_points=8, dropout=0.1, max_num_levels=1, 
                            cfgs = ["self_attn", "norm", "cross_attn", "norm", "ffn", "norm"])
        
        self.classifier = nn.Linear(embed_dims, 2)

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
        
    def forward(self, x, time):
        # time: N
        
        xx = self.encoder(x) # N, C, H, W
        N, C, H, W = xx.shape
        xx = xx.view(N, C, -1).permute(0, 2, 1)

        temporal_encoding = self.temporal_encoding(time, x.device, [N, H, W]).view(N, H*W, -1) # N, H, W, C
        xx = xx + temporal_encoding

        bev_queries = self.bev_embedding.weight.to(xx.dtype).repeat(N, 1, 1) # h*w, C -> N, h*w, C
        bev_mask = torch.zeros((N, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(xx.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(xx.dtype).flatten(2, 3).permute(0, 2, 1) # N, C, h*w -> N, h*w, C

        # bev_pos = bev_pos + temporal_encoding

        ref_2d = self.get_reference_points(
                H, W, dim='2d', bs=N, device=xx.device, dtype=xx.dtype)
        
        spatial_shapes = [(H, W)] * N  # the values has two levels with the same shape
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=xx.device) # N*2
        
        # query: [N, H*W, C]
        # query_pos: [N, H*W, C]
        # value: [N, H*W, C]  # 这个可能要改吧
        # ref_2d: [N, H*W, 1, 2]
        # spatial_shapes: [N, 2]
        bev_queries, bev_pos, xx, ref_2d, spatial_shapes = self.block1(bev_queries, bev_pos, xx, ref_2d, spatial_shapes)
        bev_queries, bev_pos, xx, ref_2d, spatial_shapes = self.block2(bev_queries, bev_pos, xx, ref_2d, spatial_shapes)
        bev_queries, bev_pos, xx, ref_2d, spatial_shapes = self.block3(bev_queries, bev_pos, xx, ref_2d, spatial_shapes)
        x = self.classifier(bev_queries)
        x = x.view(N, H, W, 2)
        return x

if __name__ == "__main__":
    model = FlowEncoderDecoder(input_channels=4, embed_dims=128)
    x = torch.rand(8, 4, 200, 504)
    time = torch.Tensor([100, 200, 0, 300, 500, 560, 700, 790])
    y = model(x, time)
    pass
   