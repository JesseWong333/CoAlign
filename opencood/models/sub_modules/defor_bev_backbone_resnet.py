"""
Resblock is much strong than normal conv

Provide api for multiscale intermeidate fuion
"""

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.defor_encoder_multi_scale import DeforEncoderMultiScale
from opencood.models.sub_modules.defor_encoder import DeforEncoder
from opencood.models.sub_modules.defor_encoder_multi_scale_single_agent import DeforEncoderMultiScaleSingleAgent
from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.SyncLSTM import SyncLSTM

DEBUG = False

class DeforResNetBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.model_cfg = model_cfg

        self.discrete_ratio = model_cfg['voxel_size'][0]
        self.downsample_rate = 1

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        self.resnet = ResNetModified(BasicBlock, 
                                        layer_nums,
                                        layer_strides,
                                        num_filters,
                                        inplanes = model_cfg.get('inplanes', 64))

        num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        
        if "multi_scale" in model_cfg and model_cfg['multi_scale'] :
            self.multi_scale = True
            self.defor_encoder = DeforEncoderMultiScale(model_cfg['defor_encoder'])
            self.defor_encoder_single = DeforEncoderMultiScaleSingleAgent(model_cfg['defor_encoder_single'])
        else:
            self.multi_scale = False
            self.defor_encoder = DeforEncoder(model_cfg['defor_encoder'])

        if 'calibrate' in model_cfg and model_cfg['calibrate']:
            if model_cfg['train_stage'] == 'stage2':
                self.syncnet = SyncLSTM(channel_size = 64, spatial_size = (100, 252), k = 3)

        # project multi-feature to the same dimention
        if self.multi_scale:
            input_proj_list = []
            for i, _ in enumerate(range(num_levels)):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=1),  # we use the same size filters as the privious upsample filters
                    nn.BatchNorm2d(num_upsample_filters[i]),
                ))

            self.input_proj = nn.ModuleList(input_proj_list)

    def get_normalized_transformation(self, H, W, pairwise_t_matrix_c):
        pairwise_t_matrix = pairwise_t_matrix_c.clone() # avoid in-place
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2
        return pairwise_t_matrix

    def featch_time_features(self, aligned_features, time_delay):
        # aligned_flow [delay_0, delay_1,  ....]
        b = time_delay.shape[0]
        aligned_feature_b = []
        for i in range(b):
            time_delay_ = time_delay[i]  # 当前batch的 time delay
            aligned_feature = aligned_features[time_delay_][i].unsqueeze(0) # 1, c, h, w
            aligned_feature_b.append(aligned_feature)
        aligned_feature_b = torch.cat(aligned_feature_b, dim=0)
        return aligned_feature_b

    def generate_aligned_features(self, history_features, time_delays):
        # history_features: [ (b, t, c, h, w), (b, t, c, h, w), ...] # t的第一维是GT
        # time_delays: [(b), (b), ...]

        history_features_stride2 = []
        for history_feature in history_features: # DAIR都是inf的，已经对齐了;
            b, t, c, h, w = history_feature.shape
            history_feature_ = history_feature.flatten(start_dim=0, end_dim=1)
            history_feature_ = self.resnet.layer0(history_feature_)
            history_feature_ = history_feature_.reshape(b, t, c, h//2, w//2)
            history_features_stride2.append(history_feature_)
        
        cav_aligned_features = []
        cav_aligned_features_gt = []
        for features_strides2, time_delay in zip(history_features_stride2, time_delays):
            max_time_delay = time_delay.max()
            aligned_gt = features_strides2[:, 0]
            cav_aligned_features_gt.append(aligned_gt)
            if max_time_delay == 0:
                pred_aligned_flow = []
            else: 
                pred_aligned_flow = self.syncnet(features_strides2[:, 1:], [max_time_delay])  # [(b, c, h, w), (b, c, h, w)]; sync里面把顺序改了
            all_aligned_features = [aligned_gt] + pred_aligned_flow
            align_feature_ = self.featch_time_features(all_aligned_features, time_delay)
            cav_aligned_features.append(align_feature_)

        return cav_aligned_features[0], cav_aligned_features_gt[0]
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        
        H, W = spatial_features.shape[2:]
        pairwise_t_matrix = self.get_normalized_transformation(H, W, pairwise_t_matrix)

        # b, c, h, w
        aligned_features_inf, aligned_features_inf_gt = self.generate_aligned_features(data_dict['history_features'], data_dict['time_delays'])
        data_dict['aligned_features'] = aligned_features_inf
        data_dict['aligned_features_GT'] = aligned_features_inf_gt

        aligned_features_inf_2 = self.resnet.layer1(aligned_features_inf)
        aligned_features_inf_3 = self.resnet.layer2(aligned_features_inf_2) # b, c, h, w

        # 重新把对齐的feature 放回去
        x = self.resnet(spatial_features)  # tuple of features [ (b*2, C, H, W), (b*2, C, H, W), (b*2, C, H, W)]
        b = aligned_features_inf.shape[0]
        for i in range(b):
            x[0][i*2+1] = aligned_features_inf[i]
            x[1][i*2+1] = aligned_features_inf_2[i]
            x[2][i*2+1] = aligned_features_inf_3[i]
            
        ups = []
        ups_multi_scale = []
        for i in range(self.num_levels):
            if self.multi_scale:
                ups_multi_scale.append(self.input_proj[i](x[i])) # project all the features to the same dimmension
            else:
                if len(self.deblocks) > 0:
                    ups.append(self.deblocks[i](x[i]))  # upsample a feature
                else:
                    ups.append(x[i])
 
        if self.multi_scale:
            single_features = self.defor_encoder_single(ups_multi_scale)
            fused_features = self.defor_encoder(ups_multi_scale, record_len, pairwise_t_matrix, None, None) # dim=128
            data_dict['single_features'] = single_features
            data_dict['fused_features'] = fused_features
        else:
            if len(ups) > 1:
                x = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x = ups[0]
            if len(self.deblocks) > self.num_levels:
                x = self.deblocks[-1](x)
            data_dict['single_features'] = x
            fused_features = self.defor_encoder(x, record_len, pairwise_t_matrix, data_dict['offset_GT'], data_dict['pred_offset'])  # dim=384
            data_dict['fused_features'] = fused_features

        return data_dict


    