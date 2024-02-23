# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.defor_att_bev_backbone import DeforBEVBackbone
from opencood.models.sub_modules.defor_bev_backbone_resnet import DeforResNetBEVBackbone
from opencood.models.meta_flow import MetaFlow

class PointPillarDeformable(nn.Module):
    def __init__(self, args):
        super(PointPillarDeformable, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = DeforResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = DeforBEVBackbone(args['base_bev_backbone'], 64)
        self.cls_head = nn.Conv2d(args['head_embed_dims'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_embed_dims'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'train_stage' in args:
            self.train_stage = args['train_stage']
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(args['head_embed_dims'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        self.supervise_single = args['supervise_single']
        if 'calibrate' in args and args['calibrate']:
            self.calibrate = True
            # create calibrate model
            if self.train_stage == 'stage2':
                self.meta_flow = MetaFlow(args['meta_flow'])
        else:
            self.calibrate = False

        if 'use_seperate_head' in args and args['use_seperate_head']:
            self.use_seperate_head = True
            self.single_cls_head = nn.Conv2d(args['head_embed_dims'], args['anchor_number'],
                                        kernel_size=1)
            self.single_reg_head = nn.Conv2d(args['head_embed_dims'], 7 * args['anchor_number'],
                                    kernel_size=1)
            if self.use_dir:
                self.single_dir_head = nn.Conv2d(args['head_embed_dims'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1)
        else:
            self.use_seperate_head = False

    @torch.no_grad()
    def get_cav_pillar_features(self, cav_history_lidar, pairwise_t_matrix, batch_index):
        # cav_id, T, lidar; process all the batchs within one cav
        num_levels = self.backbone.num_levels
        multiscale_features = [[] for _ in range(num_levels)]
        for cav_id_index, cav_lidars in enumerate(cav_history_lidar):
            cav_lidars = self.pillar_vfe(cav_lidars)
            cav_lidars = self.scatter(cav_lidars) # spatial feature: T, C, H, W
            cav_lidars['pairwise_t_matrix'] = pairwise_t_matrix
            # process all the T frames within one batch
            T_frame_features = self.backbone.get_projected_bev_features(cav_lidars, batch_index, cav_id_index) # [ (t, c0, h0, w0), ( t, c1, h1, w1), (t, c2, h2, w2) ]
            for i in range(num_levels):
                multiscale_features[i].append(T_frame_features[i].unsqueeze(0))
        multiscale_features = [ torch.cat(x, dim=0) for x in multiscale_features] # [ (cav, t, c0, h0, w0), (cav, t, c1, h1, w1), (cav, t, c2, h2, w2) ]
        return multiscale_features

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        lidar_pose = data_dict['lidar_pose']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix,
                      'ego_ids': data_dict['ego_ids']}
        # Dair-v2x: 一个batch中，所有的inf折叠
        # V2X-sim: 一个batch中，一个协同场景的其他CAV_ID折叠， batch层面分开做
        if self.calibrate:
            assert 'calibrate_data' in data_dict
            # offset_GT [ num_agent*h*w*2, num_agent*h*w*2, ... ] # 不需要改变
            GT_offset_l = data_dict['calibrate_data']['offset']
            if self.train_stage == 'stage1':
                batch_dict.update({'offset_GT': GT_offset_l,
                                'pred_offset': None
                                })
            elif self.train_stage == 'stage2':
                # data_dict['calibrate_data']['lidar_history']: list [  [cav_id_1, cav_id_2, cav_id_3] , []]
                # time_delay [ tensor, ...]
                # 在V2X-sim中， batch -> cav_id -> T
                predicted_offset_l = []
                for batch_index, (batch_cav_history, time_delay) in enumerate(zip(data_dict['calibrate_data']['lidar_history'], data_dict['calibrate_data']['time_delay'])):
                    lidar_history_features = self.get_cav_pillar_features(batch_cav_history, pairwise_t_matrix, batch_index) # [ (cav, t, c0, h0, w0), (cav, t, c1, h1, w1), (cav, t, c2, h2, w2) ]
                    predicted_offset = self.meta_flow(lidar_history_features, time_delay)
                    predicted_offset_l.append(predicted_offset)
                batch_dict.update({'offset_GT': GT_offset_l,
                                'pred_offset': predicted_offset_l
                                })
        else:
            batch_dict.update({'offset_GT': None,
                               'pred_offset': None
                               })
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        fused_features = batch_dict['fused_features']

        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
        
        if self.calibrate and self.train_stage != 'stage1':
            GT_offsets = torch.concat(GT_offset_l, dim=0)
            pred_offsets = torch.concat(predicted_offset_l, dim=0)
            data_dict['label_dict'].update({'offset': GT_offsets})  # in-place change, save back to label_dict
            output_dict.update({'pred_offset': pred_offsets})
        
        if self.supervise_single:
            single_features = batch_dict['single_features']
            if self.use_seperate_head:
                psm_single = self.single_cls_head(single_features)
                rm_single = self.single_reg_head(single_features)
                if self.use_dir:
                    dir_single = self.single_dir_head(single_features)
            else:
                psm_single = self.cls_head(single_features)
                rm_single = self.reg_head(single_features)
                if self.use_dir:
                    dir_single = self.dir_head(single_features)

            output_dict.update({'cls_preds_single': psm_single,
                                'reg_preds_single': rm_single
                                })
            if self.use_dir:
                output_dict.update({'dir_preds_single': dir_single})

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_features)})
            
        return output_dict
    