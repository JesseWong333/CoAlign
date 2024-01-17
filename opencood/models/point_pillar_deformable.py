# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.defor_att_bev_backbone import DeforBEVBackbone


class PointPillarDeformable(nn.Module):
    def __init__(self, args):
        super(PointPillarDeformable, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = DeforBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 3, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        self.supervise_single = args['supervise_single']
        if 'calibrate' in args['base_bev_backbone']['defor_encoder']:
            self.calibrate = args['base_bev_backbone']['defor_encoder']['calibrate']
        else:
            self.calibrate = False

        if 'use_seperate_head' in args and args['use_seperate_head']:
            self.use_seperate_head = True
            self.single_cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                        kernel_size=1)
            self.single_reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                    kernel_size=1)
            if self.use_dir:
                self.single_dir_head = nn.Conv2d(128 * 3, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1)
        else:
            self.use_seperate_head = False

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        lidar_pose = data_dict['lidar_pose']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        if 'offset' in data_dict:
            offset = data_dict['offset']
            offset_mask = data_dict['offset_mask']
        else:
            offset = None
            offset_mask = None

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix,
                      'offset': offset,
                      'offset_mask': offset_mask}
            
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        batch_dict = self.backbone(batch_dict)

        if self.calibrate:
            fused_features, coord_predictions = batch_dict['fused_features']
        else:
            fused_features = batch_dict['fused_features']

        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
        
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
        if self.calibrate:
            output_dict.update({'calibrate': coord_predictions})
            
        return output_dict
    