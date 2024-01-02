# Copied from SFPFlow net

import torch
import torch.nn as nn
import numpy as np
from flow_pred.model import SPFlowNet
from opencood.utils.common_utils import merge_features_to_dict
try:
    from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
except:
    from spconv.utils import VoxelGenerator

 
def collate_batch_dict(batch):
    voxel_features = \
        torch.from_numpy(np.concatenate(batch['voxel_features']))
    voxel_num_points = \
        torch.from_numpy(np.concatenate(batch['num_points_per_voxel']))
    coords = batch['coordinates']
    voxel_coords = []

    for i in range(len(coords)):
        voxel_coords.append(
            np.pad(coords[i], ((0, 0), (1, 0)),
                    mode='constant', constant_values=i))
    voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

    return {'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points}

class FLowScatter(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.num_bev_features = 2
        self.nx, self.ny, self.nz = [504, 200, 1]  # [704, 200, 1] 

        assert self.nz == 1

    def forward(self, batch_dict):
        """ 将生成的pillar按照坐标索引还原到原空间中
        Args:
            pillar_features:(M, 64)
            coords:(M, 4) 第一维是batch_index

        Returns:
            batch_spatial_features:(4, 64, H, W)
            
            |-------|
            |       |             |-------------|
            |       |     ->      |  *          |
            |       |             |             |
            | *     |             |-------------|
            |-------|

            Lidar Point Cloud        Feature Map
            x-axis up                Along with W 
            y-axis right             Along with H

            Something like clockwise rotation of 90 degree.

        """
        pillar_features, coords = batch_dict['voxel_features'], batch_dict[
            'voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
            # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
            pillars = pillars.t() # (64,batch_idx_voxel)
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
            batch_spatial_features.append(spatial_feature) 

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
        # batch_dict['spatial_features'] = batch_spatial_features

        return batch_spatial_features
    
class FlowEstimator():

    def __init__(self, args, model_file, return_filtered_points=True) -> None:
        self.return_filtered_points = return_filtered_points
        model = SPFlowNet(args)
        model.load_state_dict(torch.load(model_file))
        self.flow_model = model.cuda()
        self.flow_model.eval()
        self.voxel_generator = VoxelGenerator(
            voxel_size=[0.4, 0.4, 3],
            point_cloud_range=[-100.8, -40, -1.5, 100.8, 40, 1.5],
            max_num_points=32,
            max_voxels=32000
        )
        self.scatter = FLowScatter()

    @staticmethod
    def filter_point_cloud(pcb_np, pc_range=[-100.8, -40, -1.5, 100.8, 40, 1.5]):
        # -3.5已经比较低了，路端高5m, 人车高1到2m,信息都在
        pcb_filtered = pcb_np[ (pcb_np[:,0] > pc_range[0]) &  (pcb_np[:,0] < pc_range[3]) & (pcb_np[:,1] > pc_range[1]) & (pcb_np[:,1] < pc_range[4]) & (pcb_np[:,2] > pc_range[2]) & (pcb_np[:,2] < pc_range[5])]
        # print("trimed_point{}".format(  (pcb_np.shape[0]-pcb_filtered.shape[0])/pcb_np.shape[0]  ))
        return pcb_filtered
    
    def get_flow(self, pos1, pos2):
        pos1 = pos1[:, 0:3]
        pos2 = pos2[:, 0:3]

        # if filter_points:
        #     pos1 = self.filter_point_cloud(pos1)
        # # 第二个始终filter
        pos2 = self.filter_point_cloud(pos2)

        num_points = pos1.shape[0]

        # n1 = pos1.shape[0]
        # if n1 > num_points:
        #     sample_idx1 = np.random.choice(n1, num_points, replace=False)
        #     pos1 = pos1[sample_idx1, :]

        n2 = pos2.shape[0]
        if n2 > num_points:
            sample_idx2 = np.random.choice(n2, num_points, replace=False)
            pos2 = pos2[sample_idx2, :]

        # pos1_center = np.mean(pos1, 0)
        # pos1 -= pos1_center
        # pos2 -= pos1_center
        
        norm1 = torch.from_numpy(np.ones_like(pos1)).unsqueeze(0).cuda()
        norm2 = torch.from_numpy(np.ones_like(pos2)).unsqueeze(0).cuda()

        pos1_torch = torch.from_numpy(pos1).unsqueeze(0).cuda()
        pos2_torch = torch.from_numpy(pos2).unsqueeze(0).cuda()
        
        with torch.no_grad():
            pred_flows, _ = self.flow_model(pos1_torch, pos2_torch, norm1, norm2)
        
        flow_2d = self.gen_2d_flow(pos1_torch, pred_flows[-1]).squeeze().permute(1, 2, 0).cpu().numpy() 
        pred_flows = pred_flows[-1].squeeze(0).cpu().numpy()
        if self.return_filtered_points:
            return pos2, pred_flows, flow_2d
        else:
            return flow_2d

    def gen_2d_flow(self, pc_1, flow):
        # pc_1: B, N, 3
        # flow: B, N, 3
        # project 3D flow into 2D
        pc_flow = torch.cat([pc_1, flow[:, :, :2]], dim=-1).cpu().numpy().squeeze(0)
        voxel_flow = self.voxel_generator.generate(pc_flow) # coord后来加了batch
        # voxel_flow['voxel_features'] = voxel_flow['voxels'][:, :, 3:].max(axis=1) * 2.5  # lidar meters -> grid
        voxel_flow['voxel_features'] = \
            (voxel_flow['voxels'][:, :, 3:].sum(axis=1)) / \
                (voxel_flow['num_points_per_voxel'].astype(np.float32).reshape(-1, 1))

        voxel_flow['voxel_features'] = voxel_flow['voxel_features'] * 2.5  # lidar meters -> grid

        voxel_flow = merge_features_to_dict([voxel_flow])
        voxel_flow = collate_batch_dict(voxel_flow)
        # scatter
        flow_2d = self.scatter(voxel_flow)
        flow_2d = torch.flip(flow_2d, dims=[2])
        return flow_2d

if __name__ == '__main__':
    import cv2
    import cmd_args
    import sys
    import opencood.utils.pcd_utils as pcd_utils
    from custom_vis import convert_lidar_to_BEV_image, draw_flow

    def convert2img(point_cuda):
        point = point_cuda.squeeze().cpu().numpy()
        img = convert_lidar_to_BEV_image(point).canvas
        return img
    
    def vis_2d_flow(img, flow_2d):
        # img: 800, 2016, 3
        # flow_2d:
        flow_2d = flow_2d.squeeze().permute(1, 2, 0).cpu().numpy() # flow_2d: 200, 504, 2: flow_2d是从左下角为原点
        flow_2d = cv2.flip(flow_2d, flipCode=0) # Warning! 要不要翻转很容易出错，对不上
        flow_2d = cv2.resize(flow_2d, (2016, 800)) * 4
        flow_2d_vis = draw_flow(img, flow_2d)
        # flow_2d_vis = show_flow_hsv(flow_2d)
        return flow_2d_vis

    model_file = "./experiment_occ/SPFlowNet-KITTI_r-2023-12-15_16-24/checkpoints/SPFlowNet_011_0.0625.pth"
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    flowEstimator = FlowEstimator(args, model_file)

    pos1, _ = pcd_utils.read_pcd("sample_data/003135.pcd")
    pos2, _ = pcd_utils.read_pcd("sample_data/003137.pcd")
    pos1, pos2, flow, flow_2d = flowEstimator.get_flow(pos1, pos2)
    recover_pos2 = pos1 + flow
    
    pos1_img = convert2img(pos1)
    pos2_img = convert2img(pos2)
    recover_pos2_img = convert2img(recover_pos2)

    flow_2d_img = vis_2d_flow(pos1_img, flow_2d)
    # save
    cv2.imwrite("tmp/pos1_img.png", pos1_img)
    cv2.imwrite("tmp/pos2_img.png", pos2_img)
    cv2.imwrite("tmp/recover_pos2_img.png", recover_pos2_img)
    cv2.imwrite("tmp/flow_2d_img.png", flow_2d_img)

