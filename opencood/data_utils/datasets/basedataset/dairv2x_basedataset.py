import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import pickle
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.common_utils import merge_features_to_dict
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
try:
    from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
except:
    from spconv.utils import VoxelGenerator
from flow_pred.flow_2d_estimator import FLowScatter, collate_batch_dict, FlowEstimator

class DAIRV2XBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = read_json(split_dir)
        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info_processed_updated.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")  # 使用vehicle作为帧的ID
            self.co_data[veh_frame_id] = frame_info

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
        
        if "time_delay" in self.params:
            self.max_time_delay = params["time_delay"] # can only be [0, 1, 2, 3, 4, 5]
        else:
            self.max_time_delay = 0
        
        if "load_offset_map" in self.params:
            self.load_offset_map = params["load_offset_map"]
        else:
            self.load_offset_map = False
        
        if "bev_h" in self.params:
            self.bev_h = self.params["bev_h"]

        if "bev_w" in self.params:
            self.bev_w = self.params["bev_w"]
        
        self.past_frame = 5 # todo
        self.voxel_generator = VoxelGenerator(
            voxel_size=[0.4, 0.4, 3],
            point_cloud_range=[-100.8, -40, -1.5, 100.8, 40, 1.5],
            max_num_points=32,
            max_voxels=32000
        )
        self.scatter = FLowScatter()
    
    def reinitialize(self):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        
        # pose of agent 
        lidar_to_novatel = read_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world = read_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)
        # data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir, frame_info['cooperative_label_path']))
        data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'].replace("label_world", "label_world_backup"))) 
        data[0]['params']['vehicles_all'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'])) 

        data[1]['params']['vehicles_front'] = [] # we only load cooperative label in vehicle side
        data[1]['params']['vehicles_all'] = [] # we only load cooperative label in vehicle side

        if self.load_camera_file:
            data[0]['camera_data'] = load_camera_data([os.path.join(self.root_dir, frame_info["vehicle_image_path"])])
            data[0]['params']['camera0'] = OrderedDict()
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json')))
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json')))
            
            data[1]['camera_data']= load_camera_data([os.path.join(self.root_dir,frame_info["infrastructure_image_path"])])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json')))

        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            time_index = self.frame_select(frame_info)
            data[1]['lidar_np'] = self.load_lidar_time(time_index, frame_info)

            if self.load_offset_map:
                adjacent_flows = self.load_adjacent_flow(frame_info, time_index)
                data[0]['params']["adjacent_flows"] = adjacent_flows # None, when some frames are missing
                data[0]['params']["time_delay"] = time_index * 100 # bug2: 时间要乘以100， 预训练是干了的
                               
        # Label for single side； 路端是data[1], data[1]['params']['vehicles_single_all']是路端的标签，只是这里统一用了key
        data[0]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar_backup/{}.json'.format(veh_frame_id)))
        data[0]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))
        data[1]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))

        return data
    
    def load_adjacent_flow(self, frame_info, time_index):

        # time_index = self.frame_select(frame_info) # bug: 这里怎么又写了一次
        # load the flow 
        track_n_frame = self.max_time_delay + self.past_frame - 1
        previous_infs = [frame_info['previous_inf_'+str(i)] for i in range(1, track_n_frame + 1)]
        if None in previous_infs:
            return None #  when some frames are missing, we return a flag
        anchor_id = frame_info['infrastructure_pointcloud_path'].split('/')[-1].replace('.pcd', '')
        previous_ids = [previous_inf[0].split('/')[-1].replace('.pcd', '') for previous_inf in previous_infs]
        all_ids = [anchor_id] + previous_ids
        all_ids = all_ids[time_index:time_index+self.past_frame]
        # load backward flow between each frame
        adjacent_flows = []
        for i in range(0, len(all_ids)-1):
            with open(os.path.join(self.root_dir, 'adjacent_flow', all_ids[i] + '_' + all_ids[i+1] +'.pkl'), 'rb') as f:
                adjacent_flow = pickle.load(f)
            anchor_pos = pcd_utils.read_pcd(os.path.join(self.root_dir, 'infrastructure-side/velodyne/'+all_ids[i]+'.pcd'))[0]
            anchor_pos = FlowEstimator.filter_point_cloud(anchor_pos)[:, :3]
            adjacent_flow_2d = self.gen_2d_flow(anchor_pos, adjacent_flow) 
            adjacent_flows.append(adjacent_flow_2d)
        adjacent_flows = torch.cat(adjacent_flows, dim=1)
        return adjacent_flows

    def gen_2d_flow(self, pc_1, flow):
        # pc_1: N, 3
        # flow: N, 3
        # voxelize 3D flow into 2D
        assert pc_1.shape[0] == flow.shape[0]
        pc_flow = np.concatenate([pc_1, flow[:, :2]], axis=1)
        voxel_flow = self.voxel_generator.generate(pc_flow) # coord后来加了batch
        voxel_flow['voxel_features'] = \
            (voxel_flow['voxels'][:, :, 3:].sum(axis=1)) / \
                (voxel_flow['num_points_per_voxel'].astype(np.float32).reshape(-1, 1))

        voxel_flow['voxel_features'] = voxel_flow['voxel_features'] * 2.5  # lidar meters -> grid  # todo: magic number

        voxel_flow = merge_features_to_dict([voxel_flow])
        voxel_flow = collate_batch_dict(voxel_flow)
        # scatter
        flow_2d = self.scatter(voxel_flow)
        flow_2d = torch.flip(flow_2d, dims=[2])
        return flow_2d 
    
    def load_lidar_time(self, time_index, frame_info):
        if not self.is_frame_exits(time_index, frame_info):
            return None
        if time_index == 0:
            lidar_np, _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))
        else:
            lidar_np, _ = \
                    pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["previous_inf_"+str(time_index)][0]))
        return lidar_np
    
    @staticmethod
    def is_frame_exits(index, frame_info):
        if index == 0:
            return True
        # if index > 5:
        #     return False
        if frame_info["previous_inf_"+str(index)] is not None:
            return True
        return False
    
    def frame_select(self, frame_info):
        if self.max_time_delay == 0:
                return 0
        if self.train:
            # 训练时随机选择 [0, max_time_delay]
            while True:
                # time_index = random.randint(0, self.max_time_delay) # 随机在最大time_delay中选一帧
                time_index = np.floor(np.random.exponential(scale=2.0)).astype(np.int32) # 均值为2的指数分布
                if time_index > self.max_time_delay:
                    time_index = self.max_time_delay
                if self.is_frame_exits(time_index, frame_info):
                    return time_index
                else:
                    continue   
        else:
            # 测试时，选择max_time_delay, 没有就再找， 验证应该验证所有的
            for time_index in range( self.max_time_delay, -1, -1):
                if self.is_frame_exits(time_index, frame_info):
                    return time_index
    


    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        pass


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from opencood.hypes_yaml import yaml_utils
    hypes = yaml_utils.load_yaml("./opencood/hypes_yaml/dairv2x/lidar_time_delay/pointpillar_deformable_attn_w_end2ned_finetune.yaml")
    opencood_train_dataset = DAIRV2XBaseDataset(hypes, visualize=True, train=True)
    # train_loader = DataLoader(opencood_train_dataset,
    #                           batch_size=hypes['train_params']['batch_size'],
    #                           num_workers=4,
    #                           collate_fn=opencood_train_dataset.collate_batch_train,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           drop_last=True,
    #                           prefetch_factor=2) 
    # for i, data in enumerate(opencood_train_dataset):
    #     if i == 646:
    #         i = 646
    #         pass
    #     print(i)
    #     pass
    for i in range(len(opencood_train_dataset)):
        print(i)
        opencood_train_dataset.retrieve_base_data(i)
