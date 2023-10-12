# 如何生成标注呢？ 在2D视角下去做
# 1） 用检测的3D框标记
# 2） 用固定的坐标
# 3） 在俯视角度下，类似于2D图像进行标注
# 先尝试下可不可以单独训练，不end-2-end训练

# 有意思的是，车端的点云可以完全对上标记，路端的有一点点偏差

import numpy as np
import os
import torch
import json
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils import box_utils as box_utils
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix

data_dir = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure"

def get_calibs(calib_path):
    calib = read_json(calib_path)
    if 'transform' in calib.keys():
        calib = calib['transform']
    rotation = calib['rotation']
    translation = calib['translation']
    return rotation, translation

def rev_matrix(rotation, translation):
    rotation = np.matrix(rotation)
    rev_R = rotation.I
    rev_R = np.array(rev_R)
    rev_T = - np.dot(rev_R, translation)
    return rev_R, rev_T

def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation)

    return rotation, translation

def trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                    veh_novatel2world_path, system_error_offset=None):
    inf_lidar2world_r, inf_lidar2world_t = get_calibs(inf_lidar2world_path)  # r: rotation, t: translation
    if system_error_offset is not None:
        inf_lidar2world_t[0][0] = inf_lidar2world_t[0][0] + system_error_offset['delta_x']
        inf_lidar2world_t[1][0] = inf_lidar2world_t[1][0] + system_error_offset['delta_y']

    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
    inf_lidar2novatel_r, inf_lidar2novatel_t = mul_matrix(inf_lidar2world_r, inf_lidar2world_t,
                                                          veh_world2novatel_r, veh_world2novatel_t)

    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
    inf_lidar2lidar_r,  inf_lidar2lidar_t = mul_matrix(inf_lidar2novatel_r, inf_lidar2novatel_t,
                                                       veh_novatel2lidar_r, veh_novatel2lidar_t)

    return inf_lidar2lidar_r, inf_lidar2lidar_t  # inf雷达 到 veh的雷达

def trans_world_veh(veh_lidar2novatel_path, veh_novatel2world_path):
    # 世界坐标系投影到车端坐标系
    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_lidar2world_r, veh_lidar2world_t = mul_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t, veh_novatel2world_r, veh_novatel2world_t)
    veh_world2lidar_r, veh_world2lidar_t = rev_matrix(veh_lidar2world_r, veh_lidar2world_t)
    return veh_world2lidar_r, veh_world2lidar_t

def trans_world_inf(inf_lidar2world_path,  system_error_offset):
    # 世界坐标系投影到inf坐标系
    inf_lidar2world_r, inf_lidar2world_t = get_calibs(inf_lidar2world_path)  # r: rotation, t: translation
    if system_error_offset is not None:
        inf_lidar2world_t[0][0] = inf_lidar2world_t[0][0] + system_error_offset['delta_x']
        inf_lidar2world_t[1][0] = inf_lidar2world_t[1][0] + system_error_offset['delta_y']
    
    return rev_matrix(inf_lidar2world_r, inf_lidar2world_t)

def tfm_to_pose(tfm: np.ndarray):
    """
    turn transformation matrix to [x, y, z, roll, yaw, pitch]
    we use radians format.
    tfm is pose in transformation format, and XYZ order, i.e. roll-pitch-yaw
    """
    # There forumlas are designed from x_to_world, but equal to the one below.
    yaw = np.degrees(np.arctan2(tfm[1,0], tfm[0,0])) # clockwise in carla
    roll = np.degrees(np.arctan2(-tfm[2,1], tfm[2,2])) # but counter-clockwise in carla
    pitch = np.degrees(np.arctan2(tfm[2,0], ((tfm[2,1]**2 + tfm[2,2]**2) ** 0.5)) ) # but counter-clockwise in carla

    x, y, z = tfm[:3,3]
    return([x, y, z, roll, yaw, pitch])


def convert_tfm_matrix(rotation, translation):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = rotation
    matrix[:, 3][0:3] = np.array(translation)[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    return matrix

# -----------------------------------------------------------------------------------------------
def convert_json_tfm(json_file):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    translation = np.array(json_file["translation"])
    # translation[0][0] = translation[0][0] + system_error_offset["delta_x"]
    # translation[1][0] = translation[1][0] + system_error_offset["delta_y"]  #为啥有[1][0]??? --> translation是(3,1)的
    matrix[:, 3][0:3] = translation[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    return matrix

# 是对齐的，角度小于0.01度， 坐标平移小于0.01；
def are_dicts_equal(dict1, dict2):
    pose_1 = tfm_to_pose(convert_json_tfm(dict1))
    pose_2 = tfm_to_pose(convert_json_tfm(dict2))
    if abs(pose_1[4] - pose_2[4]) > 0.01 or abs(pose_1[0] - pose_2[0]) > 0.01:
        return False
    return True

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
    return my_json

def get_calib(inf_frame_path):
    inf_frame_id = inf_frame_path.split('/')[-1].replace('.pcd', '')
    virtuallidar_to_world = read_json(os.path.join(data_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
    return virtuallidar_to_world

def read_GT(json_file):
    # 读取target变成 N*8*3 的形式
    label_l = read_json(json_file)
    objects = []
    for object in label_l:
        objects.append(np.array(object["world_8_points"]))
    GT = np.stack(objects, axis=0)
    return GT

def vis_sample_at_veh_view(inf_frame_path, frame_info, save_index):
    # 将所有信息投影到ego车视角 去可视化
    #

    # 读Pcd
    inf_frame_id = frame_info['infrastructure_pointcloud_path'].split('/')[-1].replace('.pcd', '')
    veh_frame_id = frame_info['vehicle_pointcloud_path'].split('/')[-1].replace('.pcd', '')
    lidar_np, _ = pcd_utils.read_pcd(os.path.join(data_dir, inf_frame_path)) # N*4
    
    # 必须全部投影到车端做可视化，因为pc_range是相对于车的!!!
    inf_lidar2world_path = os.path.join(data_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json')
    veh_lidar2novatel_path = os.path.join(data_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json')
    veh_novatel2world_path = os.path.join(data_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json')
    system_error_offset= frame_info["system_error_offset"]
    inf_lidar2lidar_r, inf_lidar2lidar_t = trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path, veh_novatel2world_path, system_error_offset)
    inf2veh_transform = convert_tfm_matrix(inf_lidar2lidar_r, inf_lidar2lidar_t)

    lidar_np_projected = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], inf2veh_transform)

    # project_box3d, 世界坐标到车端
    veh_world2lidar_r, veh_world2lidar_t = trans_world_veh(veh_lidar2novatel_path, veh_novatel2world_path)
    world2veh_transform = convert_tfm_matrix(veh_world2lidar_r, veh_world2lidar_t)
    GT = read_GT(os.path.join(data_dir, frame_info['cooperative_label_path']))

    GT_projected = box_utils.project_box3d(GT, world2veh_transform)
    target = {}
    target["gt_box_tensor"] = torch.from_numpy(GT_projected)
    vis_save_path =  os.path.join('tmp', 'bev_{}_{}.png'.format(inf_frame_id, save_index))
    # 坐标明显不匹配，pcd的坐标和GT的坐标都几千了，为什么pc_range只有几百; 应该是只有几百的量级; 
    # 必须全部投影到车端做可视化，因为pc_range是相对于车的!!! todo
    simple_vis.visualize(target, torch.from_numpy(lidar_np_projected),
                                    [-100.8, -40, -3.5, 100.8, 40, 1.5],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)

def vis_sample_veh(frame_info, save_index='v'):
    inf_frame_id = frame_info['infrastructure_pointcloud_path'].split('/')[-1].replace('.pcd', '')
    veh_frame_id = frame_info['vehicle_pointcloud_path'].split('/')[-1].replace('.pcd', '')
    lidar_np, _ = pcd_utils.read_pcd(os.path.join(data_dir, frame_info['vehicle_pointcloud_path'])) # N*4
    
    veh_lidar2novatel_path = os.path.join(data_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json')
    veh_novatel2world_path = os.path.join(data_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json')
    veh_world2lidar_r, veh_world2lidar_t = trans_world_veh(veh_lidar2novatel_path, veh_novatel2world_path)
    world2veh_transform = convert_tfm_matrix(veh_world2lidar_r, veh_world2lidar_t)
    GT = read_GT(os.path.join(data_dir, frame_info['cooperative_label_path']))
    GT_projected = box_utils.project_box3d(GT, world2veh_transform)
    target = {}
    target["gt_box_tensor"] = torch.from_numpy(GT_projected)
    vis_save_path =  os.path.join('tmp', 'bev_{}_{}.png'.format(inf_frame_id, save_index))

    simple_vis.visualize(target, torch.from_numpy(lidar_np),
                                    [-100.8, -40, -3.5, 100.8, 40, 1.5],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)

def vis_sample_at_inf_view(inf_frame_path, frame_info, save_index):
    # 将所有信息投影到 inf 去展示
    # 只需要将3D box投影就好了
    inf_frame_id = frame_info['infrastructure_pointcloud_path'].split('/')[-1].replace('.pcd', '')
    veh_frame_id = frame_info['vehicle_pointcloud_path'].split('/')[-1].replace('.pcd', '')
    lidar_np, _ = pcd_utils.read_pcd(os.path.join(data_dir, inf_frame_path)) # N*4
    
    inf_lidar2world_path = os.path.join(data_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json')
    system_error_offset= frame_info["system_error_offset"]
    inf_world2lidar_r, inf_world2lidar_t = trans_world_inf(inf_lidar2world_path, system_error_offset)
    world2inf_transform = convert_tfm_matrix(inf_world2lidar_r, inf_world2lidar_t)
    
    GT = read_GT(os.path.join(data_dir, frame_info['cooperative_label_path']))
    GT_projected = box_utils.project_box3d(GT, world2inf_transform)
    target = {}
    target["gt_box_tensor"] = torch.from_numpy(GT_projected)
    vis_save_path =  os.path.join('tmp', 'bev_{}_{}.png'.format(inf_frame_id, save_index))

    simple_vis.visualize(target, torch.from_numpy(lidar_np),
                                    [-100.8, -40, -3.5, 100.8, 40, 1.5],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)
    
def label_flow(data):
    # 标注我可以是只标注路端的； 测试我可以同时标注
    # 当前时刻的 车端 和 路端本身是对齐的吗; 需要处理坐标的关系

    inf = data['infrastructure_pointcloud_path']
    vehi = data['vehicle_pointcloud_path']
    previous_infs = [data['previous_inf_'+str(i)] for i in range(1, 6)]

    # vis
    vis_sample_veh(data)
    vis_sample_at_veh_view(inf, data, 0)
    for i, previous_inf in enumerate(previous_infs):
        vis_sample_at_veh_view(previous_inf[0], data, i + 1)

    # vis_sample_at_inf_view(inf, data, 0)
    # for i, previous_inf in enumerate(previous_infs):
    #     vis_sample_at_inf_view(previous_inf[0], data, i + 1)



if __name__ == '__main__':
    co_datainfo = read_json(os.path.join(data_dir, 'cooperative/data_info_with_delay.json'))
    for data in co_datainfo:
        label_flow(data)
        pass
      
 
