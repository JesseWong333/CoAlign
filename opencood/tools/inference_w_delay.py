"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
modified by Junjie Wang
"""

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils_group as eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
import math
import json
torch.multiprocessing.set_sharing_strategy('file_system')

def write_json(path_json, data):
    with open(path_json, "w") as f:
        json.dump(data, f)

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt

def scale_boxes(boxes, pc_range=[-100.8, -40, -3.5, 100.8, 40, 1.5], bev_shape=[100, 252] ):
    # boxes: N, 4, 2
    # scale_y = 100 / (40*2)
    # scale_x = 252 / (100.8*2)
    scale_y = bev_shape[0] / (pc_range[4] - pc_range[1])
    scale_x = bev_shape[1] / (pc_range[3] - pc_range[0])

    boxes[:, 0] = (boxes[:, 0] - pc_range[0])* scale_x
    boxes[:, 1] = (boxes[:, 1] - pc_range[1])* scale_y
   
    return boxes

def get_box_speed_group(offset_map, box):
    box_2d = box[:4,:2] # 
    box_2d_scaled = scale_boxes(box_2d) # 4, 2
    mean_xy = box_2d_scaled.mean(axis=0)
    mean_x, mean_y = int(mean_xy[0]), int(mean_xy[1])
    offset = offset_map[mean_y, mean_x]
    speed = math.sqrt(offset[0]**2 + offset[1]**2) / 1.25 / 0.3 # 300ms移动距离 m/s
    return speed

def group_gt_speed(offset_map, gt):
    # gt: N, 8, 3 
    # offset_map: 
    gt_numpy_2d = gt.cpu().numpy()
    offset_map = offset_map.squeeze(0).cpu().numpy()
    number_gt_boxes = gt_numpy_2d.shape[0]
    group = []
    for i in range(number_gt_boxes):
        box = gt_numpy_2d[i]
        speed_group = get_box_speed_group(offset_map, box)
        group.append(speed_group)
    return group

def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    # if 'heter' in hypes:
    #     x_min, x_max = -140.8, 140.8
    #     y_min, y_max = -40, 40
    #     opt.note += f"_{x_max}_{y_max}"
    #     hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
    #     hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
    #     hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

    #     new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
    #                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
    #     hypes['preprocess']['cav_lidar_range'] =  new_cav_range
    #     hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
    #     hypes['postprocess']['gt_range'] = new_cav_range
    #     hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
    #     if 'camera_mask_args' in hypes['model']['args']:
    #         hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

    #     # reload anchor
    #     yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    #     for name, func in yaml_utils_lib.__dict__.items():
    #         if name == hypes["yaml_parser"]:
    #             parser_func = func
    #     hypes = parser_func(hypes)
    
    if hypes['train_params']['batch_size'] != 1:
        hypes['train_params']['batch_size'] = 1
        if hypes['model']['core_method'] == 'voxel_net':
            hypes['model']['args']['N'] = 1
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    # left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    left_hand = True
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')

    max_time_delay = 5
    val_loaders = []
    for time_delay in range(max_time_delay+1):
        hypes.update({"time_delay": time_delay})
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        val_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False,)
        val_loaders.append(val_loader)
    all_box_speed= []
    AP30 = []
    AP50 = []
    AP70 = []
    for time_delay, data_loader in enumerate(val_loaders):
        infer_info = opt.fusion_method + opt.note + 'delay_' + str(time_delay*100) + 'ms'

        # Create the dictionary for evaluation
        # # 
        result_stat = {0.3: {0: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 1: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 2: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 3: {'tp': [], 'fp': [], 'gt': 0, 'score': []}},                
                0.5: {0: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 1: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 2: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 3: {'tp': [], 'fp': [], 'gt': 0, 'score': []}},                
                0.7: {0: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 1: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 2: {'tp': [], 'fp': [], 'gt': 0, 'score': []}, 3: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}}
        # result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
        #                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
        #                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        for i, batch_data in enumerate(data_loader):
            print(f"{infer_info}_{i}")
            # if i > 50:
            #     break
            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'late':
                    infer_result = inference_utils.inference_late_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'early':
                    infer_result = inference_utils.inference_early_fusion(batch_data,
                                                            model,
                                                            opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no_w_uncertainty':
                    infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset,
                                                                    single_gt=True)
                else:
                    raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                            'fusion is supported.')

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']
                
                speed_groups = group_gt_speed(batch_data['ego']['calibrate_data'][1]['offset'], gt_box_tensor)
                # all_box_speed += speed_groups
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        speed_groups,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        speed_groups,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        speed_groups,
                                        0.7)
                if opt.save_npy:
                    npy_save_path = os.path.join(opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if not opt.no_score:
                    infer_result.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                    infer_result.update({"cav_box_np": cav_box_np, \
                                        "lidar_agent_record": lidar_agent_record})

                if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    """
                    If you want 3D visualization, uncomment lines below
                    """
                    # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                    # simple_vis.visualize(infer_result,
                    #                     batch_data['ego'][
                    #                         'origin_lidar'][0],
                    #                     hypes['postprocess']['gt_range'],
                    #                     vis_save_path,
                    #                     method='3d',
                    #                     left_hand=left_hand)
                    
                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    simple_vis.visualize(infer_result,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
            torch.cuda.empty_cache()

        # write_json("./opencood/logs/dair_speed_group.json", all_box_speed)
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, infer_info)
        AP30.append(ap30)
        AP50.append(ap50)
        AP70.append(ap70)

    dump_dict = {'ap30': AP30 ,'ap50': AP50, 'ap70': AP70}
    yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'AP030507_{opt.note}.yaml'))

if __name__ == '__main__':
    main()
