# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def get_speed_group(speed_groups, gt_index):
    speed = speed_groups[gt_index]
    if speed <= 5:
        return 0
    if 5 < speed <= 10:
        return 1
    if speed > 10:
        return 2
    # return 0

def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, speed_groups, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = {0:[], 1:[], 2:[], 3:[]}
    tp = {0:[], 1:[], 2:[], 3:[]}
    scores = {0:[], 1:[], 2:[], 3:[]}
    gt_numbers = {0: 0, 1: 0 , 2: 0, 3: 0}
    speed_groups = speed_groups.copy()
    for k, speed in enumerate(speed_groups):
        speed_group = get_speed_group(speed_groups, k)
        gt_numbers[speed_group] += 1
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))
        speed_groups_copy = speed_groups.copy()
        gt_polygon_list_copy = gt_polygon_list.copy()
        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
            dists = common_utils.compute_dist(det_polygon, gt_polygon_list_copy)
            ious_ = common_utils.compute_iou(det_polygon, gt_polygon_list_copy)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                # 多余的检测, 找最近的GT
                if ious_.shape[0]> 0 and np.max(ious_) > 0:
                    index  = np.argmax(ious_)
                else:
                    index = np.argmax(dists)
                speed_group = get_speed_group(speed_groups_copy, index)
                fp[speed_group].append(1)
                tp[speed_group].append(0)
                scores[speed_group].append(det_score[i])
                continue

            gt_index = np.argmax(ious) #
            speed_group = get_speed_group(speed_groups, gt_index)
            fp[speed_group].append(0)
            tp[speed_group].append(1) 
            scores[speed_group].append(det_score[i])
            speed_groups.pop(gt_index)
            gt_polygon_list.pop(gt_index)
    for i in range(4):
        result_stat[iou_thresh][i]['fp'] += fp[i]
        result_stat[iou_thresh][i]['tp'] += tp[i]
        result_stat[iou_thresh][i]['gt'] += gt_numbers[i]
        result_stat[iou_thresh][i]['score'] += scores[i]


def calculate_ap(result_stat, iou, speed_group):
    """
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]
    iou_5 = result_stat[iou][speed_group]

    fp = np.array(iou_5['fp'])
    tp = np.array(iou_5['tp'])
    score = np.array(iou_5['score'])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, infer_info=None):
    dump_dict = {0: {}, 1: {}, 2: {}, 3: {}}

    for i in range(4):
        ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, i)
        ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, i)
        ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, i)
        dump_dict[i].update(
            {'ap_30': ap_30,
            'ap_50': ap_50,
            'ap_70': ap_70,
            }
        )

    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    for i in range(4):
        print('speed_group ' + str(i))
        print('The Average Precision at IOU 0.3 is %.3f, '
            'The Average Precision at IOU 0.5 is %.3f, '
            'The Average Precision at IOU 0.7 is %.3f' % (dump_dict[i]['ap_30'], dump_dict[i]['ap_50'], dump_dict[i]['ap_70']))

    return ap_30, ap_50, ap_70