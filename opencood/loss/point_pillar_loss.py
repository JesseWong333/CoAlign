# -*- coding: utf-8 -*-
# Author: Yifan Lu, modified by Junjie Wang
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
from icecream import ic

class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']

        self.cls = args['cls']
        self.reg = args['reg']

        if 'dir' in args:
            self.dir = args['dir']
        else:
            self.dir = None

        if 'iou' in args:
            self.iou = args['iou']
        else:
            self.iou = None
        
        if 'calibrate' in args:
            self.calibrate = args['calibrate']
        else:
            self.calibrate = False
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0
        # cared = torch.logical_or(positives, negatives)
        # cls_labls = cls_labls * cared.type_as(cls_labls)
        # num_normalizer = cared.sum(1, keepdim=True)
        pos_normalizer = positives.sum(1, keepdim=True).float()

        # rename variable 
        if f'psm{suffix}' in output_dict:
            output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
        if f'rm{suffix}' in output_dict:
            output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']
        if f'dm{suffix}' in output_dict:
            output_dict[f'dir_preds{suffix}'] = output_dict[f'dm{suffix}']

        total_loss = 0

        # cls loss
        cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size


        ######## direction ##########
        if self.dir:
            dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, 7))
            dir_logits = output_dict[f"dir_preds{suffix}"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # [N, H*W*#anchor, 2]

            dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
            dir_loss = dir_loss.flatten() * reg_weights.flatten() 
            dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size
            total_loss += dir_loss
            self.loss_dict.update({'dir_loss': dir_loss.item()})

        ######### calibrate loss #########
        if False:
            # 1） 需要mask的地方， offset不预测
            coords_pred = output_dict["calibrate"] # B, 3, H, W 这个地方需要插值
            target_offset = target_dict['offset'].permute(0, 3, 1, 2) #  # target 是 B, H, W, 2 -> B, 2, H, W
            target_offset_mask = target_dict['offset_mask'].unsqueeze(1).float() # B, 1, H, W
            all_offset_loss = 0
            all_offset_mask_loss = 0
            for i in range(len(coords_pred)):
                pred = coords_pred[i]
                # scale target
                b, _, h, w = pred.size()
                target_offset_scaled = target_offset
                target_offset_mask_scaled = target_offset_mask
                if target_offset.shape[2] !=  h or target_offset.shape[3] != w:
                    target_offset_scaled = F.interpolate(target_offset, (h, w), mode='area')         
                    target_offset_mask_scaled = F.interpolate(target_offset_mask, (h, w), mode='area') # 只可以对float插值
                    # target_offset_mask_scaled[target_offset_mask_scaled >= 0.5] = 1.0 # BEC_loss输入float
                    # target_offset_mask_scaled[target_offset_mask_scaled < 0.5] = 0.0
                    # target_offset_mask_scaled = target_offset_mask_scaled.long()
                # select positive and negtive
                target_offset_scaled = target_offset_scaled.permute(0, 2, 3, 1).view(b*h*w, 2)  # b*h*w, 2
                target_offset_mask_scaled = target_offset_mask_scaled.squeeze(1).flatten()

                pos_boolean = (target_offset_scaled[:, 0] !=0.0) | (target_offset_scaled[:, 0] !=0.0)
                neg_boolean = (target_offset_scaled[:, 0] == 0.0) & (target_offset_scaled[:, 0] == 0.0)

                pos_mask_boolean = (target_offset_mask_scaled == 1)

                pos_inds = torch.nonzero( pos_boolean & ( ~pos_mask_boolean)).squeeze(-1) # 不管是正还是负都要去掉mask为true的部分
                neg_inds = torch.nonzero( neg_boolean & ( ~pos_mask_boolean)).squeeze(-1)

                pos_inds_mask = torch.nonzero(pos_mask_boolean).squeeze(-1)
                neg_inds_mask = torch.nonzero(target_offset_mask_scaled == 0).squeeze(-1)

                # random sample
                num_pos = pos_inds.numel()  # 可能为0
                if num_pos > 0:
                    num_neg = num_pos * 5 # 比例可设置
                else:
                    num_neg = neg_inds.numel() // 100
                neg_inds_select = self.random_sample(neg_inds, num_neg)

                num_pos_mask = pos_inds_mask.numel()
                if num_pos_mask > 0:
                    num_neg_mask = num_pos_mask * 5
                else:
                    num_neg_mask = neg_inds_mask.numel() // 100
                neg_inds_mask_select = self.random_sample(neg_inds_mask, num_neg_mask)

                all_offset_inds = torch.cat([pos_inds, neg_inds_select]).unique()
                all_offset_mask_inds = torch.cat([pos_inds_mask, neg_inds_mask_select, pos_inds]).unique() # 所有pos_inds都是roi

                pred_offset = pred[:, :2, :, :].permute(0, 2, 3, 1).reshape(b*h*w, 2)
                pred_mask = pred[:, 2, :, :].flatten()
                loss_offset = nn.MSELoss()(pred_offset[all_offset_inds, :], target_offset_scaled[all_offset_inds, :])
                loss_mask = F.binary_cross_entropy(torch.sigmoid(pred_mask[all_offset_mask_inds]), target_offset_mask_scaled[all_offset_mask_inds])

                all_offset_loss  += loss_offset
                all_offset_mask_loss += loss_mask

            total_loss = total_loss + all_offset_loss + all_offset_mask_loss
            self.loss_dict.update({'offset_loss': all_offset_loss.item(),
                               'offset_mask_loss': all_offset_mask_loss.item()})


        ######## IoU ###########
        if self.iou:
            iou_preds = output_dict["iou_preds{suffix}"].permute(0, 2, 3, 1).contiguous()
            pos_pred_mask = reg_weights.squeeze(dim=-1) > 0 # (4, 70400)
            iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
            boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().detach(),
                                                            output_dict['anchor_box'])[pos_pred_mask]
            boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(target_dict['targets'],
                                                            output_dict['anchor_box'])[pos_pred_mask]
            iou_weights = reg_weights[pos_pred_mask].view(-1)
            iou_pos_targets = aligned_boxes_iou3d_gpu(boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]], # hwl -> dx dy dz
                                                    boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
            iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
            iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou['sigma'])

            iou_loss = iou_loss.sum() * self.iou['weight'] / batch_size
            total_loss += iou_loss
            self.loss_dict.update({'iou_loss': iou_loss.item()})

        total_loss += reg_loss + cls_loss

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'reg_loss': reg_loss.item(),
                               'cls_loss': cls_loss.item()})

        return total_loss

    @staticmethod
    def random_sample(inds, expect_num):
        random_indices = torch.randperm(inds.shape[0])[:expect_num]  # 使用torch.randperm获取随机索引
        random_elements = inds[random_indices]
        return random_elements

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        num_bins = self.dir['args']['num_bins']
        dir_offset = self.dir['args']['dir_offset']
        anchor_yaw = np.deg2rad(np.array(self.dir['args']['anchor_yaw']))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets



    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        offset_loss = self.loss_dict.get('offset_loss', 0)
        offset_mask_loss = self.loss_dict.get('offset_mask_loss',0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Offset Loss: %.4f || Offsetmask Loss: %.4f " % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, offset_loss, offset_mask_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device) 
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                    
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss

def sequence_loss(flow_preds, flow_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions 
        flow_preds: [B, S, N, 2] 的list
    """
    B, S, N, D = flow_gt.shape
    assert(D==2)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = flow_preds[i]#[:,:,0:1]
        i_loss = (flow_pred - flow_gt).abs() # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=3) # B, S, N
        flow_loss += i_weight * i_loss
    flow_loss = flow_loss/n_predictions
    return flow_loss
