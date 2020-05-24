# Adapted from Detectron.pytorch/lib/modeling/model_builder.py
# for this project by Ji Zhang, 2019

from functools import wraps
import importlib
import logging
import numpy as np
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling_rel.fast_rcnn_heads as fast_rcnn_heads
# import modeling_rel.relpn_heads as relpn_heads
import modeling_rel.reldn_heads as reldn_heads
import modeling_rel.rel_pyramid_module as rel_pyramid_module
import utils_rel.boxes_rel as box_utils_rel
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils_rel.net_rel as net_utils_rel
from utils.timer import Timer
import utils.resnet_weights_helper as resnet_utils
import utils.fpn as fpn_utils

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        # these two keywords means we need to use the functions from the modeling_rel directory
        if func_name.find('VGG') >= 0 or func_name.find('roi_2mlp_head') >= 0 or func_name.find('rel') >= 0:
            dir_name = 'modeling_rel.'
        else:
            dir_name = 'modeling.'
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = dir_name + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = get_func(cfg.RPN.HEAD)(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
            self.Box_Head.dim_out)
        
        # self.Prd_RCNN = copy.deepcopy(self)
        # del self.Prd_RCNN.RPN
        # del self.Prd_RCNN.Box_Outs
        # del self.Prd_RCNN.Box_Head

        # self.RelPN = relpn_heads.generic_relpn_outputs()
        
            
         # rel pyramid connection
        if cfg.MODEL.USE_REL_PYRAMID:
            assert cfg.FPN.FPN_ON
            self.RelPyramid = rel_pyramid_module.rel_pyramid_module(self.num_roi_levels)
        
        

        self._init_modules()
        
        # initialize S/O branches AFTER init_weigths so that weights can be automatically copied

    def _init_modules(self):
        # VGG16 imagenet pretrained model is initialized in VGG16.py
        if cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS != '':
            logger.info("Loading pretrained weights from %s", cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
            resnet_utils.load_pretrained_imagenet_weights(self)
            for p in self.Conv_Body.parameters():
                p.requires_grad = False
                
        if cfg.RESNETS.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VRD_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VRD_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VG_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VG_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS)
        if cfg.VGG16.OI_REL_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.OI_REL_PRETRAINED_WEIGHTS)
        
        # if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '' or \
        #     cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '' or \
        #     cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
        #     if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '':
        #         logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS)
        #         checkpoint = torch.load(cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        #     if cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '':
        #         logger.info("loading prd pretrained weights from %s", cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS)
        #         checkpoint = torch.load(cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        #     if cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '':
        #         logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS)
        #         checkpoint = torch.load(cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        #     if cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '':
        #         logger.info("loading prd pretrained weights from %s", cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS)
        #         checkpoint = torch.load(cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        #     if cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
        #         logger.info("loading prd pretrained weights from %s", cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS)
        #         checkpoint = torch.load(cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        #     if cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
        #         logger.info("loading prd pretrained weights from %s", cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS)
        #         checkpoint = torch.load(cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        #     # not using the last softmax layers
        #
        #     del checkpoint['Box_Head.fc1.weight']
        #     del checkpoint['Box_Head.fc1.bias']
        #     del checkpoint['Box_Head.fc2.weight']
        #     del checkpoint['Box_Head.fc2.bias']
            # net_utils_rel.load_ckpt_rel(self.Prd_RCNN, checkpoint)
    
    def load_detector_weights(self, weight_name):
        logger.info("loading pretrained weights from %s", weight_name)
        checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
        net_utils_rel.load_ckpt_rel(self, checkpoint['model'])
        

    def forward(self, data, im_info, do_vis=False, dataset_name=None, roidb=None, use_gt_labels=False, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, do_vis, dataset_name, roidb, use_gt_labels, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, do_vis, dataset_name, roidb, use_gt_labels, **rpn_kwargs)

    def _forward(self, data, im_info, do_vis=False, dataset_name=None, roidb=None, use_gt_labels=False, **rpn_kwargs):
        im_data = data
        if self.training:
            # if not isinstance(roidb[0], np.array):
            #     roidb = roidb[0]
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb)) # only support one gpu
        if dataset_name is not None:
            dataset_name = blob_utils.deserialize(dataset_name)
        else:
            dataset_name = cfg.TRAIN.DATASETS[0] if self.training else cfg.TEST.DATASETS[0]  # assuming only one dataset per run

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)
        # if not cfg.MODEL.USE_REL_PYRAMID:
        #     blob_conv_prd = self.Prd_RCNN.Conv_Body(im_data)

        if self.training:
            gt_rois = np.empty((0, 5), dtype=np.float32)
            gt_classes = np.empty((0), dtype=np.int64)
            for i, r in enumerate(roidb):
                rois_i = r['boxes'] * im_info[i, 2]
                rois_i = np.hstack((i * blob_utils.ones((rois_i.shape[0], 1)), rois_i))
                gt_rois = np.append(gt_rois, rois_i, axis=0)
                gt_classes = np.append(gt_classes, r['gt_classes'], axis=0)

        if self.training or roidb is None:
            rpn_ret = self.RPN(blob_conv, im_info, roidb)




        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
            # if not cfg.MODEL.USE_REL_PYRAMID:
            #     blob_conv_prd = blob_conv_prd[-self.num_roi_levels:]
            # else:
            #     blob_conv_prd = self.RelPyramid(blob_conv)

        if self.training or roidb is None:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
            else:
                box_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
            cls_score, bbox_pred = self.Box_Outs(box_feat)

        
        # now go through the predicate branch
        use_relu = False if cfg.MODEL.NO_FC7_RELU else True
        if self.training:
            score_thresh = cfg.TEST.SCORE_THRESH
            cls_score = F.softmax(cls_score, -1)
            while score_thresh >= -1e-06:  # a negative value very close to 0.0
                det_rois, det_labels, det_scores, det_dists, det_boxes_all = \
                    self.prepare_det_rois(rpn_ret['rois'], cls_score, bbox_pred, im_info, score_thresh)
                real_area = (det_rois[:, 3] - det_rois[:, 1]) * (det_rois[:, 4] - det_rois[:, 2])
                non_zero_area_inds = np.where(real_area > 0)[0]
                det_rois = det_rois[non_zero_area_inds]
                det_labels = det_labels[non_zero_area_inds]
                det_scores = det_scores[non_zero_area_inds]
                det_dists = det_dists[non_zero_area_inds]
                det_boxes_all = det_boxes_all[non_zero_area_inds]
                # rel_ret = self.RelPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
                valid_len = len(det_rois)
                if valid_len > 0:
                    break
                logger.info('Got {} det_rois when score_thresh={}, changing to {}'.format(
                    valid_len, score_thresh, score_thresh - 0.01))
                score_thresh -= 0.01
            det_labels_gt = []
            ious = box_utils.bbox_overlaps(det_rois[:, 1:], gt_rois[:, 1:]) * \
                                          (det_rois[:, 0][:,None] == gt_rois[:, 0][None, :])
            det_labels_gt = gt_classes[ious.argmax(-1)]
            det_labels_gt[ious.max(-1) < cfg.TRAIN.FG_THRESH] = 0

        else:
            if roidb is not None:
                # raise FError('not support this mode!')
                # assert len(roidb) == 1
                im_scale = im_info.data.numpy()[:, 2][0]
                im_w = im_info.data.numpy()[:, 1][0]
                im_h = im_info.data.numpy()[:, 0][0]
                
                fpn_ret = {'gt_rois': gt_rois}
                if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                    lvl_min = cfg.FPN.ROI_MIN_LEVEL
                    lvl_max = cfg.FPN.ROI_MAX_LEVEL
                    rois_blob_names = ['gt_rois']
                    for rois_blob_name in rois_blob_names:
                        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                        target_lvls = fpn_utils.map_rois_to_fpn_levels(
                            fpn_ret[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                        fpn_utils.add_multilevel_roi_blobs(
                            fpn_ret, rois_blob_name, fpn_ret[rois_blob_name], target_lvls,
                            lvl_min, lvl_max)
                det_feats = self.Box_Head(blob_conv, fpn_ret, rois_name='det_rois', use_relu=True)
                det_dists, _ = self.Box_Outs(det_feats)
                det_boxes_all = None
                if use_gt_labels:
                    det_labels_gt = gt_classes
                    det_labels = gt_classes
            else:

                score_thresh = cfg.TEST.SCORE_THRESH
                while score_thresh >= -1e-06:  # a negative value very close to 0.0
                    det_rois, det_labels, det_scores, det_dists, det_boxes_all = \
                        self.prepare_det_rois(rpn_ret['rois'], cls_score, bbox_pred, im_info, score_thresh)
                    real_area = (det_rois[:, 3] - det_rois[:, 1]) * (det_rois[:, 4] - det_rois[:, 2])
                    non_zero_area_inds = np.where(real_area > 0)[0]
                    det_rois = det_rois[non_zero_area_inds]
                    det_labels = det_labels[non_zero_area_inds]
                    det_scores = det_scores[non_zero_area_inds]
                    det_dists = det_dists[non_zero_area_inds]
                    det_boxes_all = det_boxes_all[non_zero_area_inds]
                    # rel_ret = self.RelPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
                    valid_len = len(det_rois)
                    if valid_len > 0:
                        break
                    logger.info('Got {} det_rois when score_thresh={}, changing to {}'.format(
                        valid_len, score_thresh, score_thresh - 0.01))
                    score_thresh -= 0.01 


        return_dict['det_rois'] = det_rois
        num_rois = det_rois.shape[0]
        if not isinstance(det_dists, torch.Tensor):
            assert det_dists.shape[0] == num_rois
            det_dists = torch.from_numpy(det_dists).float().cuda(device_id)
        
        return_dict['det_dists'] = det_dists
        return_dict['det_scores'] = det_scores
        return_dict['blob_conv'] = blob_conv
        return_dict['det_boxes_all'] = det_boxes_all
        assert det_boxes_all.shape[0] == num_rois
        return_dict['det_labels'] = det_labels
        # return_dict['blob_conv_prd'] = blob_conv_prd

        if self.training or use_gt_labels:
            return_dict['det_labels_gt'] = det_labels_gt

        return return_dict
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds
    
    def prepare_det_rois(self, rois, cls_scores, bbox_pred, im_info, score_thresh=cfg.TEST.SCORE_THRESH):
        im_info = im_info.data.cpu().numpy()
        # NOTE: 'rois' is numpy array while
        # 'cls_scores' and 'bbox_pred' are pytorch tensors
        scores = cls_scores.data.cpu().numpy().squeeze()
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy().squeeze()
        
        assert rois.shape[0] == scores.shape[0] == box_deltas.shape[0]
        
        det_rois = np.empty((0, 5), dtype=np.float32)
        det_labels = np.empty((0), dtype=np.int64)
        det_scores = np.empty((0), dtype=np.float32)
        det_dists = np.empty((0, scores.shape[-1]), dtype=np.float32)
        det_boxes_all = np.empty((0, scores.shape[-1], 4), dtype=np.float32)
        for im_i in range(im_info.shape[0]):
            # get all boxes that belong to this image
            inds = np.where(abs(rois[:, 0] - im_i) < 1e-06)[0]
            # unscale back to raw image space
            im_boxes = rois[inds, 1:5] / im_info[im_i, 2]
            im_scores = scores[inds]
            # In case there is 1 proposal
            im_scores = im_scores.reshape([-1, im_scores.shape[-1]])
            # In case there is 1 proposal
            im_box_deltas = box_deltas[inds]
            im_box_deltas = im_box_deltas.reshape([-1, im_box_deltas.shape[-1]])

            im_dists, im_boxes_pre = self.get_det_boxes(im_boxes, im_scores, im_box_deltas, im_info[im_i][:2] / im_info[im_i][2])
            im_scores, keep_inds, im_labels = self.box_results_with_nms_and_limit(im_dists, im_boxes_pre, score_thresh)
            
            batch_inds = im_i * np.ones(
                (keep_inds.shape[0], 1), dtype=np.float32)
            
            im_det_rois = np.hstack((batch_inds, im_boxes[keep_inds] * im_info[im_i, 2]))
            det_rois = np.append(det_rois, im_det_rois, axis=0)
            det_labels = np.append(det_labels, im_labels, axis=0)
            det_scores = np.append(det_scores, im_scores, axis=0)
            det_dists = np.append(det_dists, im_dists[keep_inds], axis=0)
            det_boxes_all = np.append(det_boxes_all, im_boxes_pre.reshape([im_box_deltas.shape[0], -1, 4])[keep_inds], axis=0)
        
        return det_rois, det_labels, det_scores, det_dists, det_boxes_all

    def get_det_boxes(self, boxes, scores, box_deltas, h_and_w):

        if cfg.TEST.BBOX_REG:
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                # Remove predictions for bg class (compat with MSRA code)
                box_deltas = box_deltas[:, -4:]
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # (legacy) Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                             + cfg.TRAIN.BBOX_NORMALIZE_MEANS
            pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
            pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, h_and_w)
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
            # Map scores and predictions back to the original set of boxes
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]
            
        return scores, pred_boxes
    
    def box_results_with_nms_and_limit(self, scores, boxes, score_thresh=cfg.TEST.SCORE_THRESH):
        num_classes = cfg.MODEL.NUM_CLASSES
        cls_boxes = [[] for _ in range(num_classes)]
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        nms_mask = np.zeros_like(scores)
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > score_thresh)[0]
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
            if cfg.TEST.SOFT_NMS.ENABLED:
                nms_dets, keep = box_utils.soft_nms(
                    dets_j,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.TEST.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                nms_dets = dets_j[keep, :]
            nms_mask[:, j][keep] = 1.0
            # add labels
            
            # Refine the post-NMS boxes using bounding-box voting
            

        dists_all = nms_mask * scores

        scores_pre, labels_pre = dists_all.max(-1), dists_all.argmax(-1)
        inds_all = np.where(scores_pre > 0)[0]
        labels_all = labels_pre[inds_all]
        scores_all = scores_pre[inds_all]

        idx = np.argsort(-scores_all)
        if cfg.TEST.DETECTIONS_PER_IM < idx.shape[0]:
            idx = idx[:cfg.TEST.DETECTIONS_PER_IM]
        
        scores = scores_all[idx]
        labels = labels_all[idx]

        return scores, idx, labels

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)
            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
