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
from collections import defaultdict

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling_rel.fast_rcnn_heads as fast_rcnn_heads
import modeling_rel.relpn_heads as relpn_heads
import modeling_rel.reldn_heads as reldn_heads
import modeling_rel.rel_pyramid_module as rel_pyramid_module
from modeling_rel.refine_obj_feats import Merge_OBJ_Feats, Message_Passing4OBJ
from modeling_rel.detector_builder import Generalized_RCNN
import utils_rel.boxes_rel as box_utils_rel
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils_rel.net_rel as net_utils_rel
from utils.timer import Timer
import utils.resnet_weights_helper as resnet_utils
import utils.fpn as fpn_utils
from modeling_rel.sparse_targets_rel import FrequencyBias, FrequencyBias_Fix
from datasets_rel.pytorch_misc import intersect_2d
from math import pi

logger = logging.getLogger(__name__)


def get_ort_embeds(k, dims):
    ind = torch.arange(1, k + 1).float().unsqueeze(1).repeat(1, dims)
    lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k, 1)
    t = ind * lin_space
    return torch.sin(t) + torch.cos(t)




def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        # these two keywords means we need to use the functions from the modeling_rel directory
        if func_name.find('VGG') >= 0 or func_name.find('roi_2mlp_head') >= 0:
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


class Rel_Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.detector = Generalized_RCNN()
        self.obj_dim = self.detector.Box_Head.dim_out

        self.Box_Head_sg = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.detector.RPN.dim_out, self.roi_feature_transform, self.detector.Conv_Body.spatial_scale)

        self.Box_Head_prd = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD_PRD)(
            self.detector.RPN.dim_out, self.roi_feature_transform, self.detector.Conv_Body.spatial_scale)
        self.union_mask = reldn_heads.union_mask(self.detector.RPN.dim_out)

        self.ori_embed = get_ort_embeds(cfg.MODEL.NUM_CLASSES, 200)

        # self.RelDN = reldn_heads.relfusion(self.obj_dim, 512, 200)
        # self.reduce4edge = nn.Linear(self.obj_dim, 512)
        self.merge_obj_feats = Merge_OBJ_Feats(self.obj_dim, 200, 512)
        self.obj_mps1 = Message_Passing4OBJ(512)
        self.obj_mps2 = Message_Passing4OBJ(512)
        self.ObjClassifier = nn.Linear(512, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias_Fix(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias_Fix(cfg.TEST.DATASETS[0])

        if cfg.MODEL.USE_BG:
            self.num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1
        else:
            self.num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES

        self.get_phr_feats = nn.Linear(self.obj_dim, 512)

        self.merge_high = nn.Linear(512, self.obj_dim)
        self.merge_high.weight = torch.nn.init.xavier_normal_(self.merge_high.weight, gain=1.0)
        self.merge_low = nn.Linear(self.obj_dim + 200 + 5, self.obj_dim)
        self.merge_low.weight = torch.nn.init.xavier_normal_(self.merge_low.weight, gain=1.0)
        self.sbj_map = nn.Linear(self.obj_dim, self.obj_dim)
        self.sbj_map.weight = torch.nn.init.xavier_normal_(self.sbj_map.weight, gain=1.0)
        self.obj_map = nn.Linear(self.obj_dim, self.obj_dim)
        self.obj_map.weight = torch.nn.init.xavier_normal_(self.obj_map.weight, gain=1.0)
        self.rel_compress = nn.Linear(self.obj_dim, self.num_prd_classes)
        self.rel_compress.weight = torch.nn.init.xavier_normal_(self.rel_compress.weight, gain=1.0)

        self.__init_modules()

    def __init_modules(self):

        logger.info("Freeze detector weights.")
        for p in self.detector.parameters():
            p.requires_grad = False

        if cfg.RESNETS.VRD_PRETRAINED_WEIGHTS != '':
            ckpt = torch.load(cfg.RESNETS.VRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        if cfg.VGG16.VRD_PRETRAINED_WEIGHTS != '':
            ckpt = torch.load(cfg.VGG16.VRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)

        if cfg.RESNETS.VG_PRETRAINED_WEIGHTS != '':
            ckpt = torch.load(cfg.RESNETS.VG_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        if cfg.VGG16.VG_PRETRAINED_WEIGHTS != '':
            ckpt = torch.load(cfg.VGG16.VG_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)

        if cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS != '':
            ckpt = torch.load(cfg.RESNETS.OI_REL_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        if cfg.VGG16.OI_REL_PRETRAINED_WEIGHTS != '':
            ckpt = torch.load(cfg.VGG16.OI_REL_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)

        self.Box_Head_sg.heads[0].weight.data.copy_(ckpt['model']['Box_Head.heads.0.weight'])
        self.Box_Head_sg.heads[0].bias.data.copy_(ckpt['model']['Box_Head.heads.0.bias'])
        self.Box_Head_sg.heads[3].weight.data.copy_(ckpt['model']['Box_Head.heads.3.weight'])
        self.Box_Head_sg.heads[3].bias.data.copy_(ckpt['model']['Box_Head.heads.3.bias'])
        self.Box_Head_prd.heads[0].weight.data.copy_(ckpt['model']['Box_Head.heads.0.weight'])
        self.Box_Head_prd.heads[0].bias.data.copy_(ckpt['model']['Box_Head.heads.0.bias'])
        self.Box_Head_prd.heads[3].weight.data.copy_(ckpt['model']['Box_Head.heads.3.weight'])
        self.Box_Head_prd.heads[3].bias.data.copy_(ckpt['model']['Box_Head.heads.3.bias'])

        del ckpt

        if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '' or \
                cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '' or \
                cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
            if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS,
                                        map_location=lambda storage, loc: storage)
            if cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS,
                                        map_location=lambda storage, loc: storage)
            if cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.OI_REL_PRD_PRETRAINED_WEIGHTS,
                                        map_location=lambda storage, loc: storage)
            if cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.OI_REL_PRD_PRETRAINED_WEIGHTS,
                                        map_location=lambda storage, loc: storage)

            self.Box_Head_prd.heads[0].weight.data.copy_(checkpoint['model']['Box_Head.heads.0.weight'])
            self.Box_Head_prd.heads[0].bias.data.copy_(checkpoint['model']['Box_Head.heads.0.bias'])
            self.Box_Head_prd.heads[3].weight.data.copy_(checkpoint['model']['Box_Head.heads.3.weight'])
            self.Box_Head_prd.heads[3].bias.data.copy_(checkpoint['model']['Box_Head.heads.3.bias'])

        # if not cfg.TRAIN.FREEZE_PRD_CONV_BODY:  # TODO: True in default
        #     for p in self.detector.Prd_RCNN.parameters():
        #         p.requires_grad = True

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

    def get_rel_inds(self, det_rois, det_labels, roidb, im_info):

        num_img = int(det_rois[:, 0].max()) + 1

        im_inds = det_rois[:, 0].astype(np.int64)

        # TODO: Not support sgdet mode training yet.

        if self.training:

            
            return relpn_heads.rel_assignments(im_inds, det_rois, det_labels, roidb, im_info, \
                                            num_sample_per_gt=1, filter_non_overlap=True)

        else:
            if cfg.TRAIN.GT_BOXES:
                fg_rels = []
                is_cand = (im_inds[:, None] == im_inds[None])
                is_cand[np.arange(im_inds.shape[0]), np.arange(im_inds.shape[0])] = False
                for i in range(num_img):
                    gt_boxes_i = roidb[i]['boxes']
                    sbj_gt_boxes_i = roidb[i]['sbj_gt_boxes']
                    obj_gt_boxes_i = roidb[i]['obj_gt_boxes']

                    sbj_gt_inds_i = box_utils.bbox_overlaps(sbj_gt_boxes_i, gt_boxes_i).argmax(-1)
                    obj_gt_inds_i = box_utils.bbox_overlaps(obj_gt_boxes_i, gt_boxes_i).argmax(-1)
                    im_id_i = np.ones_like(sbj_gt_inds_i) * i
                    gt_rels_i = np.stack((im_id_i, sbj_gt_inds_i, obj_gt_inds_i), -1)
                    fg_rels.append(gt_rels_i)

                rel_inds = np.concatenate(fg_rels, 0)
                
            else:

                is_cand = (im_inds[:, None] == im_inds[None])
                is_cand[np.arange(im_inds.shape[0]), np.arange(im_inds.shape[0])] = False

                is_cand = (box_utils.bbox_overlaps(det_rois[:, 1:], det_rois[:, 1:]) > 0) & is_cand
                    # raise FError('not support this mode!')

                sbj_ind, obj_ind = np.where(is_cand)
                if len(sbj_ind) == 0:
                    sbj_ind, obj_ind = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
                rel_inds = np.stack((det_rois[sbj_ind, 0].astype(sbj_ind.dtype), sbj_ind, obj_ind), -1)

            return rel_inds, None

    def union_pairs(self, im_inds):
        rel_cands = im_inds[:, None] == im_inds[None]
        rel_cands[np.arange(im_inds.shape[0]), np.arange(im_inds.shape[0])] = False
        empty_ind = np.where(np.logical_not(rel_cands.any(-1)))[0]
        if empty_ind.size > 0:
            rel_cands[empty_ind, empty_ind] = True
        sbj_ind, obj_ind = np.where(rel_cands)
        # if len(sbj_ind) == 0:
        #         sbj_ind, obj_ind = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
        return np.stack((im_inds[sbj_ind].astype(sbj_ind.dtype), sbj_ind, obj_ind), -1)
    
    def get_nms_preds(self, cls_scores, det_boxes_all):
        # probs = F.softmax(cls_scores, -1).data.cpu().numpy()
        probs = cls_scores
        nms_mask = np.zeros_like(probs)
        for c in range(1, probs.shape[-1]):
            s_c = probs[:, c]
            boxes_c = det_boxes_all[:, c]
            dets_c = np.hstack((boxes_c, s_c[:, np.newaxis])).astype(np.float32, copy=False)
            keep = box_utils.nms(dets_c, cfg.TEST.NMS)
            nms_mask[:, c][keep] = 1
        obj_preds = (nms_mask * probs)[:, 1:].argmax(-1) + 1
        return obj_preds

    def get_obj_pos(self, det_rois, im_info):
        im_w = im_info.data.numpy()[:, 1]
        im_h = im_info.data.numpy()[:, 0]
        im_inds = det_rois[:, 0].astype(np.int64)
        obj_pos = np.stack((det_rois[:, 1] / im_w[im_inds], det_rois[:, 2] / im_h[im_inds],
                            det_rois[:, 3] / im_w[im_inds], det_rois[:, 4] / im_h[im_inds],
                            ((det_rois[:, 3] - det_rois[:, 1] + 1.0) * (det_rois[:, 4] - det_rois[:, 2] + 1.0)) / (
                                        im_h[im_inds] * im_w[im_inds])), -1)
        return obj_pos

    def obj_feature_map(self, blob_conv, det_rois, use_relu=True):
        fpn_ret = {'rois': det_rois}
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
            lvl_min = cfg.FPN.ROI_MIN_LEVEL
            lvl_max = cfg.FPN.ROI_MAX_LEVEL
            # when use min_rel_area, the same sbj/obj area could be mapped to different feature levels
            # when they are associated with different relationships
            # Thus we cannot get det_rois features then gather sbj/obj features
            # The only way is gather sbj/obj per relationship, thus need to return sbj_rois/obj_rois
            rois_blob_names = ['rois']
            for rois_blob_name in rois_blob_names:
                # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                target_lvls = fpn_utils.map_rois_to_fpn_levels(
                    fpn_ret[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                fpn_utils.add_multilevel_roi_blobs(
                    fpn_ret, rois_blob_name, fpn_ret[rois_blob_name], target_lvls,
                    lvl_min, lvl_max)

        return self.Box_Head_sg(blob_conv, fpn_ret, rois_name='rois', use_relu=use_relu)

    def visual_rep(self, blob_conv_prd, rois, pair_inds, device_id, use_relu=False):
        assert pair_inds.shape[1] == 2
        rel_rois = box_utils_rel.rois_union(rois[pair_inds[:, 0]], rois[pair_inds[:, 1]])
        rel_ret = {'sbj_rois': rois[pair_inds[:, 0]], 'obj_rois': rois[pair_inds[:, 1]],
                   'rel_rois': rel_rois}
        union_mask = self.union_mask(rel_ret, device_id)
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
            lvl_min = cfg.FPN.ROI_MIN_LEVEL
            lvl_max = cfg.FPN.ROI_MAX_LEVEL
            # when use min_rel_area, the same sbj/obj area could be mapped to different feature levels
            # when they are associated with different relationships
            # Thus we cannot get det_rois features then gather sbj/obj features
            # The only way is gather sbj/obj per relationship, thus need to return sbj_rois/obj_rois
            rois_blob_names = ['rel_rois']
            for rois_blob_name in rois_blob_names:
                # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                target_lvls = fpn_utils.map_rois_to_fpn_levels(
                    rel_ret[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                fpn_utils.add_multilevel_roi_blobs(
                    rel_ret, rois_blob_name, rel_ret[rois_blob_name], target_lvls,
                    lvl_min, lvl_max)
        return self.Box_Head_prd(blob_conv_prd, rel_ret, union_mask, rois_name='rel_rois', use_relu=use_relu)

    def forward(self, data, im_info, do_vis=False, dataset_name=None, roidb=None, use_gt_labels=False, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, do_vis, dataset_name, roidb, use_gt_labels, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, do_vis, dataset_name, roidb, use_gt_labels, **rpn_kwargs)

    def _forward(self, data, im_info, do_vis=False, dataset_name=None, roidb=None, use_gt_labels=False, **rpn_kwargs):

        # assuming only one dataset per run

        # if not isinstance(data, torch.Tensor):
        #     data = data[0]
        # if not isinstance(im_info, torch.Tensor):
        #     im_info = im_info[0]
        device_id = data.get_device()
        return_dicts = self.detector(data, im_info, do_vis, dataset_name, roidb, use_gt_labels, **rpn_kwargs)
        rel_return_dicts = {}

        if self.training:
            # if not isinstance(roidb[0], np.array):
            #     roidb = roidb[0]
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))
        if dataset_name is not None:
            dataset_name = blob_utils.deserialize(dataset_name)
        else:
            dataset_name = cfg.TRAIN.DATASETS[0] if self.training else cfg.TEST.DATASETS[0]

        # im_scale = im_info.data.numpy()[:, 2]
        # im_w = im_info.data.numpy()[:, 1]
        # im_h = im_info.data.numpy()[:, 0]

        det_rois, det_dists = return_dicts['det_rois'], return_dicts['det_dists']
        blob_conv = return_dicts['blob_conv']
        
        # blob_conv_prd = return_dicts['blob_conv_prd']

        if self.training or use_gt_labels:
            det_labels = return_dicts['det_labels_gt']
            if not isinstance(det_labels, torch.Tensor):
                det_labels = torch.from_numpy(det_labels).long().cuda(device_id)
        else:
            det_labels = None

        im_inds = det_rois[:, 0].astype(np.int64)
        obj_fmap = self.obj_feature_map(blob_conv, det_rois, use_relu=True)

        obj_feats = self.merge_obj_feats(obj_fmap, det_rois, det_dists, im_info)

        rel_inds, rel_labels = self.get_rel_inds(det_rois, return_dicts['det_labels_gt'] if self.training else None, \
                                                 roidb, im_info.data.cpu().numpy())
        pair_inds = self.union_pairs(im_inds)
        if (rel_labels is not None) and (not isinstance(rel_labels, torch.Tensor)):
            rel_labels = torch.from_numpy(rel_labels).long().cuda(device_id)

        vr_indices = intersect_2d(rel_inds[:, 1:], pair_inds[:, 1:]).argmax(-1)
        # rel_rois = box_utils_rel.rois_union(det_rois[rel_inds[:, 0]], det_rois[rel_inds[:, 1]])
        phr_ori = self.visual_rep(blob_conv, det_rois, pair_inds[:, 1:], device_id)
        vr = phr_ori[vr_indices]
        phr_feats_high = self.get_phr_feats(phr_ori)

        obj_feats = self.obj_mps1(obj_feats, phr_feats_high, im_inds, pair_inds)
        obj_feats = self.obj_mps2(obj_feats, phr_feats_high, im_inds, pair_inds)

        cls_scores = self.ObjClassifier(obj_feats)

        if det_labels is not None:
            det_labels_pred = det_labels
        else:

            det_labels_pred = torch.from_numpy(self.get_nms_preds(det_dists.data.cpu().numpy(), return_dicts['det_boxes_all'])).long().cuda(device_id)

        obj_embeds = self.ori_embed[det_labels_pred].clone().cuda(device_id)

        obj_pos = torch.from_numpy(self.get_obj_pos(det_rois, im_info)).float().cuda(device_id)

        obj_feats_merge = self.merge_low(torch.cat((obj_fmap, obj_embeds, obj_pos), -1)) + self.merge_high(obj_feats)

        prod = self.sbj_map(obj_feats_merge)[rel_inds[:, 1]] * self.obj_map(obj_feats_merge)[rel_inds[:, 2]] * vr

        rel_dists = self.rel_compress(prod)

        if cfg.MODEL.USE_FREQ_BIAS:
            sbj_labels = det_labels_pred[rel_inds[:, 1]]
            obj_labels = det_labels_pred[rel_inds[:, 2]]
            prd_bias_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels - 1, obj_labels - 1), 1))

            rel_dists += prd_bias_scores

        if self.training:
            rel_return_dicts['losses'] = {}
            # rel_loss_weight = torch.ones(cfg.MODEL.NUM_PRD_CLASSES+1).float().cuda(prd_scores.get_device())
            # rel_loss_weight[0] *= cfg.MODEL.BG_VS_FG

            # loss_cls_objs = importance_loss(cls_scores, det_labels, rel_labels, det_rois[:, 0])
            # im_inds_t = torch.from_numpy(im_inds).long().cuda(device_id)
            loss_cls_objs = F.cross_entropy(cls_scores, det_labels)
            prd_weight = torch.ones(self.num_prd_classes).float().cuda(device_id)
            prd_weight[0] = 0.1
            loss_cls_prd = F.cross_entropy(rel_dists, rel_labels[:, -1], weight=prd_weight) * 0.25
            rel_return_dicts['losses']['loss_cls_prd'] = loss_cls_prd
            rel_return_dicts['losses']['loss_cls_objs'] = loss_cls_objs

            prd_cls_preds = rel_dists.max(dim=1)[1].type_as(rel_labels)
            accuracy_cls_prd = prd_cls_preds.eq(rel_labels[:, -1]).float().mean(dim=0)

            obj_cls_preds = cls_scores.max(dim=1)[1].type_as(det_labels)
            accuracy_cls_obj = obj_cls_preds.eq(det_labels).float().mean(dim=0)

            rel_return_dicts['metrics'] = {}
            rel_return_dicts['metrics']['accuracy_cls_prd'] = accuracy_cls_prd
            rel_return_dicts['metrics']['accuracy_cls_obj'] = accuracy_cls_obj

            for k, v in rel_return_dicts['losses'].items():
                rel_return_dicts['losses'][k] = v.unsqueeze(0)
            
            for k, v in rel_return_dicts['metrics'].items():
                rel_return_dicts['metrics'][k] = v.unsqueeze(0)

        else:
            if use_gt_labels:
                rel_return_dicts['sbj_scores'] = np.ones(rel_inds.shape[0], dtype=np.float32)
                rel_return_dicts['obj_scores'] = np.ones(rel_inds.shape[0], dtype=np.float32)
            else:
                rel_return_dicts['sbj_scores'] = F.softmax(cls_scores, -1)[:, 1:].max(-1)[0].data.cpu().numpy()[
                    rel_inds[:, 1]]
                rel_return_dicts['obj_scores'] = F.softmax(cls_scores, -1)[:, 1:].max(-1)[0].data.cpu().numpy()[
                    rel_inds[:, 2]]
            rel_return_dicts['sbj_labels'] = det_labels_pred.data.cpu().numpy()[rel_inds[:, 1]]
            rel_return_dicts['obj_labels'] = det_labels_pred.data.cpu().numpy()[rel_inds[:, 2]]
            rel_return_dicts['sbj_rois'] = det_rois[rel_inds[:, 1]]
            rel_return_dicts['obj_rois'] = det_rois[rel_inds[:, 2]]
            rel_return_dicts['prd_scores'] = F.softmax(rel_dists, -1)
            if do_vis:
                rel_return_dicts['blob_conv'] = blob_conv

        return rel_return_dicts

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value

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


# def importance_losses(cls_scores, cls_labels, rel_labels):

#     device_id = cls_scores.get_device()
#     rel_labels_p = rel_labels[rel_labels[:, -1] > 0]
#     rel_inds = rel_labels_p[:, 1:3].data.cpu().numpy()
#     rel_inds = np.reshape(rel_inds, (-1))
#     logpt = F.log_softmax(cls_scores, dim=-1)
#     logpt = logpt.gather(1, cls_labels.unsqueeze(1)).squeeze()
#     pt = Variable(logpt.data.exp()).detach()
#     num_entities = int(pt.size(0))
#     imp = np.bincount(rel_inds, minlength=num_entities).astype(np.float32)
#     imp /= imp.sum()
#     imp = torch.from_numpy(imp).float().cuda(device_id)
#     bounding = torch.ones_like(imp) * 2.0
#     gamma = torch.min(-((1 - 2.0 * imp) ** 5) * torch.log(2.0 * imp), bounding)
#     loss_cls = (-1 * torch.pow(1-pt, gamma) * logpt).mean()
#     return loss_cls

def importance_loss(inputs, gt_classes, gt_rels, im_inds):
    device_id = inputs.get_device()
    img_nums = int(im_inds.max()) + 1
    filter_inds = torch.nonzero(gt_rels[:, 3] > 0).squeeze()
    filtered_gt_rels = gt_rels[filter_inds].data
    ajacent_matrix = torch.zeros(inputs.size(0), inputs.size(0)).cuda(device_id)
    ajacent_matrix[filtered_gt_rels[:, 1], filtered_gt_rels[:, 2]] = 1.0
    ajacent_matrix[filtered_gt_rels[:, 2], filtered_gt_rels[:, 1]] = 1.0
    pairs_count = ajacent_matrix.sum(0)

    logpt = torch.nn.functional.log_softmax(inputs, dim=-1)
    logpt = logpt.gather(1, gt_classes.unsqueeze(1)).squeeze()
    pt = Variable(logpt.data.exp())

    for i in range(img_nums):
        mask = im_inds == i
        factor = 1.0 / (pairs_count[mask.data]).sum()
        pairs_count[mask.data] *= factor

    bounding = torch.zeros_like(pairs_count).fill_(2.0)
    gama_powers = torch.min(-((1 - 2.0 * pairs_count) ** 5) * torch.log(2.0 * pairs_count), bounding)
    gama = Variable(gama_powers)
    loss = (-1 * torch.pow(1 - pt, gama) * logpt).mean()


    return loss