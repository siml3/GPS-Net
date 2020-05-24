# Written by Ji Zhang in 2019

import numpy as np
from numpy import linalg as la
import json
import logging

from torch import nn
from torch.nn import init
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from modeling_rel.generate_rel_proposal_labels import GenerateRelProposalLabelsOp
import modeling.FPN as FPN
import utils_rel.boxes_rel as box_utils_rel
import utils.fpn as fpn_utils
import numpy.random as npr
import utils.boxes as box_utils


logger = logging.getLogger(__name__)


def generic_relpn_outputs():
    return single_scale_relpn_outputs()


class single_scale_relpn_outputs(nn.Module):
    """Add RelPN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self):
        super().__init__()
        
        self.RelPN_GenerateProposalLabels = GenerateRelProposalLabelsOp()
        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds

    def remove_self_pairs(self, det_size, sbj_inds, obj_inds):
        mask = np.ones(sbj_inds.shape[0], dtype=bool)
        for i in range(det_size):
            mask[i + det_size * i] = False
        keeps = np.where(mask)[0]
        sbj_inds = sbj_inds[keeps]
        obj_inds = obj_inds[keeps]
        return sbj_inds, obj_inds

    def forward(self, det_rois, det_labels, det_scores, im_info, dataset_name, roidb=None):
        """
        det_rois: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """
        
        # Get pairwise proposals first
        if roidb is not None:
            # we always feed one image per batch during training
            assert len(roidb) == 1

        sbj_inds = np.repeat(np.arange(det_rois.shape[0]), det_rois.shape[0])
        obj_inds = np.tile(np.arange(det_rois.shape[0]), det_rois.shape[0])
        # remove self paired rois
        if det_rois.shape[0] > 1:  # no pairs to remove when there is at most one detection
            sbj_inds, obj_inds = self.remove_self_pairs(det_rois.shape[0], sbj_inds, obj_inds)
        sbj_rois = det_rois[sbj_inds]
        obj_rois = det_rois[obj_inds]
            
        im_scale = im_info.data.numpy()[:, 2][0]
        sbj_boxes = sbj_rois[:, 1:] / im_scale
        obj_boxes = obj_rois[:, 1:] / im_scale
        # filters out those roi pairs whose boxes are not overlapping in the original scales
        if cfg.MODEL.USE_OVLP_FILTER:
            ovlp_so = box_utils_rel.bbox_pair_overlaps(
                sbj_boxes.astype(dtype=np.float32, copy=False),
                obj_boxes.astype(dtype=np.float32, copy=False))
            ovlp_inds = np.where(ovlp_so > 0)[0]
            sbj_inds = sbj_inds[ovlp_inds]
            obj_inds = obj_inds[ovlp_inds]
            sbj_rois = sbj_rois[ovlp_inds]
            obj_rois = obj_rois[ovlp_inds]
            sbj_boxes = sbj_boxes[ovlp_inds]
            obj_boxes = obj_boxes[ovlp_inds]
            
        return_dict = {}
        if self.training:
            # Add binary relationships
            blobs_out = self.RelPN_GenerateProposalLabels(sbj_rois, obj_rois, det_rois, roidb, im_info)
            return_dict.update(blobs_out)
        else:
            sbj_labels = det_labels[sbj_inds]
            obj_labels = det_labels[obj_inds]
            sbj_scores = det_scores[sbj_inds]
            obj_scores = det_scores[obj_inds]
            rel_rois = box_utils_rel.rois_union(sbj_rois, obj_rois)
            return_dict['det_rois'] = det_rois
            return_dict['sbj_inds'] = sbj_inds
            return_dict['obj_inds'] = obj_inds
            return_dict['sbj_rois'] = sbj_rois
            return_dict['obj_rois'] = obj_rois
            return_dict['rel_rois'] = rel_rois
            return_dict['sbj_labels'] = sbj_labels
            return_dict['obj_labels'] = obj_labels
            return_dict['sbj_scores'] = sbj_scores
            return_dict['obj_scores'] = obj_scores
            return_dict['fg_size'] = np.array([sbj_rois.shape[0]], dtype=np.int32)

            im_scale = im_info.data.numpy()[:, 2][0]
            im_w = im_info.data.numpy()[:, 1][0]
            im_h = im_info.data.numpy()[:, 0][0]
            if cfg.MODEL.USE_SPATIAL_FEAT:
                spt_feat = box_utils_rel.get_spt_features(sbj_boxes, obj_boxes, im_w, im_h)
                return_dict['spt_feat'] = spt_feat
            if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE:
                return_dict['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False) - 1  # det_labels start from 1
                return_dict['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False) - 1  # det_labels start from 1
            if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                lvl_min = cfg.FPN.ROI_MIN_LEVEL
                lvl_max = cfg.FPN.ROI_MAX_LEVEL
                # when use min_rel_area, the same sbj/obj area could be mapped to different feature levels
                # when they are associated with different relationships
                # Thus we cannot get det_rois features then gather sbj/obj features
                # The only way is gather sbj/obj per relationship, thus need to return sbj_rois/obj_rois
                rois_blob_names = ['det_rois', 'rel_rois']
                for rois_blob_name in rois_blob_names:
                    # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                    target_lvls = fpn_utils.map_rois_to_fpn_levels(
                        return_dict[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                    fpn_utils.add_multilevel_roi_blobs(
                        return_dict, rois_blob_name, return_dict[rois_blob_name], target_lvls,
                        lvl_min, lvl_max)

        return return_dict

class single_scale_pairs_pn_outputs(nn.Module):
    """Add RelPN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self, overlap=False):
        super().__init__()
        self.overlap = overlap
       
        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds

    def remove_self_pairs(self, det_size, sbj_inds, obj_inds):
        mask = np.ones(sbj_inds.shape[0], dtype=bool)
        for i in range(det_size):
            mask[i + det_size * i] = False
        keeps = np.where(mask)[0]
        sbj_inds = sbj_inds[keeps]
        obj_inds = obj_inds[keeps]
        return sbj_inds, obj_inds

    def forward(self, det_rois, det_labels, det_scores, im_info, dataset_name, roidb=None):
        """
        det_rois: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """
        
        

        sbj_inds = np.repeat(np.arange(det_rois.shape[0]), det_rois.shape[0])
        obj_inds = np.tile(np.arange(det_rois.shape[0]), det_rois.shape[0])
        # remove self paired rois
        if det_rois.shape[0] > 1:  # no pairs to remove when there is at most one detection
            sbj_inds, obj_inds = self.remove_self_pairs(det_rois.shape[0], sbj_inds, obj_inds)
        sbj_rois = det_rois[sbj_inds]
        obj_rois = det_rois[obj_inds]
            
        im_scale = im_info.data.numpy()[:, 2][0]
        sbj_boxes = sbj_rois[:, 1:] / im_scale
        obj_boxes = obj_rois[:, 1:] / im_scale
        # filters out those roi pairs whose boxes are not overlapping in the original scales
        if self.overlap:
            ovlp_so = box_utils_rel.bbox_pair_overlaps(
                sbj_boxes.astype(dtype=np.float32, copy=False),
                obj_boxes.astype(dtype=np.float32, copy=False))
            ovlp_inds = np.where((ovlp_so > 0) & (ovlp_so < 0.5))[0]
            if ovlp_inds.size > 0:
                sbj_inds = sbj_inds[ovlp_inds]
                obj_inds = obj_inds[ovlp_inds]
                sbj_rois = sbj_rois[ovlp_inds]
                obj_rois = obj_rois[ovlp_inds]

        return_dict = {}
        sbj_labels = det_labels[sbj_inds]
        obj_labels = det_labels[obj_inds]
        sbj_scores = det_scores[sbj_inds]
        obj_scores = det_scores[obj_inds]
        rel_rois = box_utils_rel.rois_union(sbj_rois, obj_rois)
        return_dict['det_rois'] = det_rois
        return_dict['sbj_inds'] = sbj_inds
        return_dict['obj_inds'] = obj_inds
        return_dict['sbj_rois'] = sbj_rois
        return_dict['obj_rois'] = obj_rois
        return_dict['rel_rois'] = rel_rois
        return_dict['sbj_labels'] = sbj_labels
        return_dict['obj_labels'] = obj_labels
        return_dict['sbj_scores'] = sbj_scores
        return_dict['obj_scores'] = obj_scores
        return_dict['fg_size'] = np.array([sbj_rois.shape[0]], dtype=np.int32)


        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
            lvl_min = cfg.FPN.ROI_MIN_LEVEL
            lvl_max = cfg.FPN.ROI_MAX_LEVEL
            # when use min_rel_area, the same sbj/obj area could be mapped to different feature levels
            # when they are associated with different relationships
            # Thus we cannot get det_rois features then gather sbj/obj features
            # The only way is gather sbj/obj per relationship, thus need to return sbj_rois/obj_rois
            rois_blob_names = ['det_rois', 'rel_rois']
            for rois_blob_name in rois_blob_names:
                # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                target_lvls = fpn_utils.map_rois_to_fpn_levels(
                    return_dict[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                fpn_utils.add_multilevel_roi_blobs(
                    return_dict, rois_blob_name, return_dict[rois_blob_name], target_lvls,
                    lvl_min, lvl_max)

        return return_dict

def rel_assignments(im_inds, rpn_rois, roi_gtlabels, roidb, im_info,
                    num_sample_per_gt=4, filter_non_overlap=True):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    fg_rels_per_image = int(np.round(cfg.TRAIN.FG_REL_FRACTION * cfg.TRAIN.RELS_PER_IMG_REFINE))

    num_im = int(im_inds.max() + 1)
    indices_sets = [np.where(im_inds == i)[0] for i in range(num_im)]

    # print("Pred inds {} pred boxes {} pred box labels {} gt classes {} gt rels {}".format(
    #     pred_inds_np, pred_boxes_np, pred_boxlabels_np, gt_classes_np, gt_rels_np
    # ))

    rel_labels = []
    num_box_seen = 0
    for i, indices in enumerate(indices_sets):
        
        gt_boxes_i = roidb[i]['boxes']
        gt_rois_i = gt_boxes_i * im_info[i, 2]
        gt_classes_i = roidb[i]['gt_classes']
        sbj_gt_boxes_i = roidb[i]['sbj_gt_boxes']
        obj_gt_boxes_i = roidb[i]['obj_gt_boxes']
        prd_gt_classes_i = roidb[i]['prd_gt_classes']
        if cfg.MODEL.USE_BG:
            prd_gt_classes_i += 1

        sbj_gt_inds_i = box_utils.bbox_overlaps(sbj_gt_boxes_i, gt_boxes_i).argmax(-1)
        obj_gt_inds_i = box_utils.bbox_overlaps(obj_gt_boxes_i, gt_boxes_i).argmax(-1)
        gt_rels_i = np.stack((sbj_gt_inds_i, obj_gt_inds_i, prd_gt_classes_i), -1)

        # [num_pred, num_gt]
        pred_rois_i = rpn_rois[indices, 1:]
        pred_roilabels_i = roi_gtlabels[indices]

        ious = box_utils.bbox_overlaps(pred_rois_i, gt_rois_i)
        is_match = (pred_roilabels_i[:,None] == gt_classes_i[None]) & (ious >= cfg.TRAIN.FG_THRESH)

        # FOR BG. Limit ourselves to only IOUs that overlap, but are not the exact same box
        pbi_iou = box_utils.bbox_overlaps(pred_rois_i, pred_rois_i)
        if filter_non_overlap:
            rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
            rels_intersect = rel_possibilities
        else:
            rel_possibilities = np.ones((pred_rois_i.shape[0], pred_rois_i.shape[0]),
                                        dtype=np.int64) - np.eye(pred_rois_i.shape[0],
                                                                 dtype=np.int64)
            rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)

        # ONLY select relations between ground truth because otherwise we get useless data
        rel_possibilities[pred_roilabels_i == 0] = 0
        rel_possibilities[:, pred_roilabels_i == 0] = 0

        # Sample the GT relationships.
        fg_rels = []
        p_size = []
        for i, (from_gtind, to_gtind, rel_id) in enumerate(gt_rels_i):
            fg_rels_i = []
            fg_scores_i = []

            for from_ind in np.where(is_match[:, from_gtind])[0]:
                for to_ind in np.where(is_match[:, to_gtind])[0]:
                    if from_ind != to_ind:
                        fg_rels_i.append((from_ind, to_ind, rel_id))
                        fg_scores_i.append((ious[from_ind, from_gtind] * ious[to_ind, to_gtind]))
                        rel_possibilities[from_ind, to_ind] = 0
            if len(fg_rels_i) == 0:
                continue
            p = np.array(fg_scores_i)
            p = p / p.sum()
            p_size.append(p.shape[0])
            num_to_add = min(p.shape[0], num_sample_per_gt)
            for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                fg_rels.append(fg_rels_i[rel_to_add])

        fg_rels = np.array(fg_rels, dtype=np.int64)
        if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
            fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
        elif fg_rels.size == 0:
            fg_rels = np.zeros((0, 3), dtype=np.int64)

        bg_rels = np.column_stack(np.where(rel_possibilities))
        bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0], dtype=np.int64)))

        num_bg_rel = min(cfg.TRAIN.RELS_PER_IMG_REFINE - fg_rels.shape[0], bg_rels.shape[0])
        if bg_rels.size > 0:
            # Sample 4x as many intersecting relationships as non-intersecting.
            # bg_rels_intersect = rels_intersect[bg_rels[:, 0], bg_rels[:, 1]]
            # p = bg_rels_intersect.astype(np.float32)
            # p[bg_rels_intersect == 0] = 0.2
            # p[bg_rels_intersect == 1] = 0.8
            # p /= p.sum()
            bg_rels = bg_rels[
                np.random.choice(bg_rels.shape[0],
                                 #p=p,
                                 size=num_bg_rel, replace=False)]
        else:
            bg_rels = np.zeros((0, 3), dtype=np.int64)

        if fg_rels.size == 0 and bg_rels.size == 0:
            # Just put something here
            bg_rels = np.array([[0, 0, 0]], dtype=np.int64)

        # print("GTR {} -> AR {} vs {}".format(gt_rels.shape, fg_rels.shape, bg_rels.shape))
        all_rels_i = np.concatenate((fg_rels, bg_rels), 0)
        all_rels_i[:,0:2] += num_box_seen

        all_rels_i = all_rels_i[np.lexsort((all_rels_i[:,1], all_rels_i[:,0]))]

        rel_labels.append(np.column_stack((
            i*np.ones(all_rels_i.shape[0], dtype=np.int64),
            all_rels_i,
        )))

        num_box_seen += pred_rois_i.shape[0]
    
    rel_labels = np.concatenate(rel_labels, 0)
    return rel_labels[:, :-1], rel_labels

class Pairs_Pruning(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        
        self.Ws = nn.Sequential(
            nn.Linear(input_dims, output_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dims * 2, output_dims)
        )

        self.Wo = nn.Sequential(
            nn.Linear(input_dims, output_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dims * 2, output_dims)
        )

    def co_nms(self, p_det_rois, scores):
        pre_topN = cfg.TRAIN.PRUNE_PAIRS_PRE_NMS_TOP_N if self.training else cfg.TEST.PRUNE_PAIRS_PRE_NMS_TOP_N
        post_topN = cfg.TRAIN.PRUNE_PAIRS_POST_NMS_TOP_N if self.training else cfg.TEST.PRUNE_PAIRS_POST_NMS_TOP_N
        nms_thr = cfg.TRAIN.PRUNE_PAIRS_NMS_THRESH  if self.training else cfg.TEST.PRUNE_PAIRS_NMS_THRESH
        if p_det_rois.shape[0] > pre_topN:
            keep_inds = np.argsort(-scores.ravel())[:pre_topN]
            p_det_rois = p_det_rois[keep_inds]
        else:
            keep_inds = np.arange(p_det_rois.shape[0], dtype=np.int64)
        p_dets = np.concatenate((p_det_rois, scores[keep_inds]), -1)
        keep_inds_nms = box_utils.co_nms(p_dets, nms_thr)
        keep_inds = keep_inds[keep_inds_nms]
        if keep_inds.shape[0] > post_topN:
            sort_inds = np.argsort(-scores[keep_inds].ravel())[:post_topN]
            keep_inds = keep_inds[sort_inds]
        return keep_inds
        
    def _sample_pairs(self, det_rois, edge_inds, im_info, roidb):
        sbj_gt_rois = roidb['sbj_gt_boxes'] * im_info[2].data.cpu().numpy()
        obj_gt_rois = roidb['obj_gt_boxes'] * im_info[2].data.cpu().numpy()
        p_ious = (box_utils.bbox_overlaps(det_rois[edge_inds[:, 1]][:, 1:], sbj_gt_rois) * \
                  box_utils.bbox_overlaps(det_rois[edge_inds[:, 2]][:, 1:], obj_gt_rois)).max(-1)

        fg_inds = np.where(p_ious >= cfg.TRAIN.PRUNE_PAIRS_POSTIVE_OVERLAP)[0]
        bg_inds = np.where(p_ious < cfg.TRAIN.PRUNE_PAIRS_NEGATIVE_OVERLAP)[0]
        num_fg = min(fg_inds.shape[0], cfg.TRAIN.PRUNE_PAIRS_FG_FRACTION * cfg.TRAIN.PRUNE_PAIRS_BATCHSIZE)
        num_bg = min(bg_inds.shape[0], cfg.TRAIN.PRUNE_PAIRS_BATCHSIZE - num_fg)
        if fg_inds.shape[0] > num_fg:
            fg_inds = npr.choice(fg_inds, size=int(num_fg), replace=False)
        if bg_inds.shape[0] > num_bg:
            bg_inds = npr.choice(bg_inds, size=int(num_bg), replace=False)
        labels = np.concatenate((np.ones_like(fg_inds), np.zeros_like(bg_inds)), 0)
        keep_inds = np.concatenate((fg_inds, bg_inds), 0)
        return keep_inds, labels

    def forward(self, det_rois, det_dists, edge_inds, im_info, roidb=None):

        num_img = int(edge_inds[:, 0].max()) + 1
        edge_indices_sets = [np.where(edge_inds[:, 0] == i)[0] for i in range(num_img)]
        keep_edge_indices = []
        sbj_feats = self.Ws(det_dists)[edge_inds[:, 1]].unsqueeze(1)
        obj_feats = self.Wo(det_dists)[edge_inds[:, 2]].unsqueeze(-1)
        pair_scores = F.sigmoid(torch.bmm(sbj_feats, obj_feats)).squeeze(-1)
        p_det_rois = np.concatenate((det_rois[:, 1:][edge_inds[:, 1]], det_rois[:, 1:][edge_inds[:, 2]]), -1)
        if self.training:
            sample_pairs_scores_sets = []
            sample_pairs_labels_sets = []
        for i, edge_indices in enumerate(edge_indices_sets):
            keep = self.co_nms(p_det_rois[edge_indices], pair_scores.data.cpu().numpy()[edge_indices])
            keep_edge_indices.append(edge_indices[keep])
            if self.training:
                assert roidb is not None
                sample_indices, sample_labels = self._sample_pairs(det_rois, edge_inds[edge_indices], im_info[i], roidb[i])
                sample_pairs_scores_sets.append(pair_scores[edge_indices[sample_indices]])
                sample_pairs_labels_sets.append(sample_labels)

        keep_edge_indices = np.concatenate(keep_edge_indices, 0)
        if self.training:
            sample_pairs_scores = torch.cat(sample_pairs_scores_sets, 0)
            sample_pairs_labels = np.concatenate(sample_pairs_labels_sets, 0)

            return keep_edge_indices, sample_pairs_scores, sample_pairs_labels
        
        else:

            return keep_edge_indices, None, None

def pairs_pruning_losses(pair_scores, pair_labels):
    device_id = pair_scores.get_device()
    if pair_scores.dim() > 1:
        pair_scores = pair_scores.squeeze()
    pair_labels_t = Variable(torch.from_numpy(pair_labels).float().cuda(device_id))
    loss_pairs = F.binary_cross_entropy(pair_scores, pair_labels_t)

    pairs_preds = (pair_scores > 0.5).float()
    accuracy_pairs = pairs_preds.eq(pair_labels_t).float().mean(dim=0)

    return loss_pairs, accuracy_pairs






