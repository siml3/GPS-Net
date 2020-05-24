import torch
from torch import nn
import numpy as np
from modeling_rel.word_vecs import obj_edge_vectors
from core.config import cfg
import torch.nn.functional as F
from torch.autograd import Variable


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0

    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)


def entity_losses(entity_cls_scores, entity_labels_int32, fg_only=False):
    device_id = entity_cls_scores.get_device()
    entity_labels = Variable(torch.from_numpy(entity_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_entity = F.cross_entropy(entity_cls_scores, entity_labels)
    # class accuracy
    entity_cls_preds = entity_cls_scores.max(dim=1)[1].type_as(entity_labels)
    accuracy_cls_entity = entity_cls_preds.eq(entity_labels).float().mean(dim=0)

    return loss_cls_entity, accuracy_cls_entity


def entity_losses_imp(entity_cls_scores, entity_labels_int32, gamma, fg_only=False):
    device_id = entity_cls_scores.get_device()
    entity_labels = Variable(torch.from_numpy(entity_labels_int32.astype('int64'))).cuda(device_id)
    logpt = F.log_softmax(entity_cls_scores, -1)
    logpt = logpt.gather(1, entity_labels.unsqueeze(1)).squeeze()
    pt = logpt.data.exp()
    gamma_t = torch.from_numpy(gamma).float().cuda(device_id)
    loss_cls_entity = (-torch.pow(1.0 - pt, gamma_t) * logpt).mean()
    # class accuracy
    entity_cls_preds = entity_cls_scores.max(dim=1)[1].type_as(entity_labels)
    accuracy_cls_entity = entity_cls_preds.eq(entity_labels).float().mean(dim=0)

    return loss_cls_entity, accuracy_cls_entity


class Merge_OBJ_Feats(nn.Module):
    """get merged obj features."""

    def __init__(self, dim_in, embed_dim, dim_out):
        super(Merge_OBJ_Feats, self).__init__()

        self.dim_in = dim_in
        self.embed_dim = embed_dim
        self.dim_out = dim_out
        classes = cfg.LANGUAGE.OBJS_CLASSES
        assert cfg.MODEL.NUM_CLASSES == len(classes)
        classes = [s.lower() for s in classes]
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=0.01 / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        self.obj_embed = nn.Embedding(cfg.MODEL.NUM_CLASSES, self.embed_dim)
        obj_wvs = obj_edge_vectors(classes, wv_dim=self.embed_dim)
        self.obj_embed.weight.data = obj_wvs.clone()
        self.reduce_dim = nn.Linear(self.dim_in + self.embed_dim + 128, self.dim_out)

    def forward(self, vis_feats, det_rois, det_scores, im_info):
        device_id = vis_feats.get_device()
        im_inds = det_rois[:, 0].astype(np.int64)
        scale_factor = 800 / torch.stack((im_info[im_inds, 1], im_info[im_inds, 0], \
                                          im_info[im_inds, 1], im_info[im_inds, 0]), -1)
        center_detrois = center_size(torch.from_numpy(det_rois[:, 1:]).float()) * scale_factor

        pos_embed = self.pos_embed(center_detrois.cuda(device_id))
        if not isinstance(det_scores, torch.Tensor):
            det_scores = torch.from_numpy(det_scores).float().cuda(device_id)
        obj_embed = det_scores @ self.obj_embed.weight
        assert vis_feats.size(0) == pos_embed.size(0)
        assert pos_embed.size(0) == obj_embed.size(0)
        return self.reduce_dim(torch.cat((vis_feats, pos_embed, obj_embed), -1))


def mc_matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)


class Get_Atten_map_mc(nn.Module):

    def __init__(self, input_dims, p):
        super(Get_Atten_map_mc, self).__init__()
        self.input_dims = input_dims
        self.p = p
        self.ws = nn.Linear(self.input_dims, self.input_dims)
        self.wo = nn.Linear(self.input_dims, self.input_dims)
        self.w = nn.Linear(self.input_dims, self.p)

    def forward(self, obj_feats, rel_inds, union_feats, n_nodes):
        device_id = obj_feats.get_device()
        prod = self.ws(obj_feats)[rel_inds[:, 1]] * self.wo(obj_feats)[rel_inds[:, 2]]
        atten_f = self.w(prod * union_feats)
        atten_tensor = torch.zeros(n_nodes, n_nodes, self.p).cuda(device_id).float()
        # atten_tensor_mask = torch.zeros(n_nodes, n_nodes, self.p).cuda(device_id).float()
        head = rel_inds[:, 1:].min()
        atten_tensor[rel_inds[:, 1] - head, rel_inds[:, 2] - head] += atten_f
        # atten_tensor_mask[rel_inds[:, 1] - head, rel_inds[:, 2] - head] += 1.0
        remove_self_mask = 1 - torch.eye(n_nodes).float().unsqueeze(-1).repeat(1, 1, self.p).cuda()
        atten_tensor = F.sigmoid(atten_tensor)
        atten_tensor = atten_tensor * remove_self_mask
        return atten_tensor / atten_tensor.sum(1)


class GRU(nn.Module):

    def __init__(self, input_dims):
        super(GRU, self).__init__()
        self.input_dims = input_dims
        self.w3 = nn.Linear(input_dims, input_dims)
        self.u3 = nn.Linear(input_dims, input_dims)
        self.w4 = nn.Linear(input_dims, input_dims)
        self.u4 = nn.Linear(input_dims, input_dims)
        self.w5 = nn.Linear(input_dims, input_dims)
        self.u5 = nn.Linear(input_dims, input_dims)

    def forward(self, inputs, context_feats):
        zv = torch.sigmoid(self.w3(context_feats) + self.u3(inputs))

        rv = torch.sigmoid(self.w4(context_feats) + self.u4(inputs))

        hv = torch.tanh(self.w5(context_feats) + self.u5(rv * inputs))

        outputs = (1 - zv) * inputs + zv * hv

        return outputs


class Message_Passing4OBJ(nn.Module):

    def __init__(self, input_dims):
        super(Message_Passing4OBJ, self).__init__()
        self.input_dims = input_dims
        self.trans = nn.Sequential(nn.Linear(self.input_dims, input_dims // 4),
                                   nn.LayerNorm(self.input_dims // 4), nn.ReLU(),
                                   nn.Linear(self.input_dims // 4, self.input_dims))

        self.get_atten_tensor = Get_Atten_map_mc(self.input_dims, p=1)

        self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims // 2),
                                  nn.ReLU())

        # self.gru_unit = GRU(self.input_dims)
        # self.conv = nn.Linear(self.input_dims, self.input_dims // 4) # use rel in the end.

    def forward(self, obj_feats, phr_feats, im_inds, rel_inds):
        # assert ret['mps_rois'][:, 0].max() == 0
        # assert ret['edge_rois'][:, 0].max() == 0
        num_img = int(im_inds.max()) + 1
        obj_indices_sets = [np.where(im_inds == i)[0] for i in range(num_img)]
        obj2obj_feats_sets = []
        rel_indices_sets = [np.where(rel_inds[:, 0] == i)[0] for i in range(num_img)]

        for i, obj_indices in enumerate(obj_indices_sets):
            entities_num = obj_indices.shape[0]
            cur_obj_feats = obj_feats[obj_indices]
            rel_indices = rel_indices_sets[i]
            atten_tensor = self.get_atten_tensor(obj_feats, rel_inds[rel_indices], phr_feats[rel_indices], entities_num)
            # gap_mask = (atten_tensor.max(1, keepdim=True)[0] - atten_tensor.min(1, keepdim=True)[0]) > 0.4
            # max_mask = atten_tensor == atten_tensor.max(1, keepdim=True)[0]
            # atten_tensor = atten_tensor * gap_mask.float() * max_mask.float()
            atten_tensor_t = atten_tensor.transpose(1, 0)
            atten_tensor = torch.cat((atten_tensor, atten_tensor_t), -1)
            context_feats = mc_matmul(atten_tensor, self.conv(cur_obj_feats))
            # context_feats = F.relu(mc_matmul(atten_tensor, self.conv(cur_obj_feats))) # use relu in the end.
            obj2obj_feats_sets.append(self.trans(context_feats))

        return F.relu(obj_feats + torch.cat(obj2obj_feats_sets, 0))

# class Get_Atten_map_mc_nopa(nn.Module):
#
#     def __init__(self, input_dims, p):
#         super(Get_Atten_map_mc_nopa, self).__init__()
#         self.input_dims = input_dims
#         self.p = p
#
#     def forward(self, rel_inds, edges, n_nodes):
#         device_id = edges.get_device()
#         atten_tensor = torch.zeros(n_nodes, n_nodes, self.p).cuda(device_id).float()
#         head = rel_inds[:, 1:].min()
#         atten_tensor[rel_inds[:, 1] - head, rel_inds[:, 2] - head] += edges
#         atten_tensor = F.sigmoid(atten_tensor)
#         atten_tensor = atten_tensor *(1.0 - torch.eye(n_nodes).float().unsqueeze(-1).repeat(1, 1, self.p).cuda(device_id))
#         return atten_tensor / torch.max(torch.ones_like(atten_tensor.sum(1))*1e-12, atten_tensor.sum(1))
#         # return atten_tensor / atten_tensor.sum(1)
#
# class Message_Passing4OBJ_new(nn.Module):
#
#     def __init__(self, input_dims):
#         super(Message_Passing4OBJ_new, self).__init__()
#         self.input_dims = input_dims
#         self.trans = nn.Sequential(nn.Linear(self.input_dims, input_dims // 4),
#                                    nn.LayerNorm(self.input_dims // 4), nn.ReLU(),
#                                    nn.Linear(self.input_dims // 4, self.input_dims))
#
#         self.get_atten_tensor = Get_Atten_map_mc_nopa(self.input_dims, p=1)
#
#         self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims // 2),
#                                   nn.ReLU())
#         self.p = 1
#         self.ws = nn.Linear(self.input_dims, self.input_dims)
#         self.wo = nn.Linear(self.input_dims, self.input_dims)
#         self.w = nn.Linear(self.input_dims, self.p)
#         # self.conv = nn.Linear(self.input_dims, self.input_dims // 4) # use rel in the end.
#
#     def forward(self, obj_feats, phr_feats, im_inds, rel_inds):
#         # assert ret['mps_rois'][:, 0].max() == 0
#         # assert ret['edge_rois'][:, 0].max() == 0
#         num_img = int(im_inds.max()) + 1
#         obj_indices_sets = [np.where(im_inds == i)[0] for i in range(num_img)]
#         obj2obj_feats_sets = []
#         rel_indices_sets = [np.where(rel_inds[:, 0] == i)[0] for i in range(num_img)]
#         edges = self.w(self.ws(obj_feats)[rel_inds[:, 1]] * self.wo(obj_feats)[rel_inds[:, 2]] * phr_feats)
#
#         for i, obj_indices in enumerate(obj_indices_sets):
#             entities_num = obj_indices.shape[0]
#             cur_obj_feats = obj_feats[obj_indices]
#             rel_indices = rel_indices_sets[i]
#             atten_tensor = self.get_atten_tensor(rel_inds[rel_indices], edges[rel_indices], entities_num)
#             atten_tensor_t = atten_tensor.transpose(1, 0)
#             atten_tensor = torch.cat((atten_tensor, atten_tensor_t), -1)
#             context_feats = mc_matmul(atten_tensor, self.conv(cur_obj_feats))
#             # context_feats = F.relu(mc_matmul(atten_tensor, self.conv(cur_obj_feats))) # use relu in the end.
#             obj2obj_feats_sets.append(self.trans(context_feats))
#
#
#         return F.relu(obj_feats + torch.cat(obj2obj_feats_sets, 0))