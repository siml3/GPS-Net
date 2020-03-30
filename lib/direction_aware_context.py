import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

class Get_Atten_map_mc(nn.Module):

    def __init__(self, input_dims, p):
        super(Get_Atten_map_mc, self).__init__()
        self.input_dims = input_dims
        self.p = p
        self.ws = nn.Linear(input_dims, input_dims)
        self.wo = nn.Linear(input_dims, input_dims)
        self.w = nn.Linear(self.input_dims, self.p)

    def forward(self, obj_feats, rel_inds, union_feats, n_nodes):
        atten_f = self.w(self.ws(obj_feats)[rel_inds[:, 1]] * \
            self.wo(obj_feats)[rel_inds[:, 2]] * union_feats)
        atten_tensor = Variable(torch.zeros(n_nodes, n_nodes, self.p)).cuda().float()
        head = rel_inds[:, 1:].min()
        atten_tensor[rel_inds[:, 1] - head, rel_inds[:, 2] - head] += atten_f
        atten_tensor = F.sigmoid(atten_tensor)
        atten_tensor = atten_tensor * (1- Variable(torch.eye(n_nodes).float()).unsqueeze(-1).repeat(1, 1, self.p).cuda())
        return atten_tensor/torch.sum(atten_tensor,1)

def mc_matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)


class LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-5):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.

        Thanks to CyberZHG's code in https://github.com/CyberZHG/torch-layer-normalization.git .
        """
        super(LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )

class Direction_Aware_MP(nn.Module):

    def __init__(self, input_dims):
        super(Direction_Aware_MP, self).__init__()
        self.input_dims = input_dims
        self.trans = nn.Sequential(nn.Linear(self.input_dims, input_dims//4), 
                                    LayerNorm(self.input_dims//4), nn.ReLU(inplace=True),
                                    nn.Linear(self.input_dims//4, self.input_dims))

        self.get_atten_tensor = Get_Atten_map_mc(self.input_dims, p=1)

        self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims//2),
                                    nn.ReLU(inplace=True))
        # self.conv = nn.Linear(self.input_dims, self.input_dims // 4) # use rel in the end.
        
    def forward(self, obj_feats, phr_feats, im_inds, rel_inds):

        num_img = int(im_inds.max()) + 1
        obj_indices_sets = [torch.nonzero(im_inds==i).data.squeeze() for i in range(num_img)]
        obj2obj_feats_sets = []
        rel_indices_sets = [torch.nonzero(rel_inds[:, 0]==i).squeeze() for i in range(num_img)]

        for i, obj_indices in enumerate(obj_indices_sets):
            entities_num = obj_indices.size(0)
            cur_obj_feats = obj_feats[obj_indices]
            rel_indices = rel_indices_sets[i]
            atten_tensor = self.get_atten_tensor(obj_feats, rel_inds[rel_indices], phr_feats[rel_indices], entities_num)
            atten_tensor_t = torch.transpose(atten_tensor,1,0)
            atten_tensor = torch.cat((atten_tensor,atten_tensor_t),-1)
            context_feats = mc_matmul(atten_tensor, self.conv(cur_obj_feats))
            obj2obj_feats_sets.append(self.trans(context_feats))

        return F.relu(obj_feats + torch.cat(obj2obj_feats_sets, 0), inplace=True)