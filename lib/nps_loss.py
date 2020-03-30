import torch
from torch.autograd import Variable

def node_priority_sensitive_loss(inputs, gt_classes, gt_rels, im_inds, size_average=True):

    img_nums = int(im_inds.max()) + 1
    filter_inds = torch.nonzero(gt_rels[:, 3]>0).squeeze()
    filtered_gt_rels = gt_rels[filter_inds].data.cpu().numpy()
    ajacent_matrix = np.zeros((inputs.size(0), inputs.size(0)))
    ajacent_matrix[filtered_gt_rels[:, 1], filtered_gt_rels[:, 2]] = 1.0
    ajacent_matrix[filtered_gt_rels[:, 2], filtered_gt_rels[:, 1]] = 1.0
    pairs_count = torch.from_numpy(ajacent_matrix.sum(0)).float().cuda(inputs.get_device())

    logpt = torch.nn.functional.log_softmax(inputs, dim=-1)
    logpt = logpt.gather(1, gt_classes.unsqueeze(1)).squeeze()
    pt = Variable(logpt.data.exp())

    for i in range(img_nums):
        mask = im_inds == i
        factor = 1.0 / (pairs_count[mask.data]).sum()
        pairs_count[mask.data] *= factor

    gama_powers = torch.clamp(-((1 - 2.0*pairs_count) ** 5.0) * torch.log(2.0*pairs_count), max=2.0)
    gama = Variable(gama_powers)
    loss = (-1 * torch.pow(1-pt, gama) * logpt)
    
    return loss.mean() if size_average else loss.sum()