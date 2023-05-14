import torch
import torch.nn as nn
import torch.nn.functional as F

from core.loss import loss_register
from core.loss.base.atom_loss_combinator import AtomLossCombinator
from core.loss.utils import loss_dict_remake_wrapper
from utils.tensor_operator import to_device


@loss_register
class PlainContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(PlainContrastiveLoss, self).__init__()
        self.cfg = cfg
        self.cal_loss_funcs = AtomLossCombinator(self.cfg)

    @loss_dict_remake_wrapper
    def forward(self, embedding_1):
        return self.cal_loss_funcs(embedding_1)


def calc_contrastive_loss(X, Y, args):
    CL_loss_1 = do_CL(X, Y, args)
    CL_loss_2 = do_CL(Y, X, args)

    CD_loss_1 = cdist_loss(X, Y, args)
    CD_loss_2 = cdist_loss(Y, X, args)
    # todo: Is there any trick for this
    res = 0.9 * (CL_loss_1 + CL_loss_2) / 2 + 0.1 * (CD_loss_1 + CD_loss_2) / 2
    return res
    # return CL_loss_1 + CL_loss_2


def calc_contrastive_loss_with_intra_img(a, b, args, ori_b=None):
    ab_cl_loss = calc_contrastive_loss(a, b, args)
    if ori_b is None:
        return ab_cl_loss


def cl_with_mask(a, b, a2b, T, mask_diag=False):
    n_a2b = 1 - a2b
    a_f_num, b_f_num = a2b.size()
    if mask_diag:
        diag = torch.eye(a_f_num, b_f_num, device=a.device)
        n_diag = 1 - diag
    a2b_sim = torch.exp(a @ b.t() / T)
    a2b_n_sum = (a2b_sim * n_a2b).sum(dim=1, keepdims=True).repeat(1, b_f_num)
    a2b_loss = a2b_sim * a2b / (a2b_sim + a2b_n_sum)
    if mask_diag:
        a2b_loss *= n_diag
        denorm = (a2b - diag).sum(dim=1)
    else:
        denorm = a2b.sum(dim=1)
    a2b_loss = a2b_loss.sum(dim=1) / denorm
    log_a2b_loss = - torch.log(a2b_loss).mean()
    return log_a2b_loss


def do_inter_intra_CL(g, i, g2i, i2i, args):
    if args.normalize:
        g = F.normalize(g, dim=-1)
        i = F.normalize(i, dim=-1)

    g2i_cl_loss = cl_with_mask(g, i, g2i, args.T)
    i2g_cl_loss = cl_with_mask(i, g, g2i.t(), args.T)
    i2i_cl_loss = cl_with_mask(i, i, i2i, args.T, mask_diag=True)
    cl_loss = 0.9 * (g2i_cl_loss + i2g_cl_loss + i2i_cl_loss) / 3

    g2i_cdist_loss = cdist_with_mask(g, i, g2i, args.margin)
    i2g_cdist_loss = cdist_with_mask(i, g, g2i.t(), args.margin)
    i2i_cdist_loss = cdist_with_mask(i, i, i2i, args.margin, mask_diag=True)
    cd_loss = 0.1 * (g2i_cdist_loss + i2g_cdist_loss + i2i_cdist_loss) / 3
    return cl_loss + cd_loss


def inter_intra_modality_loss(X, Y, x2y, y2y, args):
    return do_inter_intra_CL(X, Y, x2y, y2y, args)


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)

    CD_loss_1 = cdist_loss(X, Y, args)
    CD_loss_2 = cdist_loss(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2, (CD_loss_1 + CD_loss_2) / 2


def cdist_loss(X, Y, args):
    dist = torch.cdist(X, Y, p=2)
    pos = torch.diag(dist)

    bs = X.size(0)
    mask = torch.eye(bs)
    if torch.cuda.is_available():
        device = X.device
        mask = mask.to(device)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / bs / (bs - 1)
    return loss


def cdist_with_mask(a, b, a2b, margin, mask_diag=False):
    dist = torch.cdist(a, b, p=2)
    pos_position = a2b
    if mask_diag:
        pos_position = pos_position - torch.eye(*a2b.shape, device=a2b.device)
    pos = dist[torch.where(pos_position == 1)]
    denorm = pos_position.sum()

    n_a2b = 1 - a2b
    neg = n_a2b * dist + a2b * margin
    neg = torch.relu(margin - neg)
    loss = pos.sum() / denorm + neg.sum() / n_a2b.sum()
    return loss


def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, args.T)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)

    return CL_loss
