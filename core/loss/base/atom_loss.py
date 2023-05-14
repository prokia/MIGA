import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tensor_operator import to_device, tensor2array, force_fp32
from . import atom_loss_register
from ..utils import symmetric_loss_wrapper


@atom_loss_register
@symmetric_loss_wrapper
@force_fp32()
def cdist_loss(X, Y, margin):
    dist = torch.cdist(X, Y, p=2)
    pos = torch.diag(dist)

    bs = X.size(0)
    mask = torch.eye(bs, device=X.device)
    neg = (1 - mask) * dist + mask * margin
    neg = torch.relu(margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / bs / (bs - 1)
    return loss


@atom_loss_register
@symmetric_loss_wrapper
def dualModalityInfoNCE_loss(X, Y, temperature, normalize=False):
    if normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, temperature)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)

    return CL_loss


@atom_loss_register
def infoNCE_loss(features, temperature):
    bn, dim = features.size()
    batch_size = bn / 2
    n_views = 2
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = to_device(labels, features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.t())
    # assert similarity_matrix.shape == (
    #     n_views * batch_size, n_views * batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)

    logits = logits / temperature

    logits = F.cross_entropy(logits, labels)
    return logits
