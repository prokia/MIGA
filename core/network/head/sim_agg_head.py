import torch
import torch.nn as nn
import torch.nn.functional as F

from core.network.head import head_register
from utils.tensor_operator import tensor2array


@head_register
class SimAggregationHead(nn.Module):
    """
    Args:
        dim (int): The embeddings dims.
  """

    def __init__(self, cfg):

        super().__init__()
        self.dim = cfg.dim

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, N, C)
            mask: (0/-inf) mask with shape of (B, N, C) or None
        """
        x_norm = F.normalize(x, dim=-1).detach()
        sim = x_norm @ x_norm.transpose(-2, -1)

        if mask is not None:
            weight = sim @ mask.float()
            weight += - (1 - mask) * 999
            weight = weight.squeeze()
        else:
            weight = torch.sum(sim, dim=-1)

        soft_weight = self.softmax(weight).unsqueeze(1)
        agg_embedding = soft_weight @ x

        return agg_embedding.squeeze()
