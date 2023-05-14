import torch
import torch.nn as nn
import torch.nn.functional as F

from core.network.head import head_register
from utils.tensor_operator import tensor2array


@head_register
class MaxAggregationHead(nn.Module):
    """
    Args:
        dim (int): The embeddings dims.
  """

    def __init__(self, cfg):

        super().__init__()
        self.dim = cfg.dim

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, N, C)
            mask: (0/1) mask with shape of (B, N, 1) or None
        """
        x_norm = F.normalize(x, dim=-1).detach()
        sim = x_norm @ x_norm.transpose(-2, -1)

        if mask is not None:
            weight = sim @ mask.float()
            weight += - (1 - mask) * 999
            weight = weight.squeeze()
        else:
            weight = torch.sum(sim, dim=-1)

        chosen_embedding_idx = torch.argmax(weight, dim=-1)
        b, _ = weight.size()
        agg_embedding = x[torch.arange(b), chosen_embedding_idx]

        return agg_embedding
