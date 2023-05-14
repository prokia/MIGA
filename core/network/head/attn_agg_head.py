import torch
import torch.nn as nn

from core.network.head import head_register
from utils.tensor_operator import tensor2array


@head_register
class SelfAttentionAggregationHead(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): The embeddings dims.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, cfg):

        super().__init__()
        dim = cfg.dim
        self.dim = dim
        self.scale = cfg.qk_scale or dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=cfg.get('qkv_bias', None))

        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, N, C)
            mask: (0/-inf) mask with shape of (B, N, C) or None
        """
        B_, N, C = x.shape
        qk = self.qk(x).reshape(B_, N, 2, C).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            weight = attn @ mask.float()
            weight = weight.squeeze()
        else:
            weight = torch.sum(attn, dim=-1)

        soft_weight = self.softmax(weight).unsqueeze(1)
        agg_embedding = soft_weight @ x

        return agg_embedding.squeeze()
