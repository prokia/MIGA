from utils.my_containers import Register

head_register = Register()
from .attn_agg_head import SelfAttentionAggregationHead
from .sim_agg_head import SimAggregationHead
from .max_agg_head import MaxAggregationHead


def get_head(head_cfg):
    return head_register[head_cfg.name](head_cfg)
