from abc import abstractmethod

import torch.nn as nn

from core.loss.base import atom_loss_register
from core.loss import loss_register
from core.loss.utils import loss_dict_remake_wrapper, merge_loss_dict
from functools import partial


class AtomLossCombinator(nn.Module):
    def __init__(self, cfg):
        super(AtomLossCombinator, self).__init__()
        self.cfg = cfg
        self.atom_loss_dict = {l_n: partial(atom_loss_register[l_n], **self.cfg[l_n])
                               for l_n in self.cfg.atom_loss_name_list}

    @loss_dict_remake_wrapper
    def forward(self, *args, **kwargs):
        loss_dict_list = []
        for atom_loss_name, atom_loss in self.atom_loss_dict.items():
            atom_loss_value = atom_loss(*args, **kwargs)
            loss_dict_list.append({atom_loss_name: atom_loss_value, 'total_loss': atom_loss_value})
        return merge_loss_dict(loss_dict_list, self.cfg.get('atom_loss_weight_dict', None))

