import os

from core.network import network_register, MIGA
from utils.my_containers import deep_dict_get
from ..head import get_head
from ...loss import get_loss
from utils.tensor_operator import tensor2array
from ...loss.utils import merge_loss_dict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

import torch


@network_register
class MilMIGA(MIGA):

    def __init__(self, *args, **kwargs):
        super(MilMIGA, self).__init__(*args, **kwargs)
        self.aggregationHead = get_head(self.cfg.aggregationHead)
        self.i2i_loss = get_loss(self.cfg.i2i_loss)

    def get_bags_img_embedding(self, data_dict):
        need = ['masks']
        masks = deep_dict_get(data_dict, need)
        before_agg_embedding = self.get_img_embedding(data_dict)['embedding']
        b, n, _ = masks.size()
        bags_raw_embedding = before_agg_embedding.view(b, n, -1)
        aggr_embedding = self.aggregationHead(bags_raw_embedding, masks)
        return {'embedding': aggr_embedding, 'raw_embedding': before_agg_embedding}

    def inter_modality_forward(self, data_dict):
        graphs_embedding = self.get_graph_embedding(data_dict)['embedding']
        bags_img_embedding_dict = self.get_bags_img_embedding(data_dict)
        agg_embedding = bags_img_embedding_dict['embedding']
        raw_embedding = bags_img_embedding_dict['raw_embedding']
        b = agg_embedding.size()[0]
        ins_num = raw_embedding.size()[0] // b
        raw_embedding = raw_embedding.view((b, ins_num, -1))
        chosen_embedding = raw_embedding[torch.arange(0, b, device=graphs_embedding.device), data_dict['chosen_idx'].squeeze()]
        gic_loss_dict = self.gic_loss(graphs_embedding, agg_embedding)
        g4i_loss_dict = self.gic_loss(graphs_embedding.detach(), chosen_embedding)
        loss_dict = merge_loss_dict([gic_loss_dict, g4i_loss_dict], loss_prefix_list=['gic_', 'g4i_'])
        return loss_dict

    def intra_modality_forward(self, data_dict):
        need = ['imgs_1', 'imgs_2']
        input_imgs = torch.cat(deep_dict_get(data_dict, need), dim=0)
        img_embedding = self.get_img_embedding({'imgs_ins': input_imgs})['embedding']
        loss_dict = self.i2i_loss(img_embedding)
        return loss_dict

    def forward(self, data_dict, labels=None):
        pass
