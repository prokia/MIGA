import os

import functools

import torch_geometric.data.batch

from core.loss import get_loss
from core.loss.utils import merge_loss_dict, loss_dict_remake_wrapper
from core.network import network_register
from core.network.utils import global_mean_pool_with_mask
from utils.my_containers import deep_dict_get
from utils.pretrain_utils import do_AttrMasking
from utils.tensor_operator import tensor2array

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from core.network.model import get_model


@network_register
class MIGA(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        from utils.distribution import DeviceManager
        self.device = DeviceManager().get_device()
        device = self.device

        # set up model
        self.molecule_gnn_model = get_model(cfg.molEncoder).to(device)
        self.molecule_readout_func = global_mean_pool

        self.molecule_cnn_model = get_model(self.cfg.imgEncoder).to(device)
        self.project_layer1 = nn.Sequential(nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.LeakyReLU(),
                                            nn.BatchNorm1d(cfg.emb_dim), nn.Linear(cfg.emb_dim, cfg.emb_dim)).to(
            device)
        self.project_layer2 = nn.Sequential(nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.LeakyReLU(),
                                            nn.BatchNorm1d(cfg.emb_dim), nn.Linear(cfg.emb_dim, cfg.emb_dim)).to(
            device)
        self.gim_head = nn.Linear(self.cfg.emb_dim * 2, 2)

        self.molecule_img_generator = get_model(self.cfg.imgGenerator).to(device)
        self.molecule_graph_generator = get_model(self.cfg.molGenerator).to(device)

        self.gic_loss = get_loss(self.cfg.gic_loss)

        self.init_mgm()

    def init_mgm(self):
        self.mgm_mode = self.cfg.get('MGM_mode', None)
        if self.mgm_mode is None:
            self.mgm_mode = []
            return
        if isinstance(self.mgm_mode, str):
            self.mgm_mode = [self.mgm_mode]
        if 'AM' in self.mgm_mode:
            self.molecule_atom_masking_model = torch.nn.Linear(self.cfg.emb_dim, 119)

    def get_img_embedding(self, data_dict):
        need = ['imgs_ins']
        img = deep_dict_get(data_dict, need)
        raw_embedding = self.molecule_cnn_model(img)
        raw_embedding = raw_embedding.float()
        embedding = self.project_layer2(raw_embedding)
        return {'embedding': embedding, 'raw_embedding': raw_embedding}

    def get_graph_embedding(self, data_dict):
        need = ['graphs']
        graph_info = deep_dict_get(data_dict, need)
        if isinstance(graph_info, torch_geometric.data.batch.Batch):
            node_repr = self.molecule_gnn_model(graph_info.x, graph_info.edge_index, graph_info.edge_attr)
            raw_molecule_graph_emb = self.molecule_readout_func(node_repr, graph_info.batch)
            molecule_graph_emb = self.project_layer1(raw_molecule_graph_emb)
        else:
            node_repr = self.molecule_gnn_model(graph_info.X, graph_info.E, data_dict['node_masks'])
            raw_molecule_graph_emb = global_mean_pool_with_mask(node_repr, data_dict['node_masks'])
            molecule_graph_emb = self.project_layer1(raw_molecule_graph_emb)
        return {'embedding': molecule_graph_emb, 'raw_embedding': molecule_graph_emb, 'node_repr': node_repr}

    @loss_dict_remake_wrapper
    def forward(self, batch):
        molecule_graph_emb_dict = self.get_graph_embedding(batch)
        molecule_graph_emb = molecule_graph_emb_dict['embedding']

        molecule_img_emb_dict = self.get_img_embedding(batch)
        molecule_img_emb = molecule_img_emb_dict['embedding']

        # To obtain Graph-Image SSL loss and acc
        loss_dict = self.gic_loss(molecule_graph_emb, molecule_img_emb)
        mgm_loss_dict = self.mgm_forward(batch, molecule_graph_emb_dict)
        re_loss_dict = merge_loss_dict([loss_dict, mgm_loss_dict], self.cfg.get('mgm_loss_weight', None))
        return re_loss_dict

    def AM_forward(self, data_dict, knowledge_dict):
        need = ['graphs']
        batch1 = deep_dict_get(data_dict, need)
        criterion = nn.CrossEntropyLoss()
        MGM_loss, _ = do_AttrMasking(
            batch=batch1, criterion=criterion, node_repr=knowledge_dict['node_repr'],
            molecule_atom_masking_model=self.molecule_atom_masking_model)
        return {'total_loss': MGM_loss, 'am_loss': tensor2array(MGM_loss)}

    def mgm_forward(self, data_dict, knowledge_dict):
        if self.mgm_mode is None:
            return {}
        loss_dict_list = []
        if isinstance(self.mgm_mode, str):
            self.mgm_mode = [self.mgm_mode]
        for m in self.mgm_mode:
            loss_dict = self.__getattribute__(f'{m}_forward')(data_dict, knowledge_dict)
            loss_dict_list.append(loss_dict)
        return merge_loss_dict(loss_dict_list)
