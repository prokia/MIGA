import os

import functools

import torch_geometric.data.batch

from core.loss import get_loss
from core.loss.utils import merge_loss_dict, loss_dict_remake_wrapper
from core.network import network_register
from core.network.utils import global_mean_pool_with_mask
from utils.my_containers import deep_dict_get
from utils.pretrain_utils import do_AttrMasking
from utils.tensor_operator import tensor2array, GraphTensorPacking

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from core.network.model import get_model


@network_register.no_cfg_register
class MIGA(nn.Module):
    def __init__(self, name='gin', 
                       molEncoder = None, imgEncoder = None, imgGenerator = None, molGenerator = None, 
                       emb_dim = 256, gic_loss = None, MGM_mode = 'AM', mgm_loss_weight = {'CL_loss': 1, 'mgm_loss': 0},
                       is_eval=False):
        super().__init__()
        self.molEncoder, self.imgEncoder = molEncoder, imgEncoder
        self.imgGenerator, self.molGenerator = imgGenerator, molGenerator
        self.gic_loss = gic_loss
        self.check_params(name, molEncoder, imgEncoder, imgGenerator, molGenerator, gic_loss)

        from utils.distribution import DeviceManager
        self.device = DeviceManager().get_device()
        device = self.device

        # set up model
        self.molecule_gnn_model = get_model(self.molEncoder).to(device)
        self.molecule_readout_func = global_mean_pool

        self.molecule_cnn_model = get_model(self.imgEncoder).to(device)
        self.project_layer1 = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LeakyReLU(),
                                            nn.BatchNorm1d(emb_dim), nn.Linear(emb_dim, emb_dim)).to(
            device)
        self.project_layer2 = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LeakyReLU(),
                                            nn.BatchNorm1d(emb_dim), nn.Linear(emb_dim, emb_dim)).to(
            device)
        self.gim_head = nn.Linear(emb_dim * 2, 2)

        self.molecule_img_generator = get_model(self.imgGenerator).to(device)
        self.molecule_graph_generator = get_model(self.molGenerator).to(device)

        if not is_eval:
            self.gic_loss = get_loss(self.gic_loss)
            self.mgm_mode = MGM_mode
            self.mgm_loss_weight = mgm_loss_weight
            self.init_mgm(emb_dim=emb_dim)

    def check_params(self, name, molEncoder, imgEncoder, imgGenerator, molGenerator, gic_loss):
        if molEncoder is None:
            if name == 'gin':
                self.molEncoder = { 
                    'name': 'GNNEncoder',
                    'gnn_type': 'gin',
                    'graph_pooling': 'mean',
                    'dropout_ratio': 0.2,
                    'num_layer': 5,
                    'emb_dim': 300,
                    'task_type': 'classification'
                }
            elif name == 'graph_transformer':
                self.molEncoder = {
                    'name': 'GraphTransformer',
                    'hidden_mlp_dims': { 'X': 256, 'E': 128},
                    'input_dims': {'X': 60, 'E': 5},
                    # The dimensions should satisfy dx % n_head == 0
                    'hidden_dims': { 'dx': 256, 'de': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128},
                    'n_layers': 3
                }
        if imgEncoder is None:
            self.imgEncoder = {
                'name': 'CnnEncoder',
                'instance_model_name': 'resnet34',
                'target_num': 300,
                'n_ch': 5,
                'pretrained': True,
            }
            if name == 'graph_transformer':
                self.imgEncoder['target_num'] = 256
        if imgGenerator is None:
            self.imgGenerator = {
                'name': 'VariationalAutoEncoder',
                'emb_dim': 300,
                'beta': 1.,
                'loss': 'l2',
            }
        if molGenerator is None:
            self.molGenerator = {
                'name': 'VariationalAutoEncoder',
                'emb_dim': 300,
                'beta': 1.,
                'loss': 'l2',
            }
        if gic_loss is None:
            self.gic_loss = {
                'name': 'PlainDualModalityContrastiveLoss',
                'atom_loss_name_list': ['cdist_loss', 'dualModalityInfoNCE_loss'],
                'cdist_loss': {'margin': 0.4},
                'dualModalityInfoNCE_loss': {'normalize': True, 'temperature': 0.1},
                'atom_loss_weight_dict': {'dualModalityInfoNCE_loss': 0.9, 'cdist_loss': 0.1}
            }

    # @classmethod
    # def from_config(cls, cfg):
    #     model = cls(
    #         molEncoder = cfg.molEncoder,
    #         imgEncoder = cfg.imgEncoder,
    #         imgGenerator = cfg.imgGenerator,
    #         molGenerator = cfg.molGenerator,
    #         emb_dim = cfg.emb_dim,
    #         gic_loss = cfg.gic_loss,
    #         mgm_mode = cfg.get('MGM_mode', None),
    #         mgm_loss_weight = cfg.get('mgm_loss_weight', {'CL_loss': 1, 'mgm_loss': 0})
    #     )
    #     return model

    def init_mgm(self, emb_dim):
        if self.mgm_mode is None:
            self.mgm_mode = []
            return
        if isinstance(self.mgm_mode, str):
            self.mgm_mode = [self.mgm_mode]
        if 'AM' in self.mgm_mode:
            self.molecule_atom_masking_model = torch.nn.Linear(emb_dim, 119)

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
        if not isinstance(graph_info, GraphTensorPacking):
            node_repr = self.molecule_gnn_model(graph_info.x, graph_info.edge_index, graph_info.edge_attr)
            raw_molecule_graph_emb = self.molecule_readout_func(node_repr, graph_info.batch)
            molecule_graph_emb = self.project_layer1(raw_molecule_graph_emb)
        else:
            node_repr = self.molecule_gnn_model(graph_info.X, graph_info.E, data_dict['node_masks'])
            raw_molecule_graph_emb = global_mean_pool_with_mask(node_repr, data_dict['node_masks'])
            molecule_graph_emb = self.project_layer1(raw_molecule_graph_emb)
        return {'embedding': molecule_graph_emb, 'node_repr': node_repr}

    @loss_dict_remake_wrapper
    def forward(self, batch):
        molecule_graph_emb_dict = self.get_graph_embedding(batch)
        molecule_graph_emb = molecule_graph_emb_dict['embedding']

        molecule_img_emb_dict = self.get_img_embedding(batch)
        molecule_img_emb = molecule_img_emb_dict['embedding']

        # To obtain Graph-Image SSL loss and acc
        loss_dict = self.gic_loss(molecule_graph_emb, molecule_img_emb)
        mgm_loss_dict = self.mgm_forward(batch, molecule_graph_emb_dict)
        re_loss_dict = merge_loss_dict({'CL_loss' : loss_dict, 'mgm_loss': mgm_loss_dict},
                                       self.mgm_loss_weight)
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
        merged_loss_dict = {}
        if isinstance(self.mgm_mode, str):
            self.mgm_mode = [self.mgm_mode]
        for m in self.mgm_mode:
            loss_dict = self.__getattribute__(f'{m}_forward')(data_dict, knowledge_dict)
            merged_loss_dict[f'mgm_{m}_loss'] = loss_dict
        return merge_loss_dict(merged_loss_dict)
