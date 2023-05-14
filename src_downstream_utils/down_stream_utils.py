import os
from collections import OrderedDict

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

from core.network.utils import global_mean_pool_with_mask
from utils.geometric_graph import ATOM_ENCODER
from utils.utils import custom_load_pretrained_dict
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F


def load_pretrain_cfg(args):
    from utils.config import ConfigConstructor
    pretrain_cfg = ConfigConstructor(args.pretrain_cfg_path)
    pretrain_cfg = pretrain_cfg.get_config()
    pretrain_cfg.run['resume'] = args.pretrain_model_path
    return pretrain_cfg


def resume_pretrain(cfg, model):
    if cfg["run"]["resume"] is None:
        return
    if os.path.isfile(cfg["run"]["resume"]):
        checkpoint = torch.load(cfg["run"]["resume"], map_location='cpu')
        pretrained_dict = checkpoint
        new_pretrained_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = '.'.join(k.split('.')[1:]) if k.startswith('module.') else k  # remove 'module.'
            new_pretrained_state_dict[name] = v
        model_dict = model.state_dict()
        custom_load_pretrained_dict(model_dict, new_pretrained_state_dict)
        model.load_state_dict(model_dict, strict=True)
        print(f'Loading checkpoint {cfg["run"]["resume"]}')
        # self.logger(
        #     "Loading checkpoint '{}' (epoch {})".format(self.cfg["run"]["resume"], checkpoint["epoch"]))
    else:
        print("No checkpoint found at '{}'".format(cfg["run"]["resume"]))


def mol_to_gformer_matrix_data(mol, i=None, removeHS=True):
    from rdkit.Chem.rdchem import BondType as BT
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    if removeHS:
        mol = Chem.RemoveHs(mol)
    N = 0

    type_idx = []
    for atom in mol.GetAtoms():
        atom_Symbol = atom.GetSymbol()
        # if atom_Symbol not in ATOM_ENCODER:
        #     return None
        type_idx.append(ATOM_ENCODER[atom_Symbol])
        N += 1

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()] + 1]

    if len(row) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bonds) + 1), dtype=torch.long)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=60).float()
    y = torch.zeros(size=(1, 0), dtype=torch.float)

    if i is not None:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


def complete_atom_encoder(smiles_list, removeHs=True):
    from utils.geometric_graph import ATOM_ENCODER
    for s in smiles_list:
        mol = AllChem.MolFromSmiles(s)
        if removeHs:
            mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in ATOM_ENCODER:
                ATOM_ENCODER[atom_symbol] = len(ATOM_ENCODER)


class LinearPredictHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearPredictHead, self).__init__()
        self.graph_pred_linear = nn.Linear(input_dim, output_dim)
        self.read_out_func = global_mean_pool_with_mask

    def forward(self, embedding, node_mask):
        read_out = self.read_out_func(embedding, node_mask)
        out = self.graph_pred_linear(read_out)
        return out
