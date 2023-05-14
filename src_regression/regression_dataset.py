import os
import pickle
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)

from src_downstream_utils.down_stream_utils import mol_to_gformer_matrix_data


def mol_to_graph_data_obj_reg(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def graph_data_obj_to_nx_reg(data):
    """ torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        temp_feature = atom_features[i]
        G.add_node(
            i,
            x0=temp_feature[0],
            x1=temp_feature[1],
            x2=temp_feature[2],
            x3=temp_feature[3],
            x4=temp_feature[4],
            x5=temp_feature[5],
            x6=temp_feature[6],
            x7=temp_feature[7],
            x8=temp_feature[8])
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        temp_feature= edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       e0=temp_feature[0],
                       e1=temp_feature[1],
                       e2=temp_feature[2])

    return G


def nx_to_graph_data_obj_reg(G):
    """ vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered. """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['x0'], node['x1'], node['x2'], node['x3'], node['x4'], node['x5'], node['x6'], node['x7'], node['x8']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['e0'], edge['e1'], edge['e2']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def create_standardized_mol_id(smiles):
    """ smiles -> inchi """

    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)\
            # c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
    return


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively """

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one """

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]



class MoleculeDatasetRegression(InMemoryDataset):
    def __init__(self, root, dataset='esol', transform=None,
                 pre_transform=None, pre_filter=None, empty=False):

        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(MoleculeDatasetRegression, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        if self.dataset == 'davis':
            file_name_list = ['davis']
        elif self.dataset == 'kiba':
            file_name_list = ['kiba']
        else:
            file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):

        def shared_extractor(smiles_list, rdkit_mol_objs, labels):
            data_list = []
            data_smiles_list = []
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=1)
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is None:
                    continue
                # data = mol_to_graph_data_obj_reg(rdkit_mol)
                data = mol_to_gformer_matrix_data(rdkit_mol)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

            return data_list, data_smiles_list


        if self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'lipo':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # For ESOL, there are molecules with single atoms and empty edges.
        valid_index = []
        neo_data_smiles_list, neo_data_list = [], []
        for i, (smiles, data) in enumerate(zip(data_smiles_list, data_list)):
            if data.edge_attr.size()[0] == 0:
                print('Invalid\t', smiles, data)
                continue
            valid_index.append(i)
            assert data.edge_attr.size()[1] == 3
            assert data.edge_index.size()[0] == 2
            assert data.x.size()[1] == 9
            assert data.id.size()[0] == 1
            assert data.y.size()[0] == 1
            neo_data_smiles_list.append(smiles)
            neo_data_list.append(data)

        old_N = len(data_smiles_list)
        neo_N = len(valid_index)
        print('{} invalid out of {}.'.format(old_N - neo_N, old_N))
        print(len(neo_data_smiles_list), '\t', len(neo_data_list))

        data_smiles_series = pd.Series(neo_data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(neo_data_list)
        torch.save((data, slices), self.processed_paths[0])

        return


def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    print('input_path', input_path)
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    print('input_path', input_path)

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

