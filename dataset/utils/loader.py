import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from torch_geometric.data import Data
import torch.nn.functional as F

# from dgl.batch import batch as batch_graphs


# allowable node and edge features
from utils.geometric_graph import ATOM_ENCODER

weight_dict = {'H': '1.01', 'He': '4.0', 'Li': '6.94', 'Be': '9.01', 'B': '10.81', 'C': '12.01', 'N': '14.01',
               'O': '16.0', 'F': '19.0', 'Ne': '20.18', 'Na': '22.99', 'Mg': '24.31', 'Al': '26.98', 'Si': '28.09',
               'P': '30.97', 'S': '32.07', 'Cl': '35.45', 'Ar': '39.95', 'K': '39.1', 'Ca': '40.08', 'Sc': '44.96',
               'Ti': '47.87', 'V': '50.94', 'Cr': '52.0', 'Mn': '54.94', 'Fe': '55.85', 'Co': '58.93', 'Ni': '58.69',
               'Cu': '63.55', 'Zn': '65.39', 'Ga': '69.72', 'Ge': '72.64', 'As': '74.92', 'Se': '78.96', 'Br': '79.9',
               'Kr': '83.8', 'Rb': '85.47', 'Sr': '87.62', 'Y': '88.91', 'Zr': '91.22', 'Nb': '92.91', 'Mo': '95.94',
               'Tc': '97.91', 'Ru': '101.07', 'Rh': '102.91', 'Pd': '106.42', 'Ag': '107.87', 'Cd': '112.41',
               'In': '114.82', 'Sn': '118.71', 'Sb': '121.76', 'Te': '127.6', 'I': '126.9', 'Xe': '131.29',
               'Cs': '132.91', 'Ba': '137.33', 'La': '138.91', 'Ce': '140.12', 'Pr': '140.91', 'Nd': '144.24',
               'Pm': '145.0', 'Sm': '150.36', 'Eu': '151.96', 'Gd': '157.25', 'Tb': '158.93', 'Dy': '162.5',
               'Ho': '164.93', 'Er': '167.26', 'Tm': '168.93', 'Yb': '173.04', 'Lu': '174.97', 'Hf': '178.49',
               'Ta': '180.95', 'W': '183.84', 'Re': '186.21', 'Os': '190.23', 'Ir': '192.22', 'Pt': '195.08',
               'Au': '196.97', 'Hg': '200.59', 'Tl': '204.38', 'Pb': '207.2', 'Bi': '208.98', 'Po': '208.98',
               'At': '209.99', 'Rn': '222.02', 'Fr': '223.0', 'Ra': '226.0', 'Ac': '227.0', 'Th': '232.04',
               'Pa': '231.04', 'U': '238.03', 'Np': '238.85', 'Pu': '242.88', 'Am': '244.86', 'Cm': '246.91',
               'Bk': '248.93', 'Cf': '252.96', 'Es': '253.97', 'Fm': '259.0', 'Md': '260.01', 'No': '261.02',
               'Lr': '264.04', 'Rf': '269.08', 'Db': '270.09', 'Sg': '273.11', 'Bh': '274.12', 'Hs': '272.11',
               'Mt': '278.15', 'Ds': '283.19', 'Rg': '282.18', 'Cn': '287.22', 'Uut': '286.22', 'Fl': '291.2',
               'Uup': '290.19', 'Lv': '295.23', 'Uus': '293.21', 'Uuo': '299.26'}

atom_cls = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
atom_mapping = dict(zip(atom_cls, [i for i in range(0, len(atom_cls))]))

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'atom_mapping': atom_mapping
}


def mol_to_graph_data_obj_simple(mol, make_one_hot=False):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def mol_to_graph_data_obj_asAtomNum(mol, make_one_hot=False):
    """
    Converts a molecule object to a graph data object.
     Parameters:
    mol (Mol): The molecule object to convert.
    make_one_hot (bool): Whether to make the output one-hot encoded.
     Returns:
    Data: The graph data object.
    """
    # atoms
    # Atom features
    num_atom_features = 1  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [ATOM_ENCODER[atom.GetSymbol()]]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    # bonds
    num_bond_features = 1  # bond type
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            # Get the indices of the atoms at the beginning and end of the bond
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Get the index of the bond type in the allowable_features list
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())]
            # Add the indices to the edges list
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            # Add the reverse indices to the edges list
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    if make_one_hot:
        edge_attr = F.one_hot(edge_attr + 1, num_classes=5).squeeze()
        x = F.one_hot(x, num_classes=60).squeeze()
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
