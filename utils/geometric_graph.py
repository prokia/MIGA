import torch
import torch_geometric
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import dense_to_sparse, to_dense_batch, to_dense_adj

from utils.tensor_operator import tensor2array, PlaceHolder

from utils.tensor_operator import make_one_hot

# ATOM_ENCODER = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11,
#                 'Na': 13, 'Ca': 14, 'Zn': 15, 'As': 16, 'K': 17, 'Cu': 18, 'Mg': 19, 'Hg': 20, 'Cr': 21, 'Zr': 22,
#                 'Sn': 23, 'Ba': 24, 'Au': 25, 'Pd': 26, 'Tl': 27, 'Fe': 28, 'Al': 29, 'Gd': 30, 'Ag': 31, 'Mo': 32,
#                 'V': 33, 'Nd': 34, 'Co': 35, 'Yb': 36, 'Pb': 37, 'Sb': 38, 'In': 39, 'Li': 40, 'Ni': 41, 'Bi': 42, 'Cd': 43,
#                 'Ti': 44, 'Dy': 45, 'Mn': 46, 'Sr': 47, 'Be': 48, 'Pt': 49, 'Ge': 50}
ATOM_ENCODER = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
ATOM_DECODER = list(ATOM_ENCODER)


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def fetch_geometric_batch(g, extra_needs=None):
    res = [g.x, g.edge_index]
    if extra_needs is None:
        return res
    for need in extra_needs:
        res.append(g.__getattribute__(need))
    return res


def to_dense(x, edge_index, edge_attr, batch, holder=True):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    if holder:
        return PlaceHolder(X=X, E=E, y=None), node_mask
    else:
        return X, E, node_mask


def matrix2geometricData(atom_matrix, edge_matrix, node_mask=None, make_one_hot_edge=False, edge_cls=5):
    if node_mask is not None:
        atom_matrix[~node_mask] = 0
    atom_idx = torch.nonzero(atom_matrix, as_tuple=True)
    atom = atom_matrix[atom_idx]
    valid_e = edge_matrix[atom_idx]
    valid_e = torch.index_select(valid_e, index=atom_idx[0], dim=1)
    edge_index, edge_attr = dense_to_sparse(valid_e)
    if make_one_hot_edge:
        edge_attr = make_one_hot(edge_attr.unsqueeze(-1), edge_cls)
    else:
        edge_attr = edge_attr.unsqueeze(-1)
    data = PyGData(x=atom.unsqueeze(-1), edge_index=edge_index, edge_attr=edge_attr)
    return atom_idx[0], data


def mask(self, node_mask, collapse=False):
    x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
    e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
    e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

    if collapse:
        self.X = torch.argmax(self.X, dim=-1)
        self.E = torch.argmax(self.E, dim=-1)

        self.X[node_mask == 0] = - 1
        self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
    else:
        self.X = self.X * x_mask
        self.E = self.E * e_mask1 * e_mask2
        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
    return self
