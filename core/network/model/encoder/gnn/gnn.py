import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import add_self_loops

from core.network.model import model_register

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr='add', task_type='classification'):
        super(GINConv, self).__init__()
        self.task_type = task_type
        if task_type == 'classification':
            # multi-layer perceptron
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(2 * emb_dim, emb_dim))
            self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
            self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

            torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
            self.aggr = aggr
        else:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                           torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
            self.eps = torch.nn.Parameter(torch.Tensor([0]))

            self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):

        if self.task_type == 'classification':
            # add self loops in the edge space
            edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_index = edge_index[0]
            edge_feature_dims = edge_attr.size()[-1]
            self_loop_attr = torch.zeros(x.size(0), edge_feature_dims, dtype=edge_attr.dtype, device=edge_attr.device)
            self_loop_attr[:, 0] = 4  # bond type for self-loop edge
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

            edge_embeddings = 0
            for d in range(edge_feature_dims):
                edge_embeddings += self.edge_embedding1(edge_attr[:, d])
            return self.propagate(edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
        else:
            edge_embedding = self.bond_encoder(edge_attr)
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
            return out

    def message(self, x_j, edge_attr):
        # return F.relu(x_j + edge_attr)
        if self.task_type == 'classification':
            return x_j + edge_attr
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


@model_register
class GNNEncoder(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, cfg):
        super(GNNEncoder, self).__init__()
        self.cfg = cfg
        self.num_layer = cfg.num_layer
        num_layer = cfg.num_layer
        emb_dim = cfg.emb_dim
        para_key = ['drop_ratio', 'JK', 'gnn_type']
        default_key_val = [0, 'last', 'gin']
        for key, v in zip(para_key, default_key_val):
            custom_v = self.cfg.get(key, v)
            self.__setattr__(key, custom_v)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.cfg.task_type == 'classification':
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if self.gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.cfg.task_type == 'classification':
            x_dims = x.size()[-1]
            emb_x = 0
            for i in range(x_dims):
                emb_x += self.x_embedding1(x[:, i])
            x = emb_x
        else:
            x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            # todo: why atom encoder make this wrong
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNNEncoder(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))
