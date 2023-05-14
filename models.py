
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.nn.inits import glorot, zeros
import timm

from config import args

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        if args.task_type == 'classification':
            #multi-layer perceptron
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
            self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
            self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

            torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
            self.aggr = aggr
        else:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
            self.eps = torch.nn.Parameter(torch.Tensor([0]))

            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
            

    def forward(self, x, edge_index, edge_attr):

        if args.task_type == 'classification':
            #add self loops in the edge space
            edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
            edge_index = edge_index[0]
            self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
            return self.propagate(edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
        else:
            edge_embedding = self.bond_encoder(edge_attr)
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
            return out

    def message(self, x_j, edge_attr):
        # return F.relu(x_j + edge_attr)
        if args.task_type == 'classification':
            return x_j + edge_attr
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class Cnn(nn.Module):
    def __init__(self, model_name, target_num, n_ch=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained)
        # print(self.model)
        if ('efficientnet' in model_name) or ('mixnet' in model_name):
            self.model.conv_stem.weight = nn.Parameter(self.model.conv_stem.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif model_name in ['resnet34d']:
            self.model.conv1[0].weight = nn.Parameter(self.model.conv1[0].weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif ('resnet' in model_name or 'resnest' in model_name) and 'vit' not in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif 'rexnet' in model_name or 'regnety' in model_name or 'nf_regnet' in model_name:
            self.model.stem.conv.weight = nn.Parameter(self.model.stem.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif 'resnext' in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif 'hrnet_w32' in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif 'densenet' in model_name:
            self.model.features.conv0.weight = nn.Parameter(self.model.features.conv0.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif 'ese_vovnet39b' in model_name or 'xception41' in model_name:
            self.model.stem[0].conv.weight = nn.Parameter(self.model.stem[0].conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif 'dpn' in model_name:
            self.model.features.conv1_1.conv.weight = nn.Parameter(self.model.features.conv1_1.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_channels, target_num)
            self.model.classifier = nn.Identity()
        elif 'inception' in model_name:
            self.model.features[0].conv.weight = nn.Parameter(self.model.features[0].conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.last_linear.in_features, target_num)
            self.model.last_linear = nn.Identity()
        elif 'vit' in model_name:
            self.model.patch_embed.proj.weight = nn.Parameter(self.model.patch_embed.proj.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        elif 'vit_base_resnet50' in model_name:
            self.model.patch_embed.backbone.stem.conv.weight = nn.Parameter(self.model.patch_embed.backbone.stem.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        else:
            raise

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def extract(self, x):
        return self.model(x)

    def forward(self, x):
        x = self.extract(x)
        # for i, dropout in enumerate(self.dropouts):
        #     if i == 0:
        #         h = self.myfc(dropout(x))
        #     else:
        #         h += self.myfc(dropout(x))
        return self.myfc(x)


class GNN(torch.nn.Module):
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
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if args.task_type == 'classification':
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.atom_encoder = AtomEncoder(emb_dim)


        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
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

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if args.task_type == 'classification':
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

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
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
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



class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim=1)


def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


class AutoEncoder(torch.nn.Module):

    def __init__(self, emb_dim, loss, detach_target):
        super(AutoEncoder, self).__init__()
        self.loss = loss
        self.emb_dim = emb_dim
        self.detach_target = detach_target

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.fc_layers = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        # why not have BN as last FC layer:
        # https://stats.stackexchange.com/questions/361700/
        return

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()
        x = self.fc_layers(x)
        loss = self.criterion(x, y)

        return loss


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + self.beta * kl_loss

        return loss



if __name__ == "__main__":
    pass
