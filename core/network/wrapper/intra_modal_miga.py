import os

from core.loss.contrastive_loss import inter_intra_modality_loss
from core.network import network_register, MIGA

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pretrain_utils import do_AttrMasking


@network_register
class IntraModalMIGA(MIGA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch, labels=None):

        cfg = self.cfg
        need = ['graphs', 'input_imgs', 'i2i', 'g2i']
        batch1, batch2, i2i, g2i = list(map(batch.get, need))
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)
        # alpha = 0.4

        molecule_graph_emb_dict = self.get_graph_embedding(batch1)
        molecule_graph_emb = molecule_graph_emb_dict['embedding']

        molecule_img_emb_dict = self.get_img_embedding(batch2)
        molecule_img_emb = molecule_img_emb_dict['embedding']

        ##### To obtain Graph-Image SSL loss and acc
        GIC_loss = inter_intra_modality_loss(molecule_graph_emb, molecule_img_emb, g2i, i2i, cfg.loss_cfg)  # INFO or EBM

        # img_generator_loss = self.molecule_img_generator(molecule_graph_emb, molecule_img_emb)
        # todo: What I concern is that
        #  an ideal representation not contain some  features of the image unrelated to graph
        #  but essential for the image reconstruction

        # graph_generator_loss = self.molecule_graph_generator(molecule_img_emb, molecule_graph_emb)
        # GIM_loss = (img_generator_loss + graph_generator_loss) / 2
        # GIM_loss = graph_generator_loss

        # Graph-Image Matching Loss
        # forward the positve image-text pair
        # output_pos = torch.cat([molecule_graph_emb, molecule_img_emb], dim=1)
        #
        # with torch.no_grad():
        #     bs = molecule_graph_emb.size(0)
        #     weights_g2i = F.softmax(torch.cdist(molecule_graph_emb, molecule_img_emb, p=2))
        #     weights_i2g = F.softmax(torch.cdist(molecule_img_emb, molecule_graph_emb, p=2))
        #     weights_i2g.fill_diagonal_(0)
        #     weights_g2i.fill_diagonal_(0)
        #
        # # select a negative image for each text
        # molecule_img_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_g2i[b], 1).item()
        #     molecule_img_embeds_neg.append(molecule_img_emb[neg_idx])
        # molecule_img_embeds_neg = torch.stack(molecule_img_embeds_neg, dim=0)
        #
        # # select a negative text for each image
        # molecule_graph_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2g[b], 1).item()
        #     molecule_graph_embeds_neg.append(molecule_graph_emb[neg_idx])
        # molecule_graph_embeds_neg = torch.stack(molecule_graph_embeds_neg, dim=0)
        #
        # molecule_graph_embeds_all = torch.cat([molecule_graph_emb, molecule_graph_embeds_neg], dim=0)
        # molecule_img_embeds_all = torch.cat([molecule_img_emb, molecule_img_embeds_neg], dim=0)
        #
        # output_neg = torch.cat([molecule_graph_embeds_all, molecule_img_embeds_all], dim=1)
        #
        # vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        # vl_output = self.gim_head(vl_embeddings)
        #
        # gim_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #                        dim=0).to(molecule_graph_emb.device)
        # GGIM_loss = GIM_loss + F.cross_entropy(vl_output, gim_labels)
        # #
        # criterion = nn.CrossEntropyLoss()
        # molecule_atom_masking_model = self.MGM_models[0]
        # masked_node_repr = self.molecule_gnn_model(batch1.masked_x, batch1.edge_index, batch1.edge_attr)
        # MGM_loss, _ = do_AttrMasking(
        #     batch=batch1, criterion=criterion, node_repr=masked_node_repr,
        #     molecule_atom_masking_model=molecule_atom_masking_model)
        MGM_loss = 0
        GGIM_loss = 0
        return GIC_loss, GGIM_loss, MGM_loss
