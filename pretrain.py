import time, os

from dataset.utils.mol_transform import MaskAtom, ExtractSubstructureContextPair

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import global_mean_pool

from tqdm import tqdm
import numpy as np

from config import args
from models import GNN, Cnn, VariationalAutoEncoder, Discriminator
from utils.utils import load_data_omics, calc_contrastive_loss

from utils.pretrain_utils import do_AttrMasking, do_ContextPred, do_InfoGraph
from dataloader import DataLoader, DataLoaderMasking, DataLoaderSubstructContext, DataLoaderMaskingGraph


class MIGA(nn.Module):

    def __init__(self, args, device):
        super().__init__()

        self.args = args
        self.device = device

        # set up model
        self.molecule_gnn_model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                                      gnn_type=args.gnn_type).to(device)
        self.molecule_readout_func = global_mean_pool

        self.molecule_cnn_model = Cnn(model_name=args.cnn_model_name, target_num=args.emb_dim,
                                      n_ch=3 if args.cDNA else 5, pretrained=True).to(device)
        self.project_layer1 = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.LeakyReLU(),
                                            nn.BatchNorm1d(args.emb_dim), nn.Linear(args.emb_dim, args.emb_dim)).to(
            device)
        self.project_layer2 = nn.Sequential(nn.Linear(args.emb_dim, args.emb_dim), nn.LeakyReLU(),
                                            nn.BatchNorm1d(args.emb_dim), nn.Linear(args.emb_dim, args.emb_dim)).to(
            device)
        self.gim_head = nn.Linear(self.args.emb_dim * 2, 2)

        self.molecule_img_generator = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.generator_loss, detach_target=args.detach_target,
            beta=args.beta).to(device)
        self.molecule_graph_generator = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.generator_loss, detach_target=args.detach_target,
            beta=args.beta).to(device)

        # set up Graph SSL model
        self.MGM_models = []

        # set up Graph SSL model for IG
        infograph_discriminator_SSL_model = None
        if args.MGM_mode == 'IG':
            infograph_discriminator_SSL_model = Discriminator(args.emb_dim).to(device)
            self.MGM_models.append(infograph_discriminator_SSL_model)

        # set up Graph SSL model for AM
        molecule_atom_masking_model = None
        if args.MGM_mode == 'AM':
            molecule_atom_masking_model = torch.nn.Linear(args.emb_dim, 119).to(device)
            self.MGM_models.append(molecule_atom_masking_model)

        # set up Graph SSL model for CP
        molecule_context_model = None
        if args.MGM_mode == 'CP':
            l1 = args.num_layer - 1
            l2 = l1 + args.csize
            molecule_context_model = GNN(int(l2 - l1), args.emb_dim, JK=args.JK,
                                         drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
            self.MGM_models.append(molecule_context_model)

        if args.MGM_mode == 'MGM':
            molecule_atom_masking_model = torch.nn.Linear(args.emb_dim, 119).to(device)
            self.MGM_models.append(molecule_atom_masking_model)

            l1 = args.num_layer - 1
            l2 = l1 + args.csize
            molecule_context_model = GNN(int(l2 - l1), args.emb_dim, JK=args.JK,
                                         drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
            self.MGM_models.append(molecule_context_model)

    def forward(self, batch):

        args = self.args
        batch1, batch2 = batch
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)
        # alpha = 0.4

        node_repr = self.molecule_gnn_model(batch1.x, batch1.edge_index, batch1.edge_attr)
        molecule_graph_emb = self.molecule_readout_func(node_repr, batch1.batch)
        molecule_img_emb = self.molecule_cnn_model(batch2)

        molecule_graph_emb = self.project_layer1(molecule_graph_emb)
        molecule_img_emb = self.project_layer2(molecule_img_emb)

        ##### To obtain Graph-Image SSL loss and acc
        GIC_loss = calc_contrastive_loss(molecule_graph_emb, molecule_img_emb, args)  # INFO or EBM

        img_generator_loss = self.molecule_img_generator(molecule_graph_emb, molecule_img_emb)
        graph_generator_loss = self.molecule_graph_generator(molecule_img_emb, molecule_graph_emb)
        GIM_loss = (img_generator_loss + graph_generator_loss) / 2

        # Graph-Image Matching Loss
        # forward the positve image-text pair
        output_pos = torch.cat([molecule_graph_emb, molecule_img_emb], dim=1)

        with torch.no_grad():
            bs = molecule_graph_emb.size(0)
            weights_g2i = F.softmax(torch.cdist(molecule_graph_emb, molecule_img_emb, p=2))
            weights_i2g = F.softmax(torch.cdist(molecule_img_emb, molecule_graph_emb, p=2))
            weights_i2g.fill_diagonal_(0)
            weights_g2i.fill_diagonal_(0)

        # select a negative image for each text
        molecule_img_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_g2i[b], 1).item()
            molecule_img_embeds_neg.append(molecule_img_emb[neg_idx])
        molecule_img_embeds_neg = torch.stack(molecule_img_embeds_neg, dim=0)

        # select a negative text for each image
        molecule_graph_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2g[b], 1).item()
            molecule_graph_embeds_neg.append(molecule_graph_emb[neg_idx])
        molecule_graph_embeds_neg = torch.stack(molecule_graph_embeds_neg, dim=0)

        molecule_graph_embeds_all = torch.cat([molecule_graph_emb, molecule_graph_embeds_neg], dim=0)
        molecule_img_embeds_all = torch.cat([molecule_img_emb, molecule_img_embeds_neg], dim=0)

        output_neg = torch.cat([molecule_graph_embeds_all, molecule_img_embeds_all], dim=1)

        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        vl_output = self.gim_head(vl_embeddings)

        gim_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(molecule_graph_emb.device)
        GGIM_loss = GIM_loss + F.cross_entropy(vl_output, gim_labels)

        if args.MGM_mode == 'IG':
            criterion = nn.BCEWithLogitsLoss()
            infograph_discriminator_SSL_model = self.MGM_models[0]
            MGM_loss, _ = do_InfoGraph(
                node_repr=node_repr, batch=batch1,
                molecule_repr=molecule_graph_emb, criterion=criterion,
                infograph_discriminator_SSL_model=infograph_discriminator_SSL_model)

        elif args.MGM_mode == 'AM':
            criterion = nn.CrossEntropyLoss()
            molecule_atom_masking_model = self.MGM_models[0]
            masked_node_repr = self.molecule_gnn_model(batch1.masked_x, batch1.edge_index, batch1.edge_attr)
            MGM_loss, _ = do_AttrMasking(
                batch=batch1, criterion=criterion, node_repr=masked_node_repr,
                molecule_atom_masking_model=molecule_atom_masking_model)

        elif args.MGM_mode == 'CP':
            criterion = nn.BCEWithLogitsLoss()
            molecule_context_model = self.MGM_models[0]
            MGM_loss, _ = do_ContextPred(
                batch=batch1, criterion=criterion, args=args,
                molecule_substruct_model=self.molecule_gnn_model,
                molecule_context_model=molecule_context_model,
                molecule_readout_func=self.molecule_readout_func
            )

        elif args.MGM_mode == 'MGM':

            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.BCEWithLogitsLoss()
            criterion = [criterion1, criterion2]

            molecule_atom_masking_model = self.MGM_models[0]
            masked_node_repr = self.molecule_gnn_model(batch1.masked_x, batch1.edge_index, batch1.edge_attr)
            MGM_loss1, _ = do_AttrMasking(
                batch=batch1, criterion=criterion[0], node_repr=masked_node_repr,
                molecule_atom_masking_model=molecule_atom_masking_model)

            molecule_context_model = self.MGM_models[1]
            MGM_loss2, _ = do_ContextPred(
                batch=batch1, criterion=criterion[1], args=args,
                molecule_substruct_model=self.molecule_gnn_model,
                molecule_context_model=molecule_context_model,
                molecule_readout_func=self.molecule_readout_func,
                molecule_img_repr=molecule_img_emb
            )

            MGM_loss = MGM_loss1 + MGM_loss2
        else:
            raise Exception

        return GIC_loss, GGIM_loss, MGM_loss


def train(model, dataset, optimizer, epoch, device, args):
    print('========== Use {} for Graph SSL =========='.format(args.MGM_mode))
    transform = None
    if args.MGM_mode == 'AM':
        transform = MaskAtom(num_atom_type=119, num_edge_type=5,
                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        loader = DataLoaderMasking(dataset, transform=transform, batch_size=args.batch_size, drop_last=True,
                                   shuffle=True, num_workers=args.num_workers)

    elif args.MGM_mode == 'IG':

        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True, collate_fn=dataset.collate_fn, num_workers=16)

    elif args.MGM_mode == 'CP':
        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform = ExtractSubstructureContextPair(args.num_layer, l1, l2)

        loader = DataLoaderSubstructContext(dataset, transform=transform, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    elif args.MGM_mode == 'MGM':
        transform1 = MaskAtom(num_atom_type=119, num_edge_type=5,
                              mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform2 = ExtractSubstructureContextPair(args.num_layer, l1, l2)

        loader = DataLoaderMaskingGraph(dataset, transforms=[transform1, transform2], batch_size=args.batch_size,
                                        shuffle=True, drop_last=True, num_workers=args.num_workers)


    else:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True, collate_fn=dataset.collate_fn, num_workers=16)

    molecule_gnn_model = model.molecule_gnn_model
    molecule_cnn_model = model.molecule_cnn_model
    molecule_gnn_model.train()
    molecule_cnn_model.train()
    project_layer1 = model.project_layer1
    project_layer1.train()
    project_layer2 = model.project_layer2
    project_layer2.train()
    molecule_img_generator = model.molecule_img_generator
    molecule_img_generator.train()
    molecule_graph_generator = model.molecule_graph_generator
    molecule_graph_generator.train()
    MGM_models = model.MGM_models
    for support_model in MGM_models:
        support_model.train()

    GIC_loss_all, GGIM_loss_all, MGM_loss_all = 0, 0, 0

    all_rankings = []
    start_time = time.time()
    for step, batch in enumerate(tqdm(loader)):
        molecule_gnn_model.eval()
        molecule_cnn_model.eval()
        project_layer1.eval()
        project_layer2.eval()
        with torch.no_grad():
            batch1, batch2 = batch
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            # alpha = 0.4

            node_repr = molecule_gnn_model(batch1.x, batch1.edge_index, batch1.edge_attr)
            molecule_graph_emb = global_mean_pool(node_repr, batch1.batch)
            molecule_img_emb = molecule_cnn_model(batch2)

            molecule_graph_emb = project_layer1(molecule_graph_emb)
            molecule_img_emb = project_layer2(molecule_img_emb)

            batch_size = molecule_graph_emb.size(0)
            ground_truth = torch.unsqueeze(torch.arange(0, batch_size), dim=1)
            if torch.cuda.is_available():
                ground_truth = ground_truth.to(device)
            # print('train mol',molecule_graph_emb)
            # print('train img',molecule_img_emb)
            dist = torch.cdist(molecule_graph_emb, molecule_img_emb, p=2)
            # print(dist)
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        molecule_gnn_model.train()
        molecule_cnn_model.train()
        project_layer1.train()
        project_layer2.train()
        GIC_loss, GGIM_loss, MGM_loss = model(batch)

        GIC_loss_all += GIC_loss.detach().cpu().item()
        GGIM_loss_all += GGIM_loss.detach().cpu().item()
        MGM_loss_all += MGM_loss.detach().cpu().item()

        loss = 0
        if args.alpha_1 > 0:
            loss += GIC_loss * args.alpha_1
        if args.alpha_2 > 0:
            loss += GGIM_loss * args.alpha_2
        if args.alpha_3 > 0:
            loss += MGM_loss * args.alpha_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(all_rankings)
    # calculate metrics
    all_rankings = np.array(all_rankings)
    mrr = float(np.mean(1 / all_rankings))
    mr = float(np.mean(all_rankings))
    h1 = float(np.mean(all_rankings <= 1))
    h3 = float(np.mean(all_rankings <= 3))
    h5 = float(np.mean(all_rankings <= 5))
    h10 = float(np.mean(all_rankings <= 10))
    h100 = float(np.mean(all_rankings <= 100))
    print('========== Training Process: Epoch {} =========='.format(epoch))
    print('mrr: %.4f mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f  h100: %.4f' % (mrr, mr, h1, h3, h5, h10, h100))

    # global optimal_loss
    GIC_loss_all /= len(loader)
    GGIM_loss_all /= len(loader)
    MGM_loss_all /= len(loader)

    print(
        'GIC Loss: {:.5f}\tGGIM Loss: {:.5f}\tMGM Loss: {:.5f}\tTime: {:.5f}'.format(
            GIC_loss_all, GGIM_loss_all, MGM_loss_all,
            time.time() - start_time))
    return molecule_gnn_model, molecule_cnn_model, project_layer1, project_layer1


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def evaluate_image(model, molecule_readout_func, dataset, device, args):
    molecule_gnn_model = model.molecule_gnn_model
    molecule_cnn_model = model.molecule_cnn_model
    project_layer1 = model.project_layer1
    project_layer2 = model.project_layer2
    print('========== Evaluate {} for Graph SSL =========='.format(args.MGM_mode))
    transform = None
    if args.MGM_mode == 'AM':
        transform = MaskAtom(num_atom_type=119, num_edge_type=5,
                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        loader = DataLoaderMasking(dataset, transform=transform, batch_size=args.batch_size, shuffle=False,
                                   drop_last=False,
                                   num_workers=args.num_workers)

    elif args.MGM_mode == 'IG':
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn, num_workers=16)

    elif args.MGM_mode == 'CP':
        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform = ExtractSubstructureContextPair(args.num_layer, l1, l2)

        loader = DataLoaderSubstructContext(dataset, transform=transform, batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    elif args.MGM_mode == 'MGM':
        transform1 = MaskAtom(num_atom_type=119, num_edge_type=5,
                              mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform2 = ExtractSubstructureContextPair(args.num_layer, l1, l2)

        loader = DataLoaderMaskingGraph(dataset, transforms=[transform1, transform2], batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers)

    else:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn, num_workers=args.num_workers)

    molecule_gnn_model.eval()
    molecule_cnn_model.eval()
    project_layer1.eval()
    project_layer2.eval()
    with torch.no_grad():
        # calculate embeddings of all cell image as the candidate pool
        all_image_embeddings = []
        for step, batch in enumerate(tqdm(loader)):
            _, batch2 = batch
            batch2 = batch2.to(device)
            batch_tta = []
            for I in range(8):
                molecule_img_emb = molecule_cnn_model(get_trans(batch2, I))
                molecule_img_emb = project_layer2(molecule_img_emb)
                batch_tta.append(molecule_img_emb)
            molecule_img_emb = torch.stack(batch_tta, 0).mean(0)

            all_image_embeddings.append(molecule_img_emb)
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        # print(all_image_embeddings.shape)
        # rank
        all_rankings = []
        i = 0
        for step, batch in enumerate(tqdm(loader)):
            batch1, batch2 = batch
            batch1 = batch1.to(device)
            node_repr = molecule_gnn_model(batch1.x, batch1.edge_index, batch1.edge_attr)
            molecule_graph_emb = molecule_readout_func(node_repr, batch1.batch)
            molecule_graph_emb = project_layer1(molecule_graph_emb)

            batch_size = molecule_graph_emb.size(0)
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + batch_size, len(dataset))), dim=1)
            if torch.cuda.is_available():
                ground_truth = ground_truth.to(device)

            dist = torch.cdist(molecule_graph_emb, all_image_embeddings, p=2).cpu().numpy()

            # print(dist.shape)
            choose_dist = np.zeros((dist.shape[0], 101))
            #
            for di in range(dist.shape[0]):
                idxs = np.arange(dist.shape[1]).tolist()
                idxs = idxs[:i + di] + idxs[i + di + 1:]
                indices = np.random.choice(idxs, 101, replace=False)
                indices[0] = i + di
                # print(indices)
                choose_dist[di] = dist[di][indices]
            # print(i+di)
            sorted_indices = np.argsort(choose_dist, axis=1)
            # print(sorted_indices)
            rankings = ((sorted_indices == 0).nonzero()[1] + 1).tolist()
            all_rankings.extend(rankings)
            i += batch_size

        # print('eval mol',molecule_graph_emb)
        # print('eval img',all_image_embeddings)
        # print('eval',dist)
        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mrr_std = float(np.std(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        mr_std = float(np.std(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))
        h100 = float(np.mean(all_rankings <= 100))

        gt = np.zeros((dist.shape[0], 101))
        gt[:, 0] = 1
        auc = roc_auc_score(gt.reshape(-1), -choose_dist.reshape(-1))
        text = 'mrr:  %.4f+-%.4f  mr: %.4f+-%.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f h100: %.4f auc: %.4f' % (
        mrr, mrr_std, mr, mr_std, h1, h3, h5, h10, h100, auc)
        print(text)
        return mr, h10, text


def evaluate_graph(model, molecule_readout_func, dataset, device, args):
    molecule_gnn_model = model.molecule_gnn_model
    molecule_cnn_model = model.molecule_cnn_model
    project_layer1 = model.project_layer1
    project_layer2 = model.project_layer2
    print('========== Evaluate {} for Graph SSL =========='.format(args.MGM_mode))
    transform = None
    if args.MGM_mode == 'AM':
        transform = MaskAtom(num_atom_type=119, num_edge_type=5,
                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        loader = DataLoaderMasking(dataset, transform=transform, batch_size=args.batch_size, shuffle=False,
                                   drop_last=False,
                                   num_workers=args.num_workers)

    elif args.MGM_mode == 'IG':
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn, num_workers=16)

    elif args.MGM_mode == 'CP':
        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform = ExtractSubstructureContextPair(args.num_layer, l1, l2)

        loader = DataLoaderSubstructContext(dataset, transform=transform, batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    elif args.MGM_mode == 'MGM':
        transform1 = MaskAtom(num_atom_type=119, num_edge_type=5,
                              mask_rate=args.mask_rate, mask_edge=args.mask_edge)

        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform2 = ExtractSubstructureContextPair(args.num_layer, l1, l2)

        loader = DataLoaderMaskingGraph(dataset, transforms=[transform1, transform2], batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers)

    else:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn, num_workers=args.num_workers)

    molecule_gnn_model.eval()
    molecule_cnn_model.eval()
    project_layer1.eval()
    project_layer2.eval()
    with torch.no_grad():
        all_graph_embeddings = []
        for step, batch in enumerate(tqdm(loader)):
            batch1, batch2 = batch
            batch1 = batch1.to(device)
            node_repr = molecule_gnn_model(batch1.x, batch1.edge_index, batch1.edge_attr)
            molecule_graph_emb = molecule_readout_func(node_repr, batch1.batch)
            molecule_graph_emb = project_layer1(molecule_graph_emb)
            all_graph_embeddings.append(molecule_graph_emb)
        all_graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
        # rank
        all_rankings = []
        i = 0
        for step, batch in enumerate(tqdm(loader)):
            _, batch2 = batch
            batch2 = batch2.to(device)
            batch_tta = []
            for I in range(8):
                molecule_img_emb = molecule_cnn_model(get_trans(batch2, I))
                molecule_img_emb = project_layer2(molecule_img_emb)
                batch_tta.append(molecule_img_emb)
            molecule_img_emb = torch.stack(batch_tta, 0).mean(0)

            batch_size = molecule_img_emb.shape[0]
            dist = torch.cdist(molecule_img_emb, all_graph_embeddings, p=2).cpu().numpy()

            # print(dist.shape)
            choose_dist = np.zeros((dist.shape[0], 101))
            #
            for di in range(dist.shape[0]):
                idxs = np.arange(dist.shape[1]).tolist()
                idxs = idxs[:i + di] + idxs[i + di + 1:]
                indices = np.random.choice(idxs, 101, replace=False)
                indices[0] = i + di
                # print(indices)
                choose_dist[di] = dist[di][indices]
            # print(i+di)
            sorted_indices = np.argsort(choose_dist, axis=1)
            # print(sorted_indices)
            rankings = ((sorted_indices == 0).nonzero()[1] + 1).tolist()
            all_rankings.extend(rankings)
            i += batch_size
        # print('eval mol',molecule_img_emb)
        # print('eval img',all_graph_embeddings)
        # print('eval',dist)
        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mrr_std = float(np.std(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        mr_std = float(np.std(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))
        h100 = float(np.mean(all_rankings <= 100))

        gt = np.zeros((dist.shape[0], 101))
        gt[:, 0] = 1
        auc = roc_auc_score(gt.reshape(-1), -choose_dist.reshape(-1))
        text = 'mrr:  %.4f+-%.4f  mr: %.4f+-%.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f h100: %.4f auc: %.4f' % (
        mrr, mrr_std, mr, mr_std, h1, h3, h5, h10, h100, auc)
        print(text)
        return mr, h10, text


def main():
    if not os.path.exists(args.output_model_dir):
        os.mkdir(args.output_model_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # set up dataset
    dataset, test_dataset = load_data_omics(args)
    model = MIGA(args, device)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device])

    # set up parameters
    model_param_group = [{'params': model.molecule_gnn_model.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},
                         {'params': model.molecule_cnn_model.parameters(),
                          'lr': args.lr * args.cnn_lr_scale},
                         {'params': model.project_layer1.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},
                         {'params': model.project_layer2.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},
                         {'params': model.molecule_img_generator.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},
                         {'params': model.molecule_graph_generator.parameters(),
                          'lr': args.lr * args.cnn_lr_scale}
                         ]
    for MGM_model in model.MGM_models:
        model_param_group.append({'params': MGM_model.parameters(),
                                  'lr': args.lr * args.gnn_lr_scale})

    # set up optimizer
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    print("====Start training")
    if args.normalize:
        print("====normalize")
    # trainer.train(dataset)
    best_mr = 1000
    best_h10 = -1
    if args.retrieval == 'image':
        evaluate = evaluate_image
    else:
        evaluate = evaluate_graph
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        # temp_loss, optimal_loss,molecule_gnn_model,molecule_cnn_model, project_layer = model(dataset)
        train(model, dataset, optimizer, epoch, device, args)

        mr, h10, text = evaluate(model, global_mean_pool, test_dataset, device, args)
        if mr < best_mr and h10 > best_h10:
            best_mr = mr
            best_h10 = h10
            best_text = text
            print('best metric:', best_text)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), args.output_model_dir + "/model_" + str(epoch) + ".pth")
            torch.save(model.molecule_gnn_model.state_dict(),
                       args.output_model_dir + "/model_gnn_" + str(epoch) + ".pth")

    print('best metric:', best_text)


if __name__ == "__main__":
    main()
