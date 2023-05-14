import asyncio
import os
import time
from collections import defaultdict, OrderedDict

from core.runner import runner_register
from dataset.utils.load_data import load_data_omics
from utils.my_containers import AverageMeter, deep_dict_get
from utils.tensor_operator import to_device, tensor2array
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from utils.utils import dict2str, custom_load_pretrained_dict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np

# from config import args
from utils.distribution import sync_barrier_re1

from dataset.utils.mol_transform import MaskAtom
from dataset.dataloader import DataLoaderMasking, DataLoaderNoiseMatrix
from utils.loggers import LoggerManager

logger = LoggerManager().get_logger()


@runner_register
class PretrainRunner(object):
    def __init__(self, cfg, writer):
        self.cfg = cfg
        self.writer = writer
        self.train_dataset, self.test_dataset = load_data_omics(cfg)
        self.train_loader = self.init_loader('train')
        self.test_loader = self.init_loader('test')
        self.dataset_name = self.cfg.dataset
        from utils.distribution import DeviceManager
        self.device = DeviceManager().get_device()
        self.loss_name_list = []
        self.n_step_global = 0
        self.logger = logger
        self.epochs = 0
        # Setup loss function

    def write_loss(self, dataset_name):
        out = '>> '
        for loss_name in self.loss_name_list:
            prefix_name = f'{dataset_name}_{loss_name}'
            meter = self.__getattribute__(f'{prefix_name}_meter')
            loss_scalar = np.round(meter.avg, 4)
            self.tb_add_scalar(prefix_name, loss_scalar, self.epochs)
            out += f' {prefix_name}: {loss_scalar}'
            # self.csv_logger[prefix_name].append(loss_scalar)
            meter.reset()
        self.logger(out)

    @sync_barrier_re1
    def tb_add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)

    def init_loader(self, dataset_name):
        # set up dataset
        dataset = self.__getattribute__(f'{dataset_name}_dataset')
        args = self.cfg
        transform = MaskAtom(num_atom_type=119, num_edge_type=5,
                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)
        drop_last, shuffle = [True if 'train' in dataset_name else False] * 2
        if self.cfg.network.molEncoder.name == 'GNNEncoder':
            loader = DataLoaderMasking(dataset, transform=transform, batch_size=args.batch_size, drop_last=drop_last,
                                       shuffle=shuffle, num_workers=args.num_workers)
        else:
            loader = DataLoaderNoiseMatrix(dataset, transforms=None, batch_size=args.batch_size, drop_last=drop_last,
                                           shuffle=shuffle, num_workers=args.num_workers)
        return loader

    def get_embedding(self, data_dict):
        data_dict = to_device(data_dict, self.device)
        graph_embedding = self.model.get_graph_embedding(data_dict)['embedding']
        imgs_embedding = self.model.get_img_embedding(data_dict)['embedding']
        return graph_embedding, imgs_embedding

    def train_eval(self, data_dict):
        all_rankings = []
        self.network.eval()
        with torch.no_grad():
            # alpha = 0.4
            molecule_graph_emb, molecule_img_emb = self.get_embedding(data_dict)
            batch_size = molecule_graph_emb.size(0)
            ground_truth = torch.unsqueeze(torch.arange(0, batch_size), dim=1)
            if torch.cuda.is_available():
                ground_truth = to_device(ground_truth, self.device)
            # logger('train mol',molecule_graph_emb)
            # logger('train img',molecule_img_emb)
            dist = torch.cdist(molecule_graph_emb, molecule_img_emb, p=2)
            # logger(dist)
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)
        # logger(all_rankings)
        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))
        h100 = float(np.mean(all_rankings <= 100))
        logger('========== Training Process: Epoch {} =========='.format(epoch))
        logger('mrr: %.4f mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f  h100: %.4f' % (
            mrr, mr, h1, h3, h5, h10, h100))

    def update_loss(self, model_out_dict):
        loss_dict = {k: v for k, v in model_out_dict.items() if '_loss' in k}
        loss_dict = {loss_name: tensor2array(loss_value) for loss_name, loss_value in loss_dict.items()}
        # if self.is_distribution_running():
        #     gather_loss_dict = [None] * self.cfg.dist.world_size
        #     dist.all_gather_object(gather_loss_dict, loss_dict)
        #     loss_dict = value_dict_mean(gather_loss_dict)
        for loss_name, loss_value in loss_dict.items():
            try:
                self.__getattribute__(f'{self.dataset_name}_{loss_name}_meter').update(tensor2array(loss_value))
            except:
                self.__setattr__(f'{self.dataset_name}_{loss_name}_meter', AverageMeter())
                self.__getattribute__(f'{self.dataset_name}_{loss_name}_meter').update(tensor2array(loss_value))
                self.loss_name_list.append(loss_name)

    def fp16_optimizer_update(self):
        fp16 = self.cfg.get('fp16', False)
        if fp16:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def train_step(self, data_dict):
        self.model.train()
        batch = to_device(data_dict, self.device)
        self.optimizer.zero_grad()
        fp16 = self.cfg.get('fp16', False)
        with autocast(fp16):
            model_out_dict = self.model(batch)
            self.fp16_backward(model_out_dict['total_loss'])
        if fp16:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.update_loss(model_out_dict)

    def fp16_backward(self, loss):
        if self.cfg.get('fp16', False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def train(self, epoch):
        args = self.cfg
        logger('========== Use {} for Graph SSL =========='.format(args.MGM_mode))

        for step, batch in enumerate(tqdm(self.train_loader)):
            self.train_step(batch)
            self.n_step_global += 1
        self.write_loss(self.dataset_name)
        self.epochs += 1
        # return molecule_gnn_model, molecule_cnn_model, project_layer1, project_layer1

    def get_trans(self, img, I):
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

    def get_tta_batch_img_embedding(self, batch):
        batch = to_device(batch, self.device)
        batch_tta = []
        img = batch['imgs_ins'].clone()
        for i in range(8):
            batch['imgs_ins'] = self.get_trans(img, i)
            molecule_img_emb = self.model.get_img_embedding(batch)
            batch_tta.append(molecule_img_emb['embedding'])
        molecule_img_emb = torch.stack(batch_tta, 0).mean(0)
        return molecule_img_emb

    def get_tta_img_embedding(self):
        all_image_embeddings = []
        for step, batch in enumerate(tqdm(self.test_loader)):
            batch = to_device(batch, self.device)
            molecule_img_emb = self.get_tta_batch_img_embedding(batch)
            all_image_embeddings.append(molecule_img_emb)
        return torch.cat(all_image_embeddings, dim=0)

    def evaluate_image(self):
        args = self.cfg
        dataset = self.test_dataset
        device = self.device
        self.model.eval()

        logger('========== Evaluate {} for Graph SSL =========='.format(args.MGM_mode))

        with torch.no_grad():
            # calculate embeddings of all cell image as the candidate pool
            all_image_embeddings = self.get_tta_img_embedding()
            # logger(all_image_embeddings.shape)
            # rank
            all_rankings = []
            i = 0
            for step, batch in enumerate(tqdm(self.test_loader)):
                batch = to_device(batch, self.device)
                molecule_graph_emb = self.model.get_graph_embedding(batch)['embedding']

                batch_size = molecule_graph_emb.size(0)
                ground_truth = torch.unsqueeze(torch.arange(i, min(i + batch_size, len(dataset))), dim=1)
                if torch.cuda.is_available():
                    ground_truth = ground_truth.to(device)

                dist = torch.cdist(molecule_graph_emb, all_image_embeddings, p=2).cpu().numpy()

                # logger(dist.shape)
                choose_dist = np.zeros((dist.shape[0], 101))
                #
                for di in range(dist.shape[0]):
                    idxs = np.arange(dist.shape[1]).tolist()
                    idxs = idxs[:i + di] + idxs[i + di + 1:]
                    indices = np.random.choice(idxs, 101, replace=False)
                    indices[0] = i + di
                    # logger(indices)
                    choose_dist[di] = dist[di][indices]
                # logger(i+di)
                sorted_indices = np.argsort(choose_dist, axis=1)
                # logger(sorted_indices)
                rankings = ((sorted_indices == 0).nonzero()[1] + 1).tolist()
                all_rankings.extend(rankings)
                i += batch_size

            # logger('eval mol',molecule_graph_emb)
            # logger('eval img',all_image_embeddings)
            # logger('eval',dist)
            # calculate metrics
            return self.process_all_rankings(all_rankings, choose_dist)

    def process_all_rankings(self, all_rankings, choose_dist):
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mrr_std = float(np.std(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        mr_std = float(np.std(all_rankings))
        pm = u'\u00B1'
        print_dict = {'mrr': f'{mrr:.4f}{pm}{mrr_std:.4f}', 'mr': f'{mr:.4f}{pm}{mr_std:.4f}'}
        metric_dict = {'mr': mr, 'mrr': mrr}
        h_dict = defaultdict(lambda: 0.)
        print_h_dict = {}
        for r in [1, 3, 5, 10, 100]:
            h = float(np.mean(all_rankings <= r))
            h_dict[f'h{r}'] = h
            print_h_dict[f'h{r}'] = f'{h:.4f}'
        metric_dict.update(h_dict)
        print_dict.update(print_h_dict)
        gt = np.zeros((choose_dist.shape[0], 101))
        gt[:, 0] = 1
        auc = roc_auc_score(gt.reshape(-1), -choose_dist.reshape(-1))
        metric_dict['auc'] = auc
        print_dict['auc'] = f'{auc:.4f}'
        self.write_eval(print_dict, metric_dict)
        return print_dict, metric_dict

    def resume(self):
        if os.path.isfile(self.cfg["run"]["resume"]):
            checkpoint = torch.load(self.cfg["run"]["resume"], map_location='cpu')
            pretrained_dict = checkpoint
            new_pretrained_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = '.'.join(k.split('.')[1:]) if k.startswith('module.') else k  # remove 'module.'
                new_pretrained_state_dict[name] = v
            model_dict = self.model.state_dict()
            custom_load_pretrained_dict(model_dict, new_pretrained_state_dict)
            self.model.load_state_dict(model_dict, strict=True)
            self.logger(f'Loading checkpoint {self.cfg["run"]["resume"]}')
            # self.logger(
            #     "Loading checkpoint '{}' (epoch {})".format(self.cfg["run"]["resume"], checkpoint["epoch"]))
        else:
            self.logger("No checkpoint found at '{}'".format(self.cfg["run"]["resume"]))

    def evaluate_graph(self):
        args = self.cfg
        loader = self.test_loader

        logger('========== Evaluate {} for Graph SSL =========='.format(args.MGM_mode))
        self.model.eval()
        with torch.no_grad():
            all_graph_embeddings = []
            for step, batch in enumerate(tqdm(loader)):
                molecule_graph_emb = self.model.get_graph_embedding(batch)['embedding']
                all_graph_embeddings.append(molecule_graph_emb)
            all_graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
            # rank
            all_rankings = []
            i = 0
            for step, batch in enumerate(tqdm(loader)):
                molecule_img_emb = self.model.get_tta_batch_img_embedding(batch)['embedding']

                batch_size = molecule_img_emb.shape[0]
                dist = torch.cdist(molecule_img_emb, all_graph_embeddings, p=2).cpu().numpy()

                # logger(dist.shape)
                choose_dist = np.zeros((dist.shape[0], 101))
                #
                for di in range(dist.shape[0]):
                    idxs = np.arange(dist.shape[1]).tolist()
                    idxs = idxs[:i + di] + idxs[i + di + 1:]
                    indices = np.random.choice(idxs, 101, replace=False)
                    indices[0] = i + di
                    # logger(indices)
                    choose_dist[di] = dist[di][indices]
                # logger(i+di)
                sorted_indices = np.argsort(choose_dist, axis=1)
                # logger(sorted_indices)
                rankings = ((sorted_indices == 0).nonzero()[1] + 1).tolist()
                all_rankings.extend(rankings)
                i += batch_size
            # logger('eval mol',molecule_img_emb)
            # logger('eval img',all_graph_embeddings)
            # logger('eval',dist)
            # calculate metrics

            return self.process_all_rankings(all_rankings, choose_dist)

    def embedding_analysis(self):
        loader = self.test_loader

        self.model.eval()
        with torch.no_grad():
            dis_list = []
            modal_name_list = ['graph', 'img']
            modality_embedding_list = [[] for _ in modal_name_list]
            for step, batch in enumerate(tqdm(loader)):
                graph_and_img_embeddings = self.get_embedding(batch)
                batch_dis = F.cosine_similarity(*graph_and_img_embeddings, dim=1)
                dis_list.extend(list(tensor2array(batch_dis)))
                for e_idx, e in enumerate(graph_and_img_embeddings):
                    modality_embedding_list[e_idx].append(tensor2array(e))
            dis_list = sorted(dis_list)
            # fig = plt.figure(f'similarity histogram')
            # ax = fig.add_subplot(1, 1, 1)
            # ax.hist(self.dis_list, 20)
            plt.hist(dis_list, 40)
            embedding_save_dir = self.cfg.run.get('embedding_save_dir', None)
            plt.savefig(os.path.join(embedding_save_dir, "distance distribution.png"))
            for embedding_cls_idx, embeddings_list in enumerate(modality_embedding_list):
                embedding_array = np.concatenate(embeddings_list, axis=0)
                save_name = f"{modal_name_list[embedding_cls_idx]}_embedding.npy"
                save_path = os.path.join(embedding_save_dir, save_name)
                np.save(save_path, embedding_array)

    def embedding_vis(self):
        embedding_dir = self.cfg.run.get('embedding_dir', None)
        if embedding_dir is not None:
            need = ['img', 'graph']
            import numpy as np
            import os
            import miga_emblaze.emblaze as m_emb
            from miga_emblaze.emblaze.utils import Field, ProjectionTechnique
            import cv2

            embedding_dir = '/rhome/lianyu.zhou/cache/cluster_test_set_gformer_l3_b256_atom60'
            mol_img_dir = '/rhome/lianyu.zhou/dataset/omics_mol_vis'
            need = ['img', 'graph']
            modal_num = len(need)
            samples_num = 300
            embeddings_list = [np.load(os.path.join(embedding_dir, f"{name}_embedding.npy"))[:samples_num]
                               for name in need]

            modal_imgs_list = [[] for _ in range(modal_num)]
            for m_i, modal_name in enumerate(need):
                modal_imgs_dir = f'/rhome/lianyu.zhou/dataset/omics_{modal_name}_vis'
                for i in range(samples_num):
                    read_path = os.path.join(modal_imgs_dir, f"{i:05d}.png")
                    img = cv2.imread(read_path)[..., ::-1]
                    modal_imgs_list[m_i].append(img)

            import umap
            color_list = [i * np.ones(e.shape[0]) for i, e in enumerate(embeddings_list)]
            color_array, merge_embedding = list(map(lambda x: np.concatenate(x, axis=0), [color_list, embeddings_list]))
            trans = umap.UMAP(metric='cosine', n_neighbors=100).fit(merge_embedding)
            reduced_embedding = trans.embedding_
            view_embedding = m_emb.EmbeddingSet(
                [m_emb.Embedding({Field.POSITION: reduced_embedding, Field.COLOR: color_array})])
            view_embedding.compute_neighbors(metric='euclidean', n_neighbors=100)

            from utils.utils import add_new_embedding

            new_reduced, new_thum = add_new_embedding([i for i in range(samples_num, samples_num + 5)],
                                                      0,
                                                      trans,
                                                      reduced_embedding,
                                                      color_array,
                                                      modal_imgs_list[0] + modal_imgs_list[1])

            # Represent the high-dimensional embedding
            color = np.ones((len(embeddings_list[0]),))
            embeddings = m_emb.EmbeddingSet([
                m_emb.Embedding({Field.POSITION: X, Field.COLOR: color}, label=emb_name)
                for X, emb_name in zip(embeddings_list, need)
            ])
            # Compute nearest neighbors in the high-D space (for display)
            embeddings.compute_neighbors(metric='cosine', n_neighbors=5)

            # Make aligned UMAP
            reduced, trans = embeddings.project(method=ProjectionTechnique.ALIGNED_UMAP, metric='cosine')
            reduced_emb_array_list = [e.field(Field.POSITION) for e in reduced.embeddings]

            new_embeddings_list = [np.load(os.path.join(embedding_dir, f"{name}_embedding.npy"))[300: 600]
                                              for name in need]
            g = new_embeddings_list[0][:1]
            reduced_g = trans.mappers_[0].transform(g)
            reduced_emb_array_list[0] = np.concatenate([reduced_emb_array_list[0], reduced_g], axis=0)
            new_reduced_embedding = m_emb.EmbeddingSet([
                m_emb.Embedding({Field.POSITION: X, Field.COLOR: i * np.ones(X.shape[0])}, label=emb_name)
                for i, (X, emb_name) in enumerate(zip(reduced_emb_array_list, need))
            ])
            new_reduced_embedding.compute_neighbors(metric='euclidean', n_neighbors=10)
            new_w = m_emb.Viewer(embeddings=new_reduced_embedding)
            w = m_emb.Viewer(embeddings=reduced)
            x = 1

    def main(self):
        args = self.cfg
        from core.network import get_network
        model = get_network(args.network)
        model.to(self.device)
        self.model = model
        if self.cfg.run.get('resume', None) is not None:
            self.resume()

        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device])

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
        # for MGM_model in model.MGM_models:
        #     model_param_group.append({'params': MGM_model.parameters(),
        #                               'lr': args.lr * args.gnn_lr_scale})

        # set up optimizer
        # if self.cfg.get('optimizer', 'adam') == 'adam':
        self.optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        # else:
        #     self.optimizer = optim.AdamW(model_param_group, lr=args.lr, weight_decay=args.decay)
        self.grad_scaler = GradScaler() if self.cfg.run.get('FP16', False) else None
        logger("====Start training")
        if args.normalize:
            logger("====normalize")
        # trainer.train(dataset)
        best_mr = 1000
        best_h10 = -1
        if args.retrieval == 'image':
            evaluate = self.evaluate_image
        else:
            evaluate = self.evaluate_graph
        for epoch in range(1, args.epochs + 1):
            logger("====epoch " + str(epoch))

            # temp_loss, optimal_loss,molecule_gnn_model,molecule_cnn_model, project_layer = model(dataset)
            run_type = self.cfg.run.get('run_type', 'train')
            if run_type == 'train':
                self.train(epoch)
            else:
                self.__getattribute__(run_type)()


            print_dict, eval_dict = evaluate()
            need = ['mr', 'h10']
            mr, h10 = deep_dict_get(eval_dict, need)
            if mr < best_mr and h10 > best_h10:
                best_mr = mr
                best_h10 = h10
                best_text = dict2str(print_dict)
                logger(f'best metric: {best_text}')
            if epoch in [0, 1] or epoch % 5 == 0:
                model_path = os.path.join(self.cfg.weight_dir, "model_" + str(epoch) + ".pth")
                self.torch_save(model.state_dict(), model_path)
                gnn_model_path = os.path.join(self.cfg.weight_dir, "model_gnn" + str(epoch) + ".pth")
                self.torch_save(model.molecule_gnn_model.state_dict(), gnn_model_path)
            # temp_loss, optimal_loss,molecule_gnn_model,molecule_cnn_model, project_layer = model(dataset)
            if self.cfg.run.get('run_type', 'train') != 'train':
                break

        logger(best_text)

    @sync_barrier_re1
    def write_eval(self, print_dict, metric_dict):
        self.logger(dict2str(print_dict))
        for e_n, e_v in metric_dict.items():
            writed_name = e_n
            item_value = e_v
            self.tb_add_scalar(writed_name, item_value, self.epochs)

    # def update_monitor(self, metric_dict):
    #     is_best = False
    #     for e_n, e_v in metric_dict:
    #         writed_name = e_n
    #         item_value = e_v
    #         if f'best_{e_n}' in self.monitor:
    #             if e_v > self.monitor[f'best_{eval_name}']:
    #                 is_best = True
    #     eval_item_dict = defaultdict(lambda: 0)
    #     for dataset_name, dataset_eval_info in eval_info.items():
    #         for eval_name, eval_value in dataset_eval_info.items():
    #             eval_item_dict[eval_name] += eval_value
    #     for eval_name, eval_value in eval_item_dict.items():
    #         eval_value /= len(eval_info)
    #         self.csv_logger[f'mean_{eval_name}'].append(eval_value)
    #         if f'best_{eval_name}' in self.monitor:
    #             if eval_value > self.monitor[f'best_{eval_name}']:
    #                 is_best = True
    #                 self.monitor[f'best_{eval_name}'] = eval_value
    #             self.csv_logger[f'best_{eval_name}'].append(self.monitor[f'best_{eval_name}'])
    #     return is_best

    @sync_barrier_re1
    def torch_save(self, state, path):
        saving_work = self.asycn_torch_save(state, path)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(saving_work)

    @staticmethod
    async def asycn_torch_save(state, path):
        torch.save(state, path)

    def __call__(self):
        self.main()
