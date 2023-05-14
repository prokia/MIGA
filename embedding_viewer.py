import os

# from config import args
import pandas as pd

from miga_emblaze.emblaze import ProjectionTechnique
from utils.distribution import DeviceManager

os.environ['RANK'] = '-1'
DeviceManager('1')
from collections import OrderedDict

import cv2
import umap
from rdkit.Chem import AllChem
from rdkit.Chem import Draw as chem_draw
from torch_geometric.data import Batch

from dataset.utils.loader import mol_to_graph_data_obj_asAtomNum
from utils.config import ConfigConstructor
from utils.emblaze_utils import to_EmbeddingSet, to_Embedding
from utils.geometric_graph import to_dense, fetch_geometric_batch
from utils.tensor_operator import to_device, tensor2array
from utils.utils import custom_load_pretrained_dict
from utils.utils import print_something_first, has_or_init_wrapper

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

import torch

import numpy as np

import miga_emblaze.emblaze as m_emb
from visualizer.visualizers import plt_show

logger = print


class CMEViewer(object):
    def __init__(self, cfg_path):
        self.cfg = ConfigConstructor(cfg_path).get_config()
        self.device = DeviceManager().get_device()
        self.init_model()
        self.modal_name_list = self.cfg.modal.modal_name_list
        self.modal_nums = len(self.cfg.modal.modal_name_list)
        self.samples_num = self.cfg.modal.samples_num
        self.embedding_label_idx_list = [self.modal_nums]
        self.ori_color_array = None
        self.n_neighbors = self.cfg.modal.n_neighbors
        self.viewer_list = []
        self.init_ori_view()

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
            print(f'Loading checkpoint {self.cfg["run"]["resume"]}')
            # print(
            #     "Loading checkpoint '{}' (epoch {})".format(self.cfg["run"]["resume"], checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(self.cfg["run"]["resume"]))

    @print_something_first
    def init_model(self):
        from core.network import get_network
        model = get_network(self.cfg.network)
        model.to(self.device)
        self.model = model
        self.resume()

    def init_img_list(self):
        self.img_list = []
        for m_i, modal_name in enumerate(self.modal_name_list):
            modal_imgs_dir = f'/rhome/lianyu.zhou/dataset/omics_{modal_name}_vis'
            for i in range(self.samples_num):
                read_path = os.path.join(modal_imgs_dir, f"{i:05d}.png")
                img = cv2.imread(read_path)[..., ::-1]
                self.img_list.append(img)

    def update_img_list(self, new_img_list):
        if not isinstance(new_img_list, list):
            new_img_list = [new_img_list]
        self.img_list.extend(new_img_list)

    @print_something_first
    def init_embeddings(self):
        need = self.cfg.modal.modal_name_list
        embeddings_dir = self.cfg.modal.embeddings_dir
        embeddings_list = [np.load(os.path.join(embeddings_dir, f"{name}_embedding.npy"))[:self.cfg.modal.samples_num]
                           for name in need]
        self.embeddings = np.concatenate(embeddings_list, axis=0)
        self.color_list = []
        for m_i, m_name in enumerate(self.modal_name_list):
            self.color_list.extend([m_name] * self.samples_num)

    def init_ori_view(self):
        self.init_embeddings()
        self.init_transformation_from_ori_embeddings()
        self.reduced_embeddings = self.trans.embedding_
        self.embeddingSet = to_EmbeddingSet(self.reduced_embeddings, self.color_list)
        self.embeddingSet.compute_neighbors(metric='euclidean', n_neighbors=self.cfg.modal.n_neighbors)
        self.init_img_list()
        self.construct_ori_dual_frame_view()
        self.construct_view()

    @print_something_first
    def get_new_smiles_embeddings(self, smiles_list):
        self.model.eval()
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        mol_list = []
        mol_img_list = []
        for s_idx, s in enumerate(smiles_list):
            # try:
            rdkit_mol = AllChem.MolFromSmiles(s)
            molecular_graph = mol_to_graph_data_obj_asAtomNum(rdkit_mol, True)
            mol_img = chem_draw.MolToImage(rdkit_mol, size=(224, 224), dpi=600)
            mol_img = mol_img.__array__()
            mol_img = mol_img.copy()
            cv2.putText(mol_img, f"{len(self.reduced_embeddings) + s_idx:05d}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            mol_img_list.append(mol_img)
            mol_list.append(molecular_graph)
        # except:
        #     print("Skip an Invalid mol.")
        geometric_batch = Batch().from_data_list(mol_list)
        matrix_graphs, node_masks = to_dense(*fetch_geometric_batch(geometric_batch, ['edge_attr', 'batch']))
        matrix_graphs, node_masks = [to_device(x, self.device) for x in [matrix_graphs, node_masks]]
        matrix_graphs.to(to_type='float')
        data_dict = {'graphs': matrix_graphs, 'node_masks': node_masks}

        with torch.no_grad():
            molecule_graph_emb = self.model.get_graph_embedding(data_dict)['embedding']
            new_graph_embedding = tensor2array(molecule_graph_emb)
        self.update_img_list(mol_img_list)
        return new_graph_embedding

    def add_new_smiles_embedding(self, smiles):
        new_embeddings = self.get_new_smiles_embeddings(smiles)
        self.update_embeddings(new_embeddings, 'graph')
        self.construct_view()

    def update_embeddings(self, new_embeddings, cls):
        new_reduced_embeddings = self.reduced_embedding_dim(new_embeddings)
        self.update_color(len(new_embeddings), cls)
        self.reduced_embeddings = np.concatenate([self.reduced_embeddings, new_reduced_embeddings], axis=0)
        self.embeddingSet = to_EmbeddingSet(self.reduced_embeddings, self.color_list)
        self.embeddingSet.compute_neighbors(metric='euclidean', n_neighbors=self.n_neighbors)

    def reduced_embedding_dim(self, in_embedding):
        return self.trans.transform(in_embedding)

    @has_or_init_wrapper
    def init_ori_embeddingSet(self):
        ori_reduced_embeddings = self.init_reduced_embeddings()
        self.ori_embeddingSet = to_EmbeddingSet(ori_reduced_embeddings)
        self.ori_embeddingSet.compute_neighbors(metric='euclidean', n_neighbors=self.cfg.modal.ori_n_neighbors)
        return self.ori_embeddingSet

    def update_cur_reduced_embedding(self, new_reduced_embedding):
        self.cur_reduced_embedding = np.concatenate([self.cur_reduced_embedding, new_reduced_embedding], axis=0)
        self.cur_embeddingSet = to_EmbeddingSet(self.cur_reduced_embedding, self.cur_color_array)
        self.ori_embeddingSet.compute_neighbors(metric='euclidean', n_neighbors=self.cfg.modal.ori_n_neighbors)
        return self.ori_embeddingSet

    @print_something_first
    def construct_view(self):
        thum = m_emb.ImageThumbnails(self.img_list) if self.is_show_img() else None
        viewer = m_emb.Viewer(embeddings=self.embeddingSet, thumbnails=thum)
        self.viewer_list.append(viewer)

    def construct_ori_dual_frame_view(self):
        embeddingSet = m_emb.EmbeddingSet([to_Embedding(
            self.embeddings[idx * self.samples_num: (idx + 1) * self.samples_num],
            self.color_list[idx * self.samples_num: (idx + 1) * self.samples_num])
            for idx in range(self.modal_nums)])
        embeddingSet.compute_neighbors(metric='cosine', n_neighbors=self.cfg.modal.n_neighbors)
        reduced_emb, _ = embeddingSet.project(method=ProjectionTechnique.ALIGNED_UMAP,
                                              metric='cosine', n_neighbors=self.cfg.modal.n_neighbors)
        reduced_emb.compute_neighbors(metric='euclidean', n_neighbors=self.cfg.modal.n_neighbors)
        w = m_emb.Viewer(embeddings=reduced_emb)
        self.viewer_list.append(w)

    @print_something_first
    def init_transformation_from_ori_embeddings(self):
        trans = umap.UMAP(metric='cosine', n_neighbors=100).fit(self.embeddings)
        self.trans = trans

    def show_img(self, show=True):
        self.cfg.run["show_img"] = show

    def is_show_img(self):
        return self.cfg.modal.get('show_img', False)

    def init_showing_imgs(self):
        self.img_list = []
        for m_i, modal_name in enumerate(self.modal_name_list):
            modal_imgs_dir = os.path.join(self.cfg.moda.modal_imgs_dir, f"omics_{modal_name}_vis")
            for i in range(self.cfg.modal.samples_num):
                read_path = os.path.join(modal_imgs_dir, f"{i:05d}.png")
                img = cv2.imread(read_path)[..., ::-1]
                self.img_list.append(img)

    def init_color_arr(self):
        self.color_array = np.concatenate([i * np.ones((self.samples_num,))
                                           for i in range(self.modal_nums)],
                                          axis=0)
        return self.color_array

    def update_color(self, add_samples_num, sample_cls):
        self.color_list.extend([sample_cls] * add_samples_num)


if __name__ == '__main__':
    path = '/rhome/lianyu.zhou/dataset/Omics/train.csv'
    smiles_list = []
    with open(path) as f:
        df = pd.read_csv(path)
        df = df.groupby('compound').agg({'smiles': lambda x: x.values[0]}).reset_index()
        for i in range(0, 10):
            smiles_list.append(df.iloc[i]['smiles'])
    config = 'config/miga_vis/embedding_vis_cfg.yaml'
    v = CMEViewer(config)
    v.add_new_smiles_embedding(smiles_list)
    x = 1
