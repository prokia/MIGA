import numpy as np
import torch
from torch_geometric.data import Batch

from . import dataset_register


@dataset_register
class NewSmilesOmicsDataset:
    def __init__(self, cfg, mode, raw_graphs=None, transform=None):
        self.cfg = cfg
        self.mode = mode
        self.raw_graphs = raw_graphs
        self.path = '../data/' + self.cfg.dataset + '/cache2/' + self.mode
        self.reactant_graphs = []
        self.img_graphs = []
        self.transform = transform

    def __len__(self):
        return int(len(self.raw_graphs) * self.cfg.get('effective_percent', 1.0))

    def __getitem__(self, i):
        reactant_graph = self.raw_graphs[i][0]
        # img = np.load(self.raw_graphs[i][1][0])
        img = np.load(np.random.choice(self.raw_graphs[i][1]))
        if self.cfg.cDNA:
            img = img[[2, 0, 1], :, :]
        if self.transform is not None:
            img = self.transform(image=img.transpose(1, 2, 0))['image'] / 15
        re_dict = {'graphs': reactant_graph, 'imgs_ins': img.unsqueeze(0)}
        return re_dict

    def collate_fn(self, items):
        batched_graphs = Batch.from_data_list([item[0] for item in items])
        imgs = torch.stack([item[1] for item in items], 0)
        return batched_graphs, imgs

