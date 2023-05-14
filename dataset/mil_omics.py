import random

import numpy as np
import torch

from . import dataset_register
from .omics import NewSmilesOmicsDataset
from .utils.auto_img_transformations import AutoImgTransformationManager as ATF


@dataset_register
class MILSmilesOmicsDataset(NewSmilesOmicsDataset):
    def __init__(self, *args, **kwargs):
        super(MILSmilesOmicsDataset, self).__init__(*args, **kwargs)
        self.base_trans = ATF(self.cfg.base_transformation)()
        aug_transformations_setting = self.cfg.get('aug_transformation', None)
        if aug_transformations_setting is not None:
            self.aug_trans = ATF(aug_transformations_setting)()

    def get_img_ins(self, i, img_idx_list):
        imgs = []
        max_ins = len(self.raw_graphs[i][1])
        masks = []
        for idx in img_idx_list:
            if idx >= max_ins:
                # todo: currently use a hard code
                img = torch.zeros((5, 128, 128))
                masks.append(0)
            else:
                img = np.load(self.raw_graphs[i][1][idx])
                img = self.transform(image=img.transpose(1, 2, 0))['image'] / 15
                masks.append(1)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        chosen_idx = random.randint(0, len(masks) - 1)
        while masks[chosen_idx] == 0:
            chosen_idx = random.randint(0, len(masks) - 1)
        masks = torch.as_tensor(masks)
        masks = masks.unsqueeze(-1)
        return imgs, masks, torch.as_tensor([chosen_idx])

    def get_img_pairs(self, ins_idx_list):
        i = 0
        while ins_idx_list[i] >= len(self.raw_graphs[i][1]):
            i += 1
        i = ins_idx_list[i]
        img = np.load(self.raw_graphs[i][1][i])
        if self.cfg.cDNA:
            img = img[[2, 0, 1], :, :]
        aug_img_pairs = []
        for _ in range(2):
            aug_img = self.transform(image=img.copy().transpose(1, 2, 0))['image'] / 15
            aug_img_pairs.append(aug_img)
        return aug_img_pairs

    def __getitem__(self, i):
        reactant_graph = self.raw_graphs[i][0]
        # img = np.load(self.raw_graphs[i][1][0])
        ins_idx_list = [i for i in range(self.cfg.instance_num)]
        random.shuffle(ins_idx_list)
        img_pairs = self.get_img_pairs(ins_idx_list)
        img_ins, masks, chosen_idx = self.get_img_ins(i, ins_idx_list)
        data_dict = {'graphs': reactant_graph,
                     'imgs_1': img_pairs[0],
                     'imgs_2': img_pairs[1],
                     'imgs_ins': img_ins, 'masks': masks,
                     'chosen_idx': chosen_idx}
        return data_dict
