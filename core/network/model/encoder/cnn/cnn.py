import timm
import torch.nn as nn
import torch


# num_atom_type = 120  # including the extra mask tokens
# num_chirality_tag = 3
#
# num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
# num_bond_direction = 3
from core.network.model import model_register


@model_register
class CnnEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        para_key = ['n_ch', 'pretrained']
        default_key_val = [3, True]
        for key, v in zip(para_key, default_key_val):
            custom_v = self.cfg.get(key, v)
            self.__setattr__(key, custom_v)
        model_name = cfg.get('instance_model_name')
        pretrained = self.pretrained
        n_ch = self.n_ch
        target_num = self.cfg.target_num
        self.model = timm.create_model(model_name, pretrained)
        # print(self.model)
        if ('efficientnet' in model_name) or ('mixnet' in model_name):
            self.model.conv_stem.weight = nn.Parameter(
                self.model.conv_stem.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif model_name in ['resnet34d']:
            self.model.conv1[0].weight = nn.Parameter(
                self.model.conv1[0].weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif ('resnet' in model_name or 'resnest' in model_name) and 'vit' not in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif 'rexnet' in model_name or 'regnety' in model_name or 'nf_regnet' in model_name:
            self.model.stem.conv.weight = nn.Parameter(
                self.model.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif 'resnext' in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif 'hrnet_w32' in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif 'densenet' in model_name:
            self.model.features.conv0.weight = nn.Parameter(
                self.model.features.conv0.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif 'ese_vovnet39b' in model_name or 'xception41' in model_name:
            self.model.stem[0].conv.weight = nn.Parameter(
                self.model.stem[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif 'dpn' in model_name:
            self.model.features.conv1_1.conv.weight = nn.Parameter(
                self.model.features.conv1_1.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_channels, target_num)
            self.model.classifier = nn.Identity()
        elif 'inception' in model_name:
            self.model.features[0].conv.weight = nn.Parameter(
                self.model.features[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.last_linear.in_features, target_num)
            self.model.last_linear = nn.Identity()
        elif 'vit' in model_name:
            self.model.patch_embed.proj.weight = nn.Parameter(
                self.model.patch_embed.proj.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        elif 'vit_base_resnet50' in model_name:
            self.model.patch_embed.backbone.stem.conv.weight = nn.Parameter(
                self.model.patch_embed.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
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
        re = self.myfc(x)
        return re
