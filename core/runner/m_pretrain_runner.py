import os
import time

from torch.cuda.amp import autocast

from core.runner import runner_register, PretrainRunner
from utils.tensor_operator import to_device, tensor2array

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

warnings.filterwarnings('ignore')

import torch
from torch_geometric.nn import global_mean_pool

from tqdm import tqdm
import numpy as np

# from config import args

from utils.loggers import LoggerManager

logger = LoggerManager().get_logger()


@runner_register
class MILPretrainRunner(PretrainRunner):
    def __init__(self, *args, **kwargs):
        super(MILPretrainRunner, self).__init__(*args, **kwargs)

    def train_step(self, data_dict):
        self.model.train()
        data_dict = to_device(data_dict, self.device)
        self.optimizer.zero_grad()
        fp16 = self.cfg.get('fp16', False)
        if self.cfg.run.get('intra_modality_forward', True):
            with autocast(fp16):
                intra_model_out_dict = self.model.intra_modality_forward(data_dict)
                self.fp16_backward(intra_model_out_dict['total_loss'])
            self.fp16_optimizer_update()
            total_loss = tensor2array(intra_model_out_dict.pop('total_loss'))
        else:
            total_loss = 0
            intra_model_out_dict = {}
        self.optimizer.zero_grad()
        with autocast(fp16):
            inter_model_out_dict = self.model.inter_modality_forward(data_dict)
            self.fp16_backward(inter_model_out_dict['total_loss'])
        self.fp16_optimizer_update()
        inter_model_out_dict = tensor2array(inter_model_out_dict)
        inter_model_out_dict['total_loss'] += total_loss
        inter_model_out_dict.update(intra_model_out_dict)
        self.update_loss(inter_model_out_dict)
