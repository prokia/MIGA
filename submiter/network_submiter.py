import datetime
import os
import shutil

import torch
import torch.distributed as dist
from ruamel import yaml as ryaml
from torch.utils.tensorboard import SummaryWriter

from submiter.basic_submiter import BasicSubmiter
from utils.config import ConfigConstructor
from utils.distribution import is_single_process, sync_barrier_re1
from utils.loggers import LoggerManager
from utils.my_containers import ObjDict
from utils.utils import mkdir
from utils.utils import setup_seeds


class NetworkSubmiter(BasicSubmiter):
    def __init__(self, cfg_path):
        super(NetworkSubmiter, self).__init__(cfg_path)
        self.init_dist()
        seed = self.cfg.get('seed', 1)
        setup_seeds(seed)
        self.prepare_record_system()
        self.logger = self.setup_logger()
        self.writer = self.setup_writer()
        from core.runner import get_runner
        self.runner = get_runner(self.cfg.runner)(self.cfg, self.writer)

    def init_dist(self, mannul=False):
        from utils.distribution import DeviceManager
        DeviceManager(self.cfg.gpus)
        if is_single_process():
            return
        self.cfg.gpus = self.cfg.gpus.split(',')[int(os.environ['RANK'])]
        torch.cuda.set_device(int(os.environ['RANK']))
        dist.init_process_group(backend='nccl')
        self.cfg['dist'] = ObjDict({'rank': int(os.environ['RANK']),
                                    'world_size': dist.get_world_size()})
        self.cfg['dataset']['dist'] = self.cfg['dist']

    def get_config(self, config_path):
        config_construct = ConfigConstructor(config_path)
        config = config_construct.get_config()
        return config

    @property
    def record_dir_name_list(self):
        dir_name_list = ['log', 'weight', 'tb', 'csv', 'update_config']
        return dir_name_list

    @property
    def record_dir_list(self):
        dir_list = list(map(lambda p: os.path.join(self.cfg['run']['save_path'], p), self.record_dir_name_list))
        return dir_list

    def prepare_record_system(self):
        timestamp = datetime.datetime.now()
        version = '%d-%d-%d-%02.0d-%02.0d-%02.0d' % \
                  (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second)
        self.cfg['run']['save_path'] = self.cfg['run']['save_path'].format(
            self.cfg['run']['task'],
            version)
        for record_dir_name, record_dir in zip(self.record_dir_name_list, self.record_dir_list):
            self.cfg[f'{record_dir_name}_dir'] = record_dir
        self.cfg.code_dir = os.path.join(self.cfg.run.save_path, 'miga')
        self.mkdir_file_system()

    @sync_barrier_re1
    def mkdir_file_system(self):
        tuple(map(mkdir, [self.cfg['run']['save_path']] + self.record_dir_list))
        shutil.copytree(".", self.cfg.code_dir)
        update_config = os.path.join(self.cfg.update_config_dir, 'update_config.yaml')
        with open(update_config, "w", encoding="utf-8") as f:
            ryaml.dump(self.cfg.to_dict(), f, Dumper=ryaml.RoundTripDumper)

    def setup_logger(self):
        logger = LoggerManager(self.cfg.log_dir).get_logger()
        logger(f"RUNDIR: {self.cfg.log_dir}")
        logger("Let the games begin :)")
        return logger

    @sync_barrier_re1
    def setup_writer(self):
        writer = SummaryWriter(self.cfg.tb_dir)
        return writer

    def submit(self):
        self.runner()
