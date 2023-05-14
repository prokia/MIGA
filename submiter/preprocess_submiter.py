import datetime
import os
import shutil

from ruamel import yaml as ryaml
from torch.utils.tensorboard import SummaryWriter

from preprocess.runner import get_preprocess_runner
from submiter.basic_submiter import BasicSubmiter
from utils.loggers import LoggerManager
from utils.utils import mkdir
from utils.utils import setup_seeds


class PreprocessSubmiter(BasicSubmiter):
    def __init__(self, cfg_path):
        super(PreprocessSubmiter, self).__init__(cfg_path)
        seed = self.cfg.get('seed', 1)
        setup_seeds(seed, set_torch=False)
        self.prepare_record_system()
        self.logger = self.setup_logger()
        self.writer = self.setup_writer()
        self.runner = self.get_runner()

    def get_runner(self):
        runner = get_preprocess_runner(self.cfg, self.writer)
        return runner

    @property
    def record_dir_name_list(self):
        dir_name_list = ['log', f'preprocessed_data', 'tb', 'update_config']
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
        self.cfg.code_dir = os.path.join(self.cfg.run.save_path, 'gemini')
        self.mkdir_file_system()

    def mkdir_file_system(self):
        tuple(map(mkdir, [self.cfg['run']['save_path']] + self.record_dir_list))
        shutil.copytree("", self.cfg.code_dir)
        update_config = os.path.join(self.cfg.update_config_dir, 'update_config.yaml')
        with open(update_config, "w", encoding="utf-8") as f:
            ryaml.dump(self.cfg.to_dict(), f, Dumper=ryaml.RoundTripDumper)

    def setup_logger(self):
        logger = LoggerManager(self.cfg.log_dir).get_logger()
        logger(f"RUNDIR: {self.cfg.log_dir}")
        logger("Let the games begin :)")
        return logger

    def setup_writer(self):
        writer = SummaryWriter(self.cfg.tb_dir)
        return writer

    def submit(self):
        self.runner()
