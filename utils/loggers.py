import os
import logging
import datetime
from functools import partial

from utils.distribution import is_master_process, sync_barrier_re1
from utils.my_containers import singleton
import re


@singleton
class LoggerManager(object):
    def __init__(self, log_dir=None):
        self.logger_dictionary = {}
        ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
        ts = ts.replace(":", "_").replace("-", "_")
        self.file_path = os.path.join(log_dir, "run_{}.log".format(ts)) if log_dir is not None else None
        self.default_level = logging.INFO
        logging.basicConfig(level=self.default_level if is_master_process() else logging.ERROR,
                            format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    def name2logger(self, name):
        pattern = re.compile(r' ')
        func_name = re.sub(pattern, '_', name)
        return func_name

    def add_logger(self, name,  level=logging.INFO, write_to_file=True):
        if name in self.logger_dictionary:
            return
        logger = logging.getLogger(name)
        if self.file_path is not None and write_to_file:
            hdlr = logging.FileHandler(self.file_path)
            hdlr.setLevel(level)
            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
            logger.setLevel(level)
        self.logger_dictionary[name] = logger
        self.__setattr__(f'write_to_{name}', partial(self._write_to_logger, name=name))

    @sync_barrier_re1
    def _write_to_logger(self, msg, level='info', name='Main Stream'):
        self.logger_dictionary[name].__getattribute__(level)(msg)

    def get_logger(self, name='Main Stream', level=logging.INFO, write_to_file=True):
        func_name = self.name2logger(name)
        write_func_name = f'write_to_{func_name }'
        if not hasattr(self, write_func_name):
            self.add_logger(func_name, level, write_to_file)
        return self.__getattribute__(write_func_name)
