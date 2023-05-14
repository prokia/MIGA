import os
import logging
import datetime

from tensorboard.backend.event_processing import event_accumulator

from utils.distribution import is_master_process, sync_barrier_re1
from utils.my_containers import singleton, Register
from torch.utils.tensorboard import SummaryWriter


@singleton
class SummaryWriterManager(object):
    def __init__(self, *args, **kwargs):
        if not is_master_process():
            return
        self.writer = SummaryWriter(*args, **kwargs)
        for func_name in dir(self.writer):
            if func_name.startswith('_'):
                continue
            self.__setattr__(func_name, sync_barrier_re1(self.writer.__getattribute__(func_name)))

    # def __getattribute__(self, item):
    #     if 'add' in item or item in ['close', 'flush']:
    #         return sync_barrier_re1(self.writer.__getattribute__(item))


def read_tensorboard(path):
    try:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
    except:
        print(f'Can not read {path} as event log')
        ea = None
    return ea


def get_event_from_dir(path):
    file_name_list = os.listdir(path)
    evs = []
    for file_name in file_name_list:
        file_path = os.path.join(path, file_name)
        ev = get_events_from_path(file_path)
        if ev is None:
            continue
        if isinstance(ev, list):
            evs.extend(ev)
        else:
            evs.append(ev)
    return evs


def get_events_from_path(path):
    if os.path.isdir(path):
        return get_event_from_dir(path)
    if os.path.split(path)[-1].split('.')[0] == 'events':
        return read_tensorboard(path)
    return None


def get_events_from_path_list(path_list):
    evs = []
    for path in path_list:
        ev = get_events_from_path(path)
        if isinstance(ev, list):
            evs.extend(ev)
        else:
            evs.append(ev)
    return evs
