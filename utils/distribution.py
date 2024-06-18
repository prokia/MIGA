import os
import torch
from torch import distributed as dist
from functools import partial

from utils.my_containers import singleton


@singleton
class DeviceManager(object):
    def __init__(self, device=''):
        self.device = setup_device(device, mannul=False)

    def get_device(self):
        return self.device


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def is_master_process():
    return os.environ.get('RANK', '-1') in ['-1', '0']


def is_single_process():
    return os.environ.get('RANK', '-1') == '-1'


def sync_barrier_reX(func, X):
    """
    Warning: Nesting doll will lead to DEAD LOCK!
    Wrapper this func X.
    :param func: function only effective in master process, must be atomic operation.
    :param X: format of func returning to create fake return for sub-process
    :return: a barrier func
    """
    def barrier_func(*args, **kwarg):
        if is_single_process():
            return func(*args, **kwarg)
        x = X
        if is_master_process():
            x = func(*args, **kwarg)
            dist.barrier()
        else:
            dist.barrier()
        return x

    return barrier_func


sync_barrier_re1 = partial(sync_barrier_reX, X=None)
sync_barriers_re2 = partial(sync_barrier_reX, X=[None, None])


def setup_device(gpus, mannul=False):
    if not mannul:
        if gpus is None:
            device = torch.device("cpu")
            return device
        # if os.environ.get('RANK', '-1') in ['-1', '0']:
        #     print('Device:', gpus)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # "0, 5, 6"
    if gpus is None or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")
