import importlib

from utils.my_containers import Register, Constructor

runner_register = Constructor()
from .pretrain_runner import PretrainRunner
from .pretrain_runner_dense_matrix import DenseGraphPretrainRunner


def get_runner(runner_name):
    return runner_register[runner_name]
