import importlib

from utils.my_containers import Register, Constructor

runner_register = Constructor()
from .pretrain_runner import PretrainRunner
from .m_pretrain_runner import MILPretrainRunner
from .pretrain_runner_noise_graph import NoiseGraphPretrainRunner


def get_runner(runner_name):
    return runner_register[runner_name]
