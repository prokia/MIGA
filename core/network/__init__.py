from utils.my_containers import Constructor

network_register = Constructor()

from .wrapper.miga import MIGA


def get_network(cfg):
    network = network_register.build_with_cfg(cfg)
    return network