from utils.my_containers import Constructor

network_register = Constructor()

from .wrapper.miga import MIGA
from .wrapper.intra_modal_miga import IntraModalMIGA
from .wrapper.mil_miga import MilMIGA


def get_network(cfg):
    network = network_register.build_with_cfg(cfg)
    return network
