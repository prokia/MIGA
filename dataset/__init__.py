from utils.my_containers import Register, Constructor

dataset_register = Constructor()
from .omics import NewSmilesOmicsDataset
from .mil_omics import MILSmilesOmicsDataset


def get_dataset(name):
    return dataset_register[name]
