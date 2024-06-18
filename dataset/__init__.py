from utils.my_containers import Constructor

dataset_register = Constructor()
from .omics import NewSmilesOmicsDataset, process_data

def get_dataset(name):
    return dataset_register[name]
