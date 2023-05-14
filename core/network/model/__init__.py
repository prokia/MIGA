from utils.my_containers import Constructor

model_register = Constructor()
from .encoder.cnn.cnn import CnnEncoder
from .encoder.gnn.gnn import GNNEncoder

from .encoder.graph_transformer.transformer_model import GraphTransformer

from .decoder.vae.vae import VariationalAutoEncoder


def get_model(model_cfg):
    return model_register.build_with_cfg(model_cfg)
