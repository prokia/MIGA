from utils.config import ConfigConstructor


class BasicSubmiter(object):
    def __init__(self, cfg_path):
        self.cfg = self.get_config(cfg_path)

    def get_config(self, config_path):
        config_construct = ConfigConstructor(config_path)
        config = config_construct.get_config()
        return config
