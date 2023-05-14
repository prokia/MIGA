import yaml
from .mmcv_config import Config as pyConfig
from collections import OrderedDict, abc as container_abc
from utils.my_containers import ObjDict
from utils.utils import ordered_load


def load_yaml(config_path):
    with open(config_path, encoding='utf-8') as fp:
        cfg = ordered_load(fp, yaml.SafeLoader)
    return ObjDict(cfg)


def load_py(config_path):
    return ObjDict(pyConfig.fromfile(config_path))


class ConfigConstructor(object):
    def __init__(self, config_path):
        self.suffix2loadMethods = {'.yaml': load_yaml, '.yml': load_yaml, '.py': load_py}
        self.inherit_tree = {}
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        for suffix, m in self.suffix2loadMethods.items():
            if suffix in config_path:
                return ObjDict(m(config_path)).transform()
        raise NotImplementedError

    def config_inherit(self, base_config_path_list):
        if isinstance(base_config_path_list, str):
            base_config_path_list = [base_config_path_list]
        base_config = ObjDict()
        for base_cfg_path in base_config_path_list:
            self.cfg_update(base_config, ConfigConstructor(base_cfg_path).get_config())
        return base_config

    def construct_config(self, config_dict):
        base_config = ObjDict()
        for key, value in config_dict.items():
            if key == '_Base_Config':
                base_config = self.config_inherit(config_dict['_Base_Config'])
            elif isinstance(value, container_abc.Mapping):
                config_dict[key] = self.construct_config(value)
        self.cfg_update(base_config, config_dict)
        if '_Base_Config' in base_config:
            base_config.pop('_Base_Config')
        return base_config

    def get_config(self):
        cfg = self.construct_config(self.config)
        return cfg

    def update_by_type(self, base_value, new_value):
        assert type(base_value) == type(new_value)
        if isinstance(new_value, container_abc.Mapping):
            base_value.update(new_value)
            return base_value
        if isinstance(base_value, list):
            base_value.extend(new_value)
            return base_value
        raise NotImplemented

    def cfg_update(self, base_cfg, new_cfg):
        if not new_cfg:
            base_cfg.clear()
        add_key = set()
        for key, value in new_cfg.items():
            if key[-1] == '*':
                ori_key = key[:-1]
                if ori_key not in base_cfg:
                    continue
                add_key.add(ori_key)
                new_cfg[ori_key] = new_cfg[key]
                new_cfg.pop(key)
        for key, value in new_cfg.items():
            if key not in base_cfg:
                base_cfg[key] = value
            elif isinstance(value, container_abc.Mapping):
                if 'name' in value and value['name'] != base_cfg[key].get('name', None):
                    if value['name'][-1] != '*':
                        base_cfg[key] = new_cfg[key]
                        continue
                    value['name'] = value['name'][:-1]
                self.cfg_update(base_cfg[key], new_cfg[key])
            else:
                if key in add_key:
                    base_cfg[key] = self.update_by_type(base_cfg[key], new_cfg[key])
                else:
                    base_cfg[key] = new_cfg[key]
