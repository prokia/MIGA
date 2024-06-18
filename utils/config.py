import yaml
from collections import OrderedDict, abc as container_abc
from utils.my_containers import ObjDict
from utils.utils import ordered_load


def load_yaml(config_path):
    with open(config_path, encoding='utf-8') as fp:
        cfg = ordered_load(fp, yaml.SafeLoader)
    return ObjDict(cfg)


class ConfigConstructor(object):
    def __init__(self, config_path):
        """

        Args:
            config_path: path to the config file.

        Rules:
        This is a factory class to construct the config.
         _Base_config is used for inheritance and overwriting in config.
        The key-value pairs in the current configuration file will replace the corresponding key-value pairs in _Base_config.
        In a special case where a dictionary in the current configuration file contains a special key "name" and
        the value of the "name" key in the corresponding dictionary in
        _base_config is different from the value of the "name" key in the current dictionary,
        the inheritance of that dictionary will ignore all key-value pairs in the corresponding dictionary in _base_config.
        There is also a special case of class inheritance where we want to inherit the corresponding dictionary in _base_config
        even if the value of "name" has been changed. In this case,
        you need to disable this mechanism by marking the "name" as "name*".
        """
        self.suffix2loadMethods = {'.yaml': load_yaml, '.yml': load_yaml}
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

    def construct_config(self, config_dict, kwargs=None):
        base_config = ObjDict()
        for key, value in config_dict.items():
            if key == '_Base_Config':
                base_config = self.config_inherit(config_dict['_Base_Config'])
            elif isinstance(value, container_abc.Mapping):
                config_dict[key] = self.construct_config(value)
        self.cfg_update(base_config, config_dict)
        if kwargs is not None:
            self.cfg_update(base_config, kwargs)
        if '_Base_Config' in base_config:
            base_config.pop('_Base_Config')
        return base_config

    def get_config(self, kwargs=None):
        cfg = self.construct_config(self.config, kwargs)
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
        # Check if new_cfg is empty
        if not new_cfg:
            # Clear base_cfg
            base_cfg.clear()
        # Set to store keys with '*' at the end
        add_key = set()
        # Iterate over each key-value pair in new_cfg
        for key, value in new_cfg.items():
            # Check if key ends with '*'
            if key[-1] == '*':
                # Get original key
                ori_key = key[:-1]
                # Check if original key is present in base_cfg
                if ori_key not in base_cfg:
                    # Skip if not present
                    continue
                # Add original key to add_key set
                add_key.add(ori_key)
                # Update new_cfg with original key
                new_cfg[ori_key] = new_cfg[key]
                # Remove key with '*' from new_cfg
                new_cfg.pop(key)
        # Iterate over each key-value pair in new_cfg
        for key, value in new_cfg.items():
            # Check if key is present in base_cfg
            if key not in base_cfg:
                # Add key-value pair to base_cfg
                base_cfg[key] = value
            # Check if value is a mapping
            elif isinstance(value, container_abc.Mapping):
                # Check if 'name' is present in value and it is not equal to name in base_cfg
                if 'name' in value and value['name'] != base_cfg[key].get('name', None):
                    # Check if 'name' in value does not end with '*'
                    if value['name'][-1] != '*':
                        # Add key-value pair to base_cfg
                        base_cfg[key] = new_cfg[key]
                        # Skip to next iteration
                        continue
                    # Remove '*' from 'name' in value
                    value['name'] = value['name'][:-1]
                # Recursively update base_cfg
                self.cfg_update(base_cfg[key], new_cfg[key])
            else:
                # Check if key is present in add_key set
                if key in add_key:
                    # Update base_cfg with new_cfg using update_by_type
                    base_cfg[key] = self.update_by_type(base_cfg[key], new_cfg[key])
                else:
                    # Add key-value pair to base_cfg
                    base_cfg[key] = new_cfg[key]
