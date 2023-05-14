from collections import OrderedDict, abc as container_abc
from copy import deepcopy

import inspect


class Fake(object):
    def __init__(self):
        self.fake = 1


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x: add_register_item(target, x)

    def register_function(self, target):
        return self.register(target())

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


def convert_to_yaml(elements):
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [convert_to_yaml(e) for e in elements]
    if isinstance(elements, dict):
        return {key: convert_to_yaml(value) for key, value in elements.items()}
    return elements


class Constructor(Register):
    def __init__(self):
        super(Constructor, self).__init__()
        self.func_args_dict = {}
        self.non_cfg_obj_set = set()

    def no_cfg_register(self, target):
        self.non_cfg_obj_set.add(target.__name__)
        self.register(target)
        return target

    def build_with_cfg(self, name_with_cfg, pop_name=True):
        if name_with_cfg is None:
            return None
        name_with_cfg = name_with_cfg.copy()
        name_with_cfg = ObjDict(name_with_cfg)
        name = name_with_cfg.pop('name')
        if not pop_name:
            name_with_cfg['name'] = name
        if name in self.non_cfg_obj_set:
            return self[name](**name_with_cfg)
        return self[name](name_with_cfg)


class MethodManager(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, name=None):
        if name is None:
            name = self.cfg.name
        return self.__getattribute__(name)


class ObjDict(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val

    def transform(self):
        for key, item in self.items():
            if isinstance(item, OrderedDict):
                self[key] = ObjDict(item)
                self[key].transform()
        return self

    def to_dict(self):
        re_dict = {}
        for key, item in self.items():
            if isinstance(item, ObjDict):
                re_dict[key] = item.to_dict()
            else:
                re_dict[key] = item
        return re_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def deep_dict_update(shop, custom):
    for key, value in custom.items():
        if key not in shop:
            shop[key] = value
        elif isinstance(value, container_abc.Mapping):
            deep_dict_update(shop[key], custom[key])
        else:
            shop[key] = custom[key]


def boss_update(shop, custom):
    for key, value in custom.items():
        if key not in shop:
            shop[key] = value
        elif isinstance(value, container_abc.Mapping):
            if 'name' in value and value['name'] != shop[key].get('name', None):
                shop[key] = custom[key]
            else:
                boss_update(shop[key], custom[key])
        else:
            shop[key] = custom[key]


def value_dict_mean(dict_list):
    re_dict = deepcopy(dict_list[0])
    for i in range(1, len(dict_list)):
        for key, value in dict_list[i].items():
            re_dict[key] += value
    for key in re_dict:
        re_dict[key] /= len(dict_list)
    return re_dict


def deep_dict_add(dict_1, dict_2, key_pool=None):
    for key, value in dict_1.items():
        if isinstance(value, container_abc.Mapping):
            dict_1[key] = deep_dict_add(dict_1[key], dict_2[key], key_pool)
            continue
        if key_pool is not None and key not in key_pool:
            continue
        dict_1[key] += dict_2[key]
    return dict_1


def deep_dict_sum(dict_list, key_pool):
    re = dict_list[0]
    for each_dict in dict_list[1:]:
        re = deep_dict_add(re, each_dict, key_pool)
    return re


def deep_dict_search(in_dict, target_key):
    if target_key in in_dict:
        return in_dict[target_key]
    else:
        for key in in_dict:
            if isinstance(in_dict[key], container_abc.Mapping):
                sub_search = deep_dict_search(in_dict[key], target_key)
                if sub_search is not None:
                    return sub_search
        return None


def deep_dict_func(in_dict, func, key_pool=None, strict=False):
    def judge_key(in_key):
        if key_pool is None:
            return False
        if strict and not isinstance(key_pool, list):
            return in_key == key_pool
        return in_key in key_pool

    for key, value in in_dict.items():
        if isinstance(value, container_abc.Mapping):
            in_dict[key] = deep_dict_func(in_dict[key], func, key_pool, strict)
            continue
        if not judge_key(key):
            continue
        in_dict[key] = func(in_dict[key])
    return in_dict


def deep_dict_prefix(in_dict, prefix):
    if not isinstance(in_dict, container_abc.Mapping):
        return in_dict
    return {f'{prefix}_{k}': v for k, v in in_dict.items()}


def deep_dict_mean(dict_list, key_pool):
    re_dict = deep_dict_sum(dict_list, key_pool)
    re_dict = deep_dict_func(re_dict, lambda x: x / len(dict_list), key_pool)
    return re_dict


def merge_dis_dict(gather_list):
    re_dict = {}
    for each_dict in gather_list:
        re_dict.update(each_dict)
    return re_dict


def deep_dict_get(in_dict, target, no_get=None):
    if isinstance(target, list):
        re = []
        for t_i, t in enumerate(target):
            n = no_get[t_i] if no_get is not None else None
            re.append(deep_dict_get(in_dict, t, n))
        if len(re) == 1:
            return re[0]
        return re
    if target in in_dict:
        return in_dict[target]
    re = no_get
    for key, value in in_dict.items():
        if isinstance(value, container_abc.Mapping):
            re = deep_dict_get(value, target, no_get)
            if re != no_get:
                return re
    return re


def str2list(in_str):
    if isinstance(in_str, list):
        return in_str
    return eval(in_str)
