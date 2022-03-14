import yaml
import datetime
import os.path as osp
from types import SimpleNamespace

class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

def load_config(conf):
    with open(conf) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dot = NestedNamespace(config_dict)
    return config_dot

def gen_dirname(top):
    basename = 'trans'
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    dirname = osp.join(top,  "_".join([basename, suffix]))
    return dirname


