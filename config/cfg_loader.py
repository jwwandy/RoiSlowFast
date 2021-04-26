import yaml
from .cfg_node import get_default_cfg

def load_cfg(filename):
    with open(filename, "r") as f:
        raw_cfg = yaml.load(f, Loader=yaml.CLoader)
    cfg = get_default_cfg()
    cfg.update(raw_cfg)
    return cfg

