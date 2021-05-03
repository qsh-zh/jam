from .meta import Singleton


class Gcfg(metaclass=Singleton):
    def __init__(self, cfg):
        self.cfg = cfg
        for key, value in cfg.__dict__():
            self.__dict__[key] = value


pub_cfg = None


def set_pub_cfg(cfg):
    global pub_cfg
    pub_cfg = cfg


def get_pub_cfg(cfg):
    return pub_cfg
