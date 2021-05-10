from .meta import Singleton

__all__ = ["Gcfg", "pub_cfg", "set_pub_cfg", "get_pub_cfg"]


class Gcfg(metaclass=Singleton):
    def __init__(self, cfg):
        assert "ready" not in cfg
        assert "cfg" not in cfg
        self.cfg = cfg
        for key, value in dict(cfg).items():
            self.__dict__[key] = value
            setattr(Gcfg, key, value)


pub_cfg = None


def set_pub_cfg(cfg):
    global pub_cfg
    pub_cfg = cfg


def get_pub_cfg(cfg):
    return pub_cfg
