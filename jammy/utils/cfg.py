from .meta import Singleton


class Gcfg(metaclass=Singleton):
    ok = False

    def __init__(self, cfg):
        self.cfg = cfg
        for key, value in dict(cfg).items():
            self.__dict__[key] = value
            setattr(Gcfg, key, value)
        Gcfg.ok = True


pub_cfg = None


def set_pub_cfg(cfg):
    global pub_cfg
    pub_cfg = cfg


def get_pub_cfg(cfg):
    return pub_cfg
