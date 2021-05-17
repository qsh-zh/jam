from .meta import Singleton
from jammy.cli import timeout_input

__all__ = ["Gcfg", "pub_cfg", "set_pub_cfg", "get_pub_cfg", "omgcfg_create"]


class Gcfg(metaclass=Singleton):
    def __init__(self, cfg):
        assert "ready" not in cfg
        assert "cfg" not in cfg
        assert "reset" not in cfg
        assert "clear" not in cfg
        self.cfg = cfg
        for key, value in dict(cfg).items():
            self.__dict__[key] = value
            setattr(Gcfg, key, value)

    @classmethod
    def reset(cls, cfg=None):
        if cfg is None:
            cls._instances.clear()
        return Gcfg(cfg)


pub_cfg = None


def set_pub_cfg(cfg):
    global pub_cfg
    pub_cfg = cfg


def get_pub_cfg(cfg):
    return pub_cfg


def assert_name(cfg, default="Default", strict=True):
    if "name" not in cfg or cfg.name is None:
        cfg.name = timeout_input("cfg.name", 10, default, strict)
        return False
    return True


def omgcfg_create(obj):
    from omegaconf import OmegaConf

    return OmegaConf.create(obj)
