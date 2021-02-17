from omegaconf import OmegaConf
import os.path as osp
import hydra
import inspect
from functools import partial, update_wrapper
import jammy.utils.imp as imp

__all__ = ["hydpath"]


def hydpath(input_path=None):
    if input_path is None or len(input_path) == 0:
        return hydra.utils.to_absolute_path(".")
    if input_path[0] == "/":
        return input_path
    if input_path[0] == "~":
        return osp.expanduser(input_path)
    return hydra.utils.to_absolute_path(input_path)


def hyd_instantiate(cfg, *args, **kwargs):
    """
    A helper func helps initialize from omegaconf(hydra) cfg

    Args:
        cfg: omegaconf.DictConfig, must contain `_target_` key word
             if `param` or `params` present in the cfg, they are
             high priority to initialize instance
             otherwise, use cfg

    Note:
        when use the method pay attention to the parameters order

    args and kwargs: other parameters needed to initialize network
    """
    if cfg is None:
        return None
    module = imp.load_class(cfg["_target_"])
    if "params" in cfg:
        _params = OmegaConf.to_container(cfg.params, resolve=True)
    elif "param" in cfg:
        _params = OmegaConf.to_container(cfg.param, resolve=True)
    else:
        _params = OmegaConf.to_container(cfg, resolve=True)
        del _params["_target_"]
    if inspect.isclass(module):
        return module(*args, **kwargs, **_params)
    elif inspect.isfunction(module):
        partial_fn = partial(module, *args, **kwargs, **_params)
        update_wrapper(partial_fn, module)
        return partial_fn
    else:
        raise RuntimeError(" only support function and class")