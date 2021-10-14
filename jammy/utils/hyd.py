import inspect
import os
import os.path as osp
from collections.abc import Mapping
from functools import partial, update_wrapper

import hydra
from omegaconf import DictConfig, OmegaConf

from jammy import io
from jammy.logging import get_logger
from jammy.utils import imp

logger = get_logger()

__all__ = ["hydpath", "instantiate", "hyd_instantiate", "link_hyd_run", "update_cfg"]


def path(input_path=None):
    if input_path is None or len(input_path) == 0:
        return hydra.utils.to_absolute_path(".")
    if input_path[0] == "/":
        return input_path
    if input_path[0] == "~":
        return osp.expanduser(input_path)
    return hydra.utils.to_absolute_path(input_path)


def hydpath(input_path=None):
    return path(input_path)


def hyd_instantiate(_cfg, *args, **kwargs):
    """
    A helper func helps initialize from omegaconf(hydra) _cfg

    Args:
        _cfg: omegaconf.DictConfig, must contain `_target_` key word
             if `param` or `params` present in the _cfg, they are
             high priority to initialize instance
             otherwise, use _cfg

    Note:
        when use the method pay attention to the parameters order

    args and kwargs: other parameters needed to initialize network
    """
    if _cfg is None:
        stack = inspect.stack()[1]
        logger.info(
            f"File {stack.filename} {stack.lineno}: {stack.function} creating None object"
        )
        return None
    module = imp.load_class(_cfg["_target_"])
    if "params" in _cfg:
        _params = OmegaConf.to_container(_cfg.params, resolve=True)
    elif "param" in _cfg:
        _params = OmegaConf.to_container(_cfg.param, resolve=True)
    else:
        _params = OmegaConf.to_container(_cfg, resolve=True)
        del _params["_target_"]
    if inspect.isclass(module):
        return module(*args, **kwargs, **_params)
    if inspect.isfunction(module):
        partial_fn = partial(module, *args, **kwargs, **_params)
        update_wrapper(partial_fn, module)
        return partial_fn

    raise RuntimeError(" only support function and class")


def instantiate(_cfg, *args, **kwargs):
    """
    A helper func helps instantiate from omegaconf(hydra) _cfg

    Args:
        _cfg: omegaconf.DictConfig, must contain `_target_` key word
             if `param` or `params` present in the _cfg, they are
             high priority to initialize instance
             otherwise, use _cfg

    Note:
        when use the method pay attention to the parameters order

    args and kwargs: other parameters needed to initialize network
    """
    instance_ = hyd_instantiate(_cfg, *args, **kwargs)
    if inspect.isfunction(instance_):
        return instance_()
    if isinstance(instance_, partial):
        return instance_()
    return instance_


def flatten_dict(cfg):
    rtn = {}
    for k, v in cfg.items():
        if isinstance(v, Mapping):
            sub = {f"{k}.{sub_k}": sub_v for sub_k, sub_v in flatten_dict(v).items()}
            rtn.update(sub)
        else:
            rtn[k] = v
    return rtn


def link_hyd_run(dst_fname=".latest_exp", proj_path=None):
    exp_folder = os.getcwd()
    if proj_path is None:
        proj_path = hydra.utils.get_original_cwd()
    io.link(exp_folder, osp.join(proj_path, dst_fname), use_relative_path=False)
    logger.info(f"{exp_folder} ==>> {osp.join(proj_path, dst_fname)}")


def update_cfg(cfg: DictConfig, dotlist):
    new_conf = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.merge(cfg, new_conf)
