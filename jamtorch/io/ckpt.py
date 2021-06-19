import collections
import inspect
import os.path as osp
from io import StringIO

import numpy as np
import torch
import torch.nn as nn

import jammy.io as io
from jammy.utils.matching import IENameMatcher
from jammy.utils.printing import stprint
from jamtorch.logging import get_logger

__all__ = [
    "state_dict",
    "attr_dict",
    "save_ckpt",
    "load_ckpt",
    "aug_ckpt",
    "resume_cfg",
    "set_ckpt",
    "load_state_dict",
]


def state_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, nn.Module):
        if hasattr(obj, "module"):
            obj = obj.module
    if hasattr(obj, "state_dict") and inspect.ismethod(obj.state_dict):
        return obj.state_dict()

    raise RuntimeError


def set_ckpt(src, des, atrs):
    for atr in atrs:
        setattr(des, atr, getattr(src, atr))


def attr_dict(obj, key_list):
    if isinstance(key_list, str):
        key_list = [key_list]
    if isinstance(
        key_list, (collections.abc.Set, collections.Sequence, collections.UserList)
    ):
        rtn = {}
        for key in key_list:
            rtn[key] = state_dict(getattr(obj, key))
        return rtn
    raise RuntimeError


def save_ckpt(state, is_best, filename="checkpoint", bestname="best"):
    filename = "{}.pth".format(filename)
    torch.save(state, filename)
    if is_best:
        io.link(filename, "{}.pth".format(bestname), use_relative_path=False)
        logger = get_logger()
        logger.info("Save best up to now: {}".format(filename))


def load_ckpt(gpu=None, filename="checkpoint"):
    logger = get_logger()
    for ckpt_file in [filename, f"{filename}.pth"]:
        if osp.isfile(ckpt_file):
            break
    if isinstance(gpu, int):
        gpu = f"cuda:{gpu}"
    if osp.isfile(ckpt_file):
        logger.critical("==> Loading from checkpoint '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file, map_location=gpu)
        if "env" in checkpoint and checkpoint["env"] is not None:
            env = checkpoint["env"]
            mem_buffer = StringIO()
            stprint(env, file=mem_buffer)
            logger.info("\n" + mem_buffer.getvalue())
        return checkpoint

    logger.critical("==> Checkpoint '{}' not found".format(filename))
    raise RuntimeError


def aug_ckpt(ckpt, aug_dict, is_save=False, ckpt_file=None):
    """augment the checkpoint,
    :returns: ckpt

    """
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt, torch.device("cpu"))

    assert isinstance(ckpt, dict)

    ckpt.update(aug_dict)

    if is_save:
        assert ckpt_file is not None
        torch.save(ckpt, ckpt_file)

    return ckpt


def resume_cfg(cfg):
    """resume cfg from ckpt"""
    if cfg.trainer is not None:
        n_cfg = cfg.trainer
    else:
        n_cfg = cfg

    if n_cfg.resume and n_cfg.ckpt is not None:
        ckpt_origin = n_cfg.ckpt
        for pth_file in [ckpt_origin, f"{ckpt_origin}.pth"]:
            if osp.isfile(pth_file):
                break
        ckpt = torch.load(pth_file, map_location=None)
        cfg = ckpt["cfg"] if "cfg" in ckpt else cfg
    return cfg


def load_state_dict(
    model, ckpt_state_dict, include=None, exclude=None
):  # pylint: disable= too-many-locals
    logger = get_logger()
    if hasattr(model, "module"):
        model = model.module

    matcher = IENameMatcher(include, exclude)
    with matcher:
        ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if matcher.match(k)}
    stat = matcher.get_last_stat()  # examine unused rules
    if len(stat[1]) > 0:
        logger.critical(
            "Weights {}: {}.".format(stat[0], ", ".join(sorted(list(stat[1]))))
        )

    # Build the tensors.
    for k, v in ckpt_state_dict.items():
        if isinstance(v, np.ndarray):
            ckpt_state_dict[k] = torch.from_numpy(v)

    error_msg, warn_msg = [], []
    own_state = model.state_dict()
    for name, param in ckpt_state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:  # pylint: disable=broad-except
                error_msg.append(
                    "While copying the parameter named {}, "
                    "whose dimensions in the model are {} and "
                    "whose dimensions in the checkpoint are {}.".format(
                        name, own_state[name].size(), param.size()
                    )
                )

    missing = set(own_state.keys()) - set(ckpt_state_dict.keys())
    if len(missing) > 0:
        warn_msg.append('Missing keys in state_dict: "{}".'.format(missing))

    unexpected = set(ckpt_state_dict.keys()) - set(own_state.keys())
    if len(unexpected) > 0:
        warn_msg.append('Unexpected key "{}" in state_dict.'.format(unexpected))

    if len(error_msg) > 0:
        raise KeyError("\n".join(error_msg))

    if len(warn_msg) > 0:
        logger.critical("\n".join(warn_msg))
