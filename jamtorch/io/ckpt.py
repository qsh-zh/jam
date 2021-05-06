import torch
import torch.nn as nn
import inspect
import collections
import jammy.io as io
from io import StringIO
from jammy.utils.printing import stprint
import os.path as osp
from jamtorch.logging import get_logger

__all__ = [
    "state_dict",
    "attr_dict",
    "save_ckpt",
    "load_ckpt",
    "aug_ckpt",
    "resume_cfg",
]

logger = get_logger()


def state_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, nn.Module):
        if hasattr(obj, "module"):
            obj = obj.module
    if hasattr(obj, "state_dict") and inspect.ismethod(obj.state_dict):
        return obj.state_dict()

    raise RuntimeError


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


def save_ckpt(state, is_best, filename="checkpoint", bestname="model_best"):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        io.copy(filename, "{}.pth.tar".format(bestname))
        logger.info("Save best up to now: {}".format(filename))


def load_ckpt(gpu=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)
    if isinstance(gpu, int):
        gpu = f"cuda:{gpu}"
    if osp.isfile(filename):
        logger.critical("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=gpu)
        if "env" in checkpoint and checkpoint["env"] is not None:
            env = checkpoint["env"]
            mem_buffer = StringIO()
            stprint(env, file=mem_buffer)
            logger.info("\n" + mem_buffer.getvalue())
        return checkpoint
    else:
        logger.critical("==> Checkpoint '{}' not found".format(filename))
        exit(0)


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
        for pth_file in [ckpt_origin, f"{ckpt_origin}.pth", f"{ckpt_origin}.pth.tar"]:
            if osp.isfile(pth_file):
                break
        ckpt = torch.load(pth_file, map_location=None)
        cfg = ckpt["cfg"] if "cfg" in ckpt else cfg
    return cfg
