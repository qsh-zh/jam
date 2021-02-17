import torch
import torch.nn as nn
import inspect
import collections
import jammy.io as io
from io import StringIO
from jammy.utils.printing import stprint
import os.path as osp
from jamtorch.logging import get_logger

__all__ = ["state_dict", "attr_dict", "save_ckpt", "load_ckpt"]

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
