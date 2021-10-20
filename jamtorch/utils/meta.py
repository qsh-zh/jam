#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meta.py
# Author : Qinsheng
# Email  : qsh.27@gmail.com
# Date   : 01/24/2018
#
# This file modified based on Jacinle.
# Distributed under terms of the MIT license.

import functools

import numpy as np
import six
import torch
import torch.distributed as dist

from jammy.utils.meta import stmap

SKIP_TYPES = six.string_types

__all__ = [
    "as_tensor",
    "as_numpy",
    "as_float",
    "as_cuda",
    "as_device",
    "as_cpu",
    "as_detached",
    "is_master",
]


def _as_tensor(cur_obj):
    if isinstance(cur_obj, SKIP_TYPES):
        return cur_obj
    if torch.is_tensor(cur_obj):
        return cur_obj
    return torch.from_numpy(np.array(cur_obj))


def as_tensor(obj):
    return stmap(_as_tensor, obj)


def _as_numpy(cur_obj):
    if isinstance(cur_obj, SKIP_TYPES):
        return cur_obj
    if torch.is_tensor(cur_obj):
        return cur_obj.cpu().numpy()
    return np.array(cur_obj)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(cur_obj):
    if isinstance(cur_obj, SKIP_TYPES):
        return cur_obj
    if torch.is_tensor(cur_obj):
        return cur_obj.item()
    arr = as_numpy(cur_obj)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)


def _as_cuda(cur_obj):
    if torch.is_tensor(cur_obj):
        return cur_obj.cuda()
    return cur_obj


def as_cuda(obj):
    return stmap(_as_cuda, obj)


def _as_device(cur_obj, device):
    if isinstance(cur_obj, SKIP_TYPES):
        return cur_obj
    if torch.is_tensor(cur_obj):
        cur_obj.to(device)
    return cur_obj


def as_device(obj, device=torch.device("cuda:0")):
    return stmap(functools.partial(_as_device, device=device), obj)


as_cpu = functools.partial(as_device, device=torch.device("cpu"))


def _as_detached(cur_obj, clone=False):
    if torch.is_tensor(cur_obj):
        if clone:
            return cur_obj.clone().detach()
        return cur_obj.detach()
    return cur_obj


def as_detached(obj, clone=False):
    return stmap(functools.partial(_as_detached, clone=clone), obj)


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0
