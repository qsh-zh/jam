#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import functools
import numpy as np

import torch
import torch.distributed as dist
from jammy.utils.meta import stmap

SKIP_TYPES = six.string_types

__all__ = [
    "as_tensor",
    "as_numpy",
    "as_float",
    "as_cuda",
    "as_cpu",
    "as_detached",
    "is_master",
]


def _as_tensor(o):
    from torch.autograd import Variable

    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        return o
    if torch.is_tensor(o):
        return o
    return torch.from_numpy(np.array(o))


def as_tensor(obj):
    return stmap(_as_tensor, obj)


def _as_numpy(o):
    from torch.autograd import Variable

    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        o = o
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(o):
    if isinstance(o, SKIP_TYPES):
        return o
    if torch.is_tensor(o):
        return o.item()
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)


def _as_cpu(o):
    from torch.autograd import Variable

    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cpu()
    return o


def as_cpu(obj):
    return stmap(_as_cpu, obj)


def _as_cuda(o):
    from torch.autograd import Variable

    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cuda()
    return o


def as_cuda(obj):
    return stmap(_as_cuda, obj)


def _as_detached(o, clone=False):
    if torch.is_tensor(o):
        if clone:
            return o.clone().detach()
        return o.detach()
    return o


def as_detached(obj, clone=False):
    return stmap(functools.partial(_as_detached, clone=clone), obj)


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0
