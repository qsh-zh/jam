import functools

import numpy as np
import torch
from torch.utils import data
from tqdm.auto import tqdm

from jammy.random.rng import gen_rng
from jamtorch.utils.pytree import merge_tree

__all__ = ["get_batch", "get_subset", "num_to_groups"]


def get_batch_loader(data_loader):
    return next(iter(data_loader))


def get_batch(fdata, size=10):
    if isinstance(fdata, torch.utils.data.DataLoader):
        return get_batch_loader(fdata)
    if isinstance(fdata, torch.utils.data.Dataset):
        rng = gen_rng()
        subset = torch.utils.data.Subset(fdata, rng.choice(len(fdata), size=size))
        loader = torch.utils.data.DataLoader(subset, batch_size=size)
        return get_batch_loader(loader)

    raise RuntimeError(f"Type {type(fdata)} not supported")


def get_subset(dataset, target_size):
    if target_size is None:
        return dataset
    return data.Subset(dataset, np.random.choice(len(dataset), size=target_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def batch_run(func, batch_size=100, is_tqdm=False):
    @functools.wraps(func)
    def new_func(total_batch, *args, **kwargs):
        arr = num_to_groups(total_batch, batch_size)
        rtn = []
        if is_tqdm:
            arr = tqdm(arr)
        for cur_batch_size in arr:
            rtn.append(func(cur_batch_size, *args, **kwargs))
        return merge_tree(rtn)

    return new_func
