import torch
from jammy.random.rng import gen_rng
import torch.utils.data as data
import numpy as np

__all__ = ["get_batch", "get_subset"]


def get_batch_loader(data_loader):
    return next(iter(data_loader))[0]


def get_batch(data, size=10):
    if isinstance(data, torch.utils.data.DataLoader):
        return get_batch_loader(data)
    elif isinstance(data, torch.utils.data.Dataset):
        rng = gen_rng()
        subset = torch.utils.data.Subset(data, rng.choice(len(data), size=size))
        loader = torch.utils.data.DataLoader(subset, batch_size=size)
        return get_batch_loader(loader)
    else:
        raise RuntimeError(f"Type {type(data)} not supported")

def get_subset(dataset, target_size):
    if target_size is None:
        return dataset
    return data.Subset(
        dataset,
        np.random.choice(len(dataset),size=target_size)
    )
