import torch
from jammy.random.rng import gen_rng

__all__ = ["get_batch"]


def get_batch_loader(data_loader):
    return next(iter(data_loader))[0]


def get_batch(data, size=10):
    if isinstance(data, torch.utils.data.DataLoader):
        return get_batch_loader(data)
    elif isinstance(data, torch.utils.data.Dataset):
        rng = gen_rng()
        subset = torch.utils.data.Subset(data, rng.choice(len(data)), size=size)
        loader = torch.utils.data.DataLoader(subset, batch_size=size)
        return get_batch_loader(loader)
    else:
        raise RuntimeError
