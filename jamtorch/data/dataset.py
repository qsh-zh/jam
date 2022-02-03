import numpy as np
import torch as th
from torch.utils.data import Dataset

__all__ = ["TensorDataset"]


class TensorDataset(Dataset):
    def __init__(self, tensor):
        if isinstance(tensor, np.ndarray):
            data = th.from_numpy(tensor)
        elif isinstance(tensor, th.Tensor):
            data = tensor
        else:
            raise RuntimeError(f"{type(tensor)} not supported yet")

        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
