from typing import Tuple

import torch as th


class TorchRingBuffer:
    def __init__(self, item_shape: Tuple, max_size: int = 10000):
        self.max_size = max_size
        self._data = th.zeros((max_size, *item_shape))
        self.ptr_cnt = 0

    def add_item(self, item):
        self._data[self.ptr_cnt] = item
        self.ptr_cnt = (self.ptr_cnt + 1) % self.max_size

    def add_batch(self, batch_item):
        length = batch_item.shape[0]
        end_length = length + self.ptr_cnt
        if end_length > self.max_size:
            cut = end_length - self.max_size
            self._data[self.ptr_cnt :] = batch_item[:-cut]
            self._data[:cut] = batch_item[-cut:]
            self.ptr_cnt = cut
        else:
            self._data[self.ptr_cnt : end_length] = batch_item
            self.ptr_cnt = end_length

    def __getitem__(self, key):
        return self._data[key]

    @property
    def data(self):
        return self._data

    def to(self, device):  # pylint: disable=invalid-name
        self._data = self._data.to(device)


if __name__ == "__main__":
    buffer = TorchRingBuffer((3,), max_size=10)
    data = th.arange(15).view(-1, 1).expand((15, 3))
    buffer.add_batch(data)
    print(buffer.data)
    print(buffer[:])
    print(buffer[[1, 2, 3]])
