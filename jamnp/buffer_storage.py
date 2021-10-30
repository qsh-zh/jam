from typing import Tuple

import numpy as np


class NpRingBuffer:
    def __init__(self, item_shape: Tuple, max_size: int = 10000):
        self.max_size = max_size
        self._data = np.zeros((max_size, *item_shape))
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


if __name__ == "__main__":
    buffer = NpRingBuffer((3,), max_size=10)
    data = np.repeat(np.arange(15).reshape(-1, 1), 3, axis=-1)
    buffer.add_batch(data)
    print(buffer.data)
    print(buffer[:])
    print(buffer[[1, 2, 3]])
