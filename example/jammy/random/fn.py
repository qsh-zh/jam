import torch
import numpy
import os
from jammy.utils.registry import Registry
# from jammy.logging import get_logger

def worker():
    pid = os.getpid()
    registry = Registry()
    registry.register("torch", lambda: torch.rand(4))
    registry.register("numpy", lambda: numpy.random.randn(4))
    for k, fn in registry.items():
        print(f"{pid:05d}\t {k}\t {fn()}")
