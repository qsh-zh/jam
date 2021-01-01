import functools

import torch

__all__ = ["no_grad_func"]


def no_grad_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func
