import numpy as np

__all__ = ["param_cnt"]


def param_cnt(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
