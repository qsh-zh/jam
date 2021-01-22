from .pytorch_util import *
from .pytorch_util import set_gpu_mode as _set_gpu_mode


def set_gpu_mode(mode, gpu_id=0):
    # this is to fix the buf import variable from file
    import torch

    global device
    device = torch.device("cuda:" + str(gpu_id) if mode else "cpu")
    _set_gpu_mode(mode, gpu_id)
