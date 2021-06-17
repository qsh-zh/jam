from .pytorch_util import *
from .pytorch_util import set_gpu_mode as _set_gpu_mode
from .optim import *


def set_gpu_mode(is_gpu=True, gpu_id=0):
    """ this is to fix the buf import variable from file.
    set jam pytorch utils device

    :param is_gpu: [description], defaults to True
    :type is_gpu: bool, optional
    :param gpu_id: [description], defaults to 0
    :type gpu_id: int, optional
    """
    import torch

    global device
    device = torch.device("cuda:" + str(gpu_id) if is_gpu else "cpu")
    _set_gpu_mode(is_gpu, gpu_id)
