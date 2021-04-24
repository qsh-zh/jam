import jammy.logging as logging
from jamtorch.utils.meta import is_master

try:
    import torch_xla.core.xla_model as xm

    IS_TPU = True
except Exception as e:
    IS_TPU = False

__all__ = ["get_logger"]


def get_logger(*args, **kwargs):
    if IS_TPU:
        if xm.get_ordinal == 0:
            return logging.get_logger(*args, **kwargs)
    elif is_master():
        return logging.get_logger(*args, **kwargs)

    return logging.fake_logger
