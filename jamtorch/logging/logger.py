import jammy.logging as logging
from jamtorch.utils.meta import is_master

__all__ = ["get_logger"]


def get_logger(*args, **kwargs):
    if is_master():
        return logging.get_logger(*args, **kwargs)
    else:
        return logging.fake_logger
