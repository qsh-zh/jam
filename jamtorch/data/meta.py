
__all__ = ["get_batch"]

def get_batch(data_loader):
    return next(iter(data_loader))