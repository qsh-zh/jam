from pathlib import Path
import os
import os.path as osp
import errno

__all__ = ["makedirs"]


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
