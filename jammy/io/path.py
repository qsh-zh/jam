from pathlib import Path
import re
import os
import os.path as osp
import errno

__all__ = ["makedirs", "glob"]


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def glob(path, glob_mask="**/*", regex="", inverse=False, ftype="fd"):
    """glob files under path

    path: targeted search path
    glob_mask: simple glob pattern, use "*" when only scan current folder
    regex: re
    inverse: return not match if True
    ftype: "f" for file and "d" for dir

    Example:
        >>> glob("ckpt", regex="ckpt-[0-9]{4}\.pth", ftype="f")
        >>> ["ckpt/ckpt-0000.pth", "ckpt/ckpt-0001.pth", "ckpt/unet/ckpt-0001.pth"]
        >>> glob("ckpt", regex="ckpt-[0-9]{4}", ftype="d")
        >>> ["ckpt/ckpt-0000/", "ckpt/unet/ckpt-0000/"]
    """
    cur_path = Path(path)

    def select_type(_file):
        if "f" not in ftype:
            return osp.isdir(_file)
        if "d" not in ftype:
            return osp.isfile(_file)
        return True

    glob_files = [str(f) for f in cur_path.glob(glob_mask) if select_type(str(f))]
    if inverse:
        res = [f for f in glob_files if not re.search(regex, f)]
    else:
        res = [f for f in glob_files if re.search(regex, f)]
    return res
