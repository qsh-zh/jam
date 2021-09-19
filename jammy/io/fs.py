import contextlib
import glob
import gzip
import os
import os.path as osp
import pickle
import platform
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import scipy.io as sio
import six
import yaml

from jammy.logging import get_logger
from jammy.utils.enum import JamEnum
from jammy.utils.filelock import FileLock
from jammy.utils.registry import CallbackRegistry, RegistryGroup

from .common import get_ext

logger = get_logger()

__all__ = [
    "as_file_descriptor",
    "fs_verbose",
    "set_fs_verbose",
    "open",
    "open_txt",
    "open_h5",
    "open_gz",
    "load",
    "load_txt",
    "load_h5",
    "load_pkl",
    "load_pklgz",
    "load_npy",
    "load_npz",
    "load_mat",
    "load_pth",
    "load_yaml",
    "dump",
    "dump_pkl",
    "dump_pklgz",
    "dump_npy",
    "dump_npz",
    "dump_mat",
    "dump_pth",
    "safe_dump",
    "compress",
    "compress_zip",
    "extract",
    "extract",
    "link",
    "mkdir",
    "lsdir",
    "remove",
    "locate_newest_file",
    "move",
    "copy",
    "replace",
    "io_function_registry",
    "latest_time",
]

sys_open = open


def as_file_descriptor(fd_or_fname, mode="r"):
    if isinstance(fd_or_fname, str):
        return sys_open(fd_or_fname, mode)  # pylint: disable=consider-using-with
    return fd_or_fname


def open_h5(file, mode, **kwargs):
    import h5py

    return h5py.File(file, mode, **kwargs)


def open_txt(file, mode, **kwargs):
    return sys_open(file, mode, **kwargs)  # pylint: disable=consider-using-with


def open_gz(file, mode):
    return gzip.open(file, mode)


def extract_zip(file, *args, **kwargs):
    with ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(*args, **kwargs)


def load_pkl(file, **kwargs):
    with as_file_descriptor(file, "rb") as f:
        try:
            return pickle.load(f, **kwargs)
        except UnicodeDecodeError:
            if "encoding" in kwargs:
                raise
            return pickle.load(f, encoding="latin1", **kwargs)


# pylint: disable=unused-argument
def load_pklgz(file, **kwargs):
    with open_gz(file, "rb") as f:
        return load_pkl(f)


def load_h5(file, **kwargs):
    return open_h5(file, "r", **kwargs)


def load_txt(file, **kwargs):
    with sys_open(file, "r", **kwargs) as f:
        return f.readlines()


def load_npy(file, **kwargs):
    return np.load(file, **kwargs)


def load_npz(file, **kwargs):
    return np.load(file, **kwargs)


def load_mat(file, **kwargs):
    return sio.loadmat(file, **kwargs)


def load_pth(file, **kwargs):
    import torch

    return torch.load(file, **kwargs)


def load_yaml(file, **kwargs):
    with sys_open(file, "r") as yamlfile:
        return yaml.load(yamlfile)


def dump_pkl(file, obj, **kwargs):
    with as_file_descriptor(file, "wb") as f:
        return pickle.dump(obj, f, **kwargs)


def dump_pklgz(file, obj, **kwargs):
    with open_gz(file, "wb") as f:
        return pickle.dump(obj, f)


def dump_yaml(file, obj, **kwargs):
    with sys_open(file, "w") as f:
        return yaml.dump(obj, f)


def dump_npy(file, obj, **kwargs):
    return np.save(file, obj)


def dump_npz(file, obj, **kwargs):
    return np.savez(file, obj)


def dump_mat(file, obj, **kwargs):
    return sio.savemat(file, obj, **kwargs)


def dump_pth(file, obj, **kwargs):
    import torch

    return torch.save(obj, file)


def compress_zip(  # pylint: disable=inconsistent-return-statements
    file, file_list, verbose=True, **kwargs
):
    from jammy.cli import yes_or_no

    with ZipFile(file, "w", ZIP_DEFLATED) as cur_zip:
        for l_file in file_list:
            try:
                cur_zip.write(l_file)
            except FileNotFoundError:
                is_continue = yes_or_no(f"Missing {l_file}, continue?")
                if is_continue:
                    pass
                else:
                    return None


class _IOFunctionRegistryGroup(RegistryGroup):
    __base_class__ = CallbackRegistry

    def dispatch(self, registry_name, file, *args, **kwargs):
        entry = get_ext(file)
        callback = self.lookup(
            registry_name, entry, fallback=True, default=_default_io_fallback
        )
        return callback(file, *args, **kwargs)


def _default_io_fallback(file, *args, **kwargs):
    raise ValueError('Unknown file extension: "{}".'.format(file))


io_function_registry = _IOFunctionRegistryGroup()
io_function_registry.register("open", ".txt", open_txt)
io_function_registry.register("open", ".h5", open_h5)
io_function_registry.register("open", ".gz", open_gz)
io_function_registry.register("open", "__fallback__", sys_open)

io_function_registry.register("load", ".pkl", load_pkl)
io_function_registry.register("load", ".pklgz", load_pklgz)
io_function_registry.register("load", ".txt", load_txt)
io_function_registry.register("load", ".h5", load_h5)
io_function_registry.register("load", ".npy", load_npy)
io_function_registry.register("load", ".npz", load_npz)
io_function_registry.register("load", ".mat", load_mat)
io_function_registry.register("load", ".pth", load_pth)
io_function_registry.register("load", ".cfg", load_pkl)
io_function_registry.register("load", ".yaml", load_yaml)
io_function_registry.register("load", ".yml", load_yaml)

io_function_registry.register("dump", ".pkl", dump_pkl)
io_function_registry.register("dump", ".pklgz", dump_pklgz)
io_function_registry.register("dump", ".npy", dump_npy)
io_function_registry.register("dump", ".npz", dump_npz)
io_function_registry.register("dump", ".mat", dump_mat)
io_function_registry.register("dump", ".pth", dump_pth)
io_function_registry.register("dump", ".cfg", dump_pkl)
io_function_registry.register("dump", ".yaml", dump_yaml)
io_function_registry.register("dump", ".yml", dump_yaml)


io_function_registry.register("extract", ".zip", extract_zip)
io_function_registry.register("compress", ".zip", compress_zip)


_fs_verbose = False

# pylint: disable=global-statement
@contextlib.contextmanager
def fs_verbose(mode=True):
    global _fs_verbose

    _fs_verbose, mode = mode, _fs_verbose
    yield
    _fs_verbose = mode


def set_fs_verbose(mode=True):
    global _fs_verbose
    _fs_verbose = mode


def open(file, mode, **kwargs):  # pylint: disable=redefined-builtin
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('Opening file: "{}", mode={}.'.format(file, mode))
    return io_function_registry.dispatch("open", file, mode, **kwargs)


def load(file, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('Loading data from file: "{}".'.format(file))
    return io_function_registry.dispatch("load", file, **kwargs)


def dump(file, obj, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('Dumping data to file: "{}".'.format(file))
    return io_function_registry.dispatch("dump", file, obj, **kwargs)


def compress(file, obj, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('compress data to file: "{}".'.format(file))
    return io_function_registry.dispatch("compress", file, obj, **kwargs)


def extract(file, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('extract data to file: "{}".'.format(file))
    return io_function_registry.dispatch("extract", file, **kwargs)


def safe_dump(fname, data, use_lock=True, use_temp=True, lock_timeout=10):
    temp_fname = "temp." + fname
    lock_fname = "lock." + fname

    def safe_dump_inner():
        if use_temp:
            dump(temp_fname, data)
            os.replace(temp_fname, fname)
            return True
        else:
            return dump(temp_fname, data)

    if use_lock:
        with FileLock(lock_fname, lock_timeout) as flock:
            if flock.is_locked:
                return safe_dump_inner()
            else:
                logger.warning("Cannot lock the file: {}.".format(fname))
                return False
    else:
        return safe_dump_inner()


def link(path_origin, *paths, use_relative_path=True):
    for item in paths:
        if os.path.exists(item):
            os.remove(item)
        if use_relative_path:
            src_path = os.path.relpath(path_origin, start=os.path.dirname(item))
        else:
            src_path = path_origin
        try:
            os.symlink(src_path, item)
        except FileExistsError:
            os.unlink(item)
            os.symlink(src_path, item)


def mkdir(path):
    return os.makedirs(path, exist_ok=True)


class LSDirectoryReturnType(JamEnum):
    BASE = "base"
    NAME = "name"
    REL = "rel"
    FULL = "full"
    REAL = "real"


def lsdir(dirname, pattern=None, return_type="full"):
    assert "*" in dirname or "?" in dirname or osp.isdir(dirname)

    return_type = LSDirectoryReturnType.from_string(return_type)
    if pattern is not None:
        files = glob.glob(osp.join(dirname, pattern), recursive=True)
    elif "*" in dirname:
        files = glob.glob(dirname)
    else:
        files = os.listdir(dirname)

    if return_type is LSDirectoryReturnType.BASE:
        return [osp.basename(f) for f in files]
    elif return_type is LSDirectoryReturnType.NAME:
        return [osp.splitext(osp.basename(f))[0] for f in files]
    elif return_type is LSDirectoryReturnType.REL:
        assert (
            "*" not in dirname and "?" not in dirname
        ), "Cannot use * or ? for relative paths."
        return [osp.relpath(f, dirname) for f in files]
    elif return_type is LSDirectoryReturnType.FULL:
        return files
    elif return_type is LSDirectoryReturnType.REAL:
        return [osp.realpath(osp.join(dirname, f)) for f in files]
    else:
        raise ValueError("Unknown lsdir return type: {}.".format(return_type))


def remove(file):
    if osp.exists(file):
        if osp.isdir(file):
            shutil.rmtree(file, ignore_errors=True)
        if osp.isfile(file):
            os.remove(file)


def copy(src, dst):
    if osp.exists(src):
        if osp.isdir(src):
            _copy = shutil.copytree
        if osp.isfile(src):
            _copy = shutil.copyfile

        _copy(src, dst)


def move(src, dst):
    if osp.exists(src):
        os.rename(src, dst)


def replace(src, dst):
    if osp.exists(src):
        if osp.exists(dst):
            remove(dst)
        os.replace(src, dst)


def locate_newest_file(dirname, pattern):
    files = lsdir(dirname, pattern, return_type="full")
    if len(files) == 0:
        return None
    return max(files, key=osp.getmtime)


def latest_time(fname):
    import datetime

    if platform.system() == "Windows":
        ftime = os.path.getctime(fname)
    else:
        stat = os.stat(fname)
        try:
            ftime = stat.st_birthtime
        except AttributeError:
            # probably on Linux.
            ftime = stat.st_mtime
    return datetime.datetime.fromtimestamp(ftime)
