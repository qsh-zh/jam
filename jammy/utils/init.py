import os
import pathlib
import resource
import sys

from jammy.utils.env import jam_getenv, jam_is_debug

from .meta import run_once


def release_syslim():
    if jam_getenv("SYSLIM", default="n", type="bool"):
        sys.setrecursionlimit(1000000)
        try:
            slim = 65536 * 1024
            resource.setrlimit(resource.RLIMIT_STACK, (slim, slim))
        except ValueError:
            pass


def tune_opencv():
    os.environ["OPENCV_OPENCL_RUNTIME"] = ""


def enable_ipdb():
    if jam_is_debug():
        if jam_getenv("IMPORT_ALL", "true", "bool"):
            from jammy.utils.debug import hook_exception_ipdb

            hook_exception_ipdb()


def init_main():
    release_syslim()
    tune_opencv()
    enable_ipdb()
    main_path()


@run_once
def main_path():
    if jam_getenv("proj_path") is None:
        os.environ["JAM_PROJ_PATH"] = str(pathlib.Path().absolute())
        return pathlib.Path().absolute()
    return jam_getenv("proj_path")
