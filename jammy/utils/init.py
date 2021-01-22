import os
import sys
import resource
from jammy.utils.env import jam_getenv, jam_is_debug, jam_getenv

def release_syslim():
    if jam_getenv('SYSLIM', default='n', type='bool'):
        sys.setrecursionlimit(1000000)
        try:
            slim = 65536 * 1024
            resource.setrlimit(resource.RLIMIT_STACK, (slim, slim))
        except ValueError:
            pass


def tune_opencv():
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''


def enable_ipdb():
    if jam_is_debug():
        if jam_getenv('IMPORT_ALL', 'true', 'bool'):
            from jammy.utils.debug import hook_exception_ipdb
            hook_exception_ipdb()


def init_main():
    release_syslim()
    tune_opencv()
    enable_ipdb()