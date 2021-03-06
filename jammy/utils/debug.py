import contextlib
import functools
import sys
import threading

from jammy.utils.env import jam_getenv

if jam_getenv("pdb", "ipdb").lower() == "ipdb":
    import ipdb as pdb
else:
    import pudb as pdb

__all__ = ["hook_exception_ipdb", "unhook_exception_ipdb", "exception_hook"]

# pylint: disable=invalid-name, redefined-builtin


def _custom_exception_hook(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback

        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


def hook_exception_ipdb():
    if not hasattr(_custom_exception_hook, "origin_hook"):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    assert hasattr(_custom_exception_hook, "origin_hook")
    sys.excepthook = _custom_exception_hook.origin_hook


@contextlib.contextmanager
def exception_hook(enable=True):
    if enable:
        hook_exception_ipdb()
        yield
        unhook_exception_ipdb()
    else:
        yield


def decorate_exception_hook(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with exception_hook():
            return func(*args, **kwargs)

    return wrapped


def _TimeoutEnterIpdbThread(locals_, cv, timeout):
    del locals_
    with cv:
        if not cv.wait(timeout):

            pdb.set_trace()


@contextlib.contextmanager
def timeout_ipdb(locals_, timeout=3):
    cv = threading.Condition()
    thread = threading.Thread(
        target=_TimeoutEnterIpdbThread, args=(locals_, cv, timeout)
    )
    thread.start()
    yield
    with cv:
        cv.notify_all()
