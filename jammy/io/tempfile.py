import tempfile as tempfile_lib
import contextlib
import os

__all__ = ["tempfile"]


@contextlib.contextmanager
def tempfile(mode="w+b", suffix="", prefix="tmp"):
    f = tempfile_lib.NamedTemporaryFile(
        mode, suffix=suffix, prefix=prefix, delete=False
    )
    yield f
    os.unlink(f.name)
