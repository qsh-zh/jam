import os
import sys

from jammy.cli.keyboard import str2bool

from .cache import cached_result

__all__ = ["jam_getenv", "jam_is_verbose", "jam_is_debug", "jam_setenv"]


def jam_getenv(
    name, default=None, type=None, prefix=None
):  # pylint: disable=redefined-builtin
    if prefix is None:
        prefix = "JAM_"

    value = os.getenv((prefix + name).upper(), default)

    if value is None:
        return None

    if type is None:
        return value
    elif type == "bool":
        return str2bool(value)
    else:
        return type(value)


def jam_setenv(key, value, prefix=None):
    if prefix is None:
        prefix = "JAM_"
    os.environ[prefix + key.upper()] = str(value)


@cached_result
def jam_get_dashdebug_arg():
    # Return True if there is a '-debug' or '--debug' arg in the argv.
    for value in sys.argv:
        if value in ("-debug", "--debug"):
            return True
    return False


@cached_result
def jam_is_verbose(default="n", prefix=None):
    return jam_getenv("verbose", default, type="bool", prefix=prefix)


@cached_result
def jam_is_debug(default="n", prefix=None):
    return jam_get_dashdebug_arg() or jam_getenv(
        "debug", default, type="bool", prefix=prefix
    )
