import itertools
from collections.abc import Iterable
from functools import partial


def _args_single_param(item, _format="args"):
    key, value = item
    rtn = []
    if not isinstance(value, Iterable):
        value = [value]
    fist_element = value[0] if isinstance(value, list) else value
    if _format == "args":
        if isinstance(fist_element, bool):
            for cur_value in value:
                arg = "--" + key if cur_value else "--no_" + key
                rtn.append(arg)
            return rtn
        for i in value:
            rtn.append(f"--{key} {i}")
    elif _format == "hydra":
        for i in value:
            rtn.append(f"{key}={i}")
    else:
        raise RuntimeError(f"{_format} not supported yet")

    return rtn


def param_sweep(prefix=None, **kwargs):
    """generate cmds

    :param prefix: defaults to None
    : conditional "_format" -> args, hydra
    """
    _format = "args"
    if "_format" in kwargs:
        _format = kwargs.get("_format")
        del kwargs["_format"]
    assert _format in ["args", "hydra"]
    args_list = list(map(partial(_args_single_param, _format=_format), kwargs.items()))
    if prefix is not None:
        args_list.insert(0, [prefix])
    permutation = list(itertools.product(*args_list))
    return [" ".join(item) for item in permutation]
