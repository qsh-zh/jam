from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch as th

from jammy.utils.registry import CallbackRegistry

__all__ = ["merge", "merge_tree"]


def merge_dict(obj: List[Dict]):
    first_item = obj[0]
    rtn_dict = {}
    for key in first_item.keys():
        items = [cur_obj[key] for cur_obj in obj]
        rtn_dict[key] = merge(items)
    return rtn_dict


def merge_list(obj: List[List]):
    assert type(obj[0]) in [list]
    rtn = []
    for cur_list in obj:
        rtn.extend(cur_list)
    return rtn


def merge_basic(obj: List[Union[int, float, str, complex]]):
    assert type(obj[0]) in [int, float, str, complex]
    return obj


def merge_tuple(obj: List[Tuple]):
    length = len(obj[0])
    rtn = []
    for cur_idx in range(length):
        cur_items = [cur_obj[cur_idx] for cur_obj in obj]
        rtn.append(merge(cur_items))
    return tuple(rtn)


def merge_ndarray(obj: List[np.ndarray]):
    return np.concatenate(obj)


def merge_thtensor(obj: List[th.Tensor]):
    return th.cat(obj)


def _default_pytree_fallback(obj, *args, **kwargs):
    del args, kwargs
    if len(obj) == 0:
        return obj
    item_type = type(obj[0])
    raise ValueError('Unknown itme type: "{}".'.format(item_type))


class _PyTreeCallbackRegistry(CallbackRegistry):
    def dispatch(self, name, *args, **kwargs):
        return super().dispatch_direct(name, *args, **kwargs)


_pytree_registry = _PyTreeCallbackRegistry()
_pytree_registry.set_fallback_callback(_default_pytree_fallback)
_pytree_registry.register(int, merge_basic)
_pytree_registry.register(float, merge_basic)
_pytree_registry.register(complex, merge_basic)
_pytree_registry.register(np.ndarray, merge_ndarray)
_pytree_registry.register(th.Tensor, merge_thtensor)
_pytree_registry.register(tuple, merge_tuple)
_pytree_registry.register(dict, merge_dict)
_pytree_registry.register(list, merge_list)


def merge(obj: List[Any]):
    """
    List can only be in the leaf
    """
    if len(obj) == 0:
        return []
    item = obj[0]
    return _pytree_registry.dispatch(type(item), obj)


merge_tree = merge

if __name__ == "__main__":
    from jammy.utils.printing import stprint

    a = (1, np.arange(3), th.arange(4), {"a": 8})
    b = (2, np.arange(3) + 1, th.arange(4) + 5, {"a": 8})
    stprint(merge([a, b]))
