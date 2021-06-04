import functools
from types import FunctionType

class LazyDict(dict):
    def __getitem__(self, key):
        obj = dict.__getitem__(self, key)
        if isinstance(obj, FunctionType) or isinstance(obj, functools.partial):
            obj = obj()
            dict.__setitem__(self, key, obj)
        return obj