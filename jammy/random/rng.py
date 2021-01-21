import os
import random as sys_random

import numpy as np
import numpy.random as npr

from jammy.utils.defaults import defaults_manager
from jammy.utils.registry import Registry

__all__ = ['JamRandomState', 'get_default_rng', 'gen_seed', 'gen_rng', 'reset_global_seed']


class JamRandomState(npr.RandomState):
    def choice_list(self, list_, size=1, replace=False, p=None):
        """Efficiently draw an element from an list, if the rng is given, use it instead of the system one."""
        if size == 1:
            if type(list_) in (list, tuple):
                return list_[self.choice(len(list_), p=p)]
            return self.choice(list_, p=p)
        else:
            if type(list_) in (list, tuple):
                inds = self.choice(len(list_), size=size, replace=replace, p=p)
                return [list_[i] for i in inds]
            return self.choice(list_, size=size, replace=replace, p=p)

    def shuffle_list(self, list_):
        if type(list_) is list:
            sys_random.shuffle(list_, random=self.random_sample)
        else:
            self.shuffle(list_)

    def shuffle_multi(self, *arrs):
        length = len(arrs[0])
        for a in arrs:
            assert len(a) == length, 'non-compatible length when shuffling multiple arrays'

        inds = np.arange(length)
        self.shuffle(inds)
        return tuple(map(lambda x: x[inds], arrs))

    @defaults_manager.wrap_custom_as_default(is_local=True)
    def as_default(self):
        yield self


_rng = JamRandomState()


get_default_rng = defaults_manager.gen_get_default(JamRandomState, default_getter=lambda: _rng)


def gen_seed():
    return get_default_rng().randint(4294967296)


def gen_rng(seed=None):
    return JamRandomState(seed)


global_rng_registry = Registry()
global_rng_registry.register('jaminle', lambda: _rng.seed)
global_rng_registry.register('numpy', lambda: npr.seed)
global_rng_registry.register('sys', lambda: sys_random.seed)


def reset_global_seed(seed=None, verbose=False):
    if seed is None:
        seed = gen_seed()
    for k, seed_getter in global_rng_registry.items():
        if verbose:
            from jammy.logging import get_logger
            logger = get_logger()
            logger.critical('Reset random seed for: {} (pid={}, seed={}).'.format(k, os.getpid(), seed))
        seed_getter()(seed)


def _initialize_global_seed():
    seed = os.getenv('JAM_RANDOM_SEED', None)
    if seed is not None:
        reset_global_seed(seed)


_initialize_global_seed()
