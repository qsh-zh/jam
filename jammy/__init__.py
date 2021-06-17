from jammy.utils.init import init_main
import jammy.utils.git as git
from jammy.utils.env import jam_getenv

init_main()

del init_main

__hash__ = git.git_hash(__file__)

if jam_getenv("IMPORT_ALL", "true", "bool"):
    from jammy import io
    from jammy.utils.hyd import hydpath, hyd_instantiate
    from jammy.utils.hyd import instantiate as jam_instantiate

    from jammy.utils.meta import (
            gofor,
            run_once, try_run,
            map_exec, filter_exec, first_n, stmap,
            method2func, map_exec_method,
            decorator_with_optional_args,
            cond_with, cond_with_group,
            merge_iterable,
            dict_deep_update, dict_deep_kv, dict_deep_keys,
            assert_instance, assert_none, assert_notnone,
            notnone_property, synchronized, make_dummy_func, Singleton
    )
    from jammy.utils.imp import load_module, load_class