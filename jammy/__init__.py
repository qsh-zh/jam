from jammy.utils import git
from jammy.utils.env import jam_getenv
from jammy.utils.init import init_main

init_main()

del init_main

__hash__ = git.git_hash(__file__)

if jam_getenv("IMPORT_ALL", "true", "bool"):
    from jammy import io
    from jammy.utils.cnt import BufferCnt, CBCnt
    from jammy.utils.hyd import hyd_instantiate, hydpath
    from jammy.utils.hyd import instantiate as jam_instantiate
    from jammy.utils.hyd import link_hyd_run, update_cfg
    from jammy.utils.imp import load_class, load_module
    from jammy.utils.meta import (
        Singleton,
        assert_instance,
        assert_none,
        assert_notnone,
        cond_with,
        cond_with_group,
        decorator_with_optional_args,
        dict_deep_keys,
        dict_deep_kv,
        dict_deep_update,
        filter_exec,
        first_n,
        gofor,
        make_dummy_func,
        map_exec,
        map_exec_method,
        merge_iterable,
        method2func,
        notnone_property,
        run_once,
        stmap,
        synchronized,
        try_run,
    )


def get_jam_repo_git():
    """get git info running jammy

    :return: (jam_sha, jam_diff)
    """
    return git.log_repo(__file__)
