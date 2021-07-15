import gpustat
import numpy as np


def select_gpu():
    query = gpustat.new_query()
    if len(query) == 0:
        raise RuntimeError("gpu not available")
    if len(query) == 1:
        return query[0].entry["index"]

    _, utils_list = get_memfree_util()
    least_utils_id = np.argmin(utils_list)

    return query[least_utils_id].entry["index"]


def gpu_by_util():
    query = gpustat.new_query()
    _, utils_list = get_memfree_util()
    ids = np.argsort(utils_list)
    return [query[id_item].entry["index"] for id_item in ids]


def get_memfree_util():
    query = gpustat.new_query()
    free_space_list = [item.memory_free for item in query]
    utils_list = [item.utilization for item in query]
    return free_space_list, utils_list
