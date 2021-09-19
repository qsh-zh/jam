import gpustat
import numpy as np


def select_gpu(mem_prior=1.0):
    mem_prior = np.clip(mem_prior, 0.0,1.0)
    query = gpustat.new_query()
    if len(query) == 0:
        raise RuntimeError("gpu not available")
    if len(query) == 1:
        return query[0].entry["index"]

    mem_list, utils_list = get_mem_util()
    mem, utils = np.array(mem_list), np.array(utils_list)
    weight = mem*mem_prior + utils * (1-mem_prior)
    least_utils_id = np.argmin(weight)

    return query[least_utils_id].entry["index"]


def gpu_by_util():
    query = gpustat.new_query()
    _, utils_list = get_mem_util()
    ids = np.argsort(utils_list)
    return [query[id_item].entry["index"] for id_item in ids]


def get_mem_util():
    query = gpustat.new_query()
    used_space_list = [1.0 * item.memory_used/item.memory_total for item in query]
    utils_list = [item.utilization for item in query]
    return used_space_list, utils_list
