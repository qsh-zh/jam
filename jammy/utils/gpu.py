import gpustat
import numpy as np


def is_gpu_free(ids=0, mem_thres=0.2, util_thres=0.3):
    if isinstance(ids, (list, tuple)):
        return all(is_gpu_free(item, mem_thres, util_thres) for item in ids)
    if isinstance(ids, int):
        query = gpustat.new_query()
        fixed_gpu = query[ids]
        used_mem = 1.0 * fixed_gpu.memory_used / fixed_gpu.memory_total
        if used_mem < mem_thres and fixed_gpu.utilization < util_thres:
            return True
        return False
    if ids == "all":
        query = gpustat.new_query()
        return is_gpu_free(list(range(len(query))), mem_thres, util_thres)
    raise RuntimeError(f"{ids} not supprted")


def gpu_by_weight(mem_prior=1.0):
    mem_prior = np.clip(mem_prior, 0.0, 1.0)
    query = gpustat.new_query()
    if len(query) == 0:
        raise RuntimeError("gpu not available")
    if len(query) == 1:
        return query[0].entry["index"]

    mem_list, utils_list = get_mem_util()
    mem, utils = np.array(mem_list), np.array(utils_list)
    weight = mem * mem_prior + utils * (1 - mem_prior)

    ids = np.argsort(weight)
    return [query[id_item].entry["index"] for id_item in ids]


def gpu_by_util():
    query = gpustat.new_query()
    _, utils_list = get_mem_util()
    ids = np.argsort(utils_list)
    return [query[id_item].entry["index"] for id_item in ids]


def get_mem_util():
    query = gpustat.new_query()
    used_space_list = [1.0 * item.memory_used / item.memory_total for item in query]
    utils_list = [item.utilization for item in query]
    return used_space_list, utils_list
