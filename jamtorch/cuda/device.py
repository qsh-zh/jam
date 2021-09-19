import torch

import jamtorch.prototype as jampt
from jammy.utils.gpu import select_gpu
from jamtorch.logging import get_logger

__all__ = [
    "set_best_device",
]

logger = get_logger()
# from gpustat import GPUStatCollection

# import subprocess
# def get_pname(id):
#     p = subprocess.Popen(["ps -o cmd= {}".format(id)], stdout=subprocess.PIPE, shell=True)
#     return str(p.communicate()[0])

# def cuda_process():
#     gpu_stat = GPUStatCollection.new_query()


def set_best_device(mem_prior=1.0):
    gpu_id = select_gpu(mem_prior)
    logger.critical(f"select device: CUDA{gpu_id} ")
    torch.cuda.set_device(gpu_id)
    jampt.set_gpu_mode(True, gpu_id)
    return gpu_id
